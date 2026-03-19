"""
Per-camera diagnostic visualisation for detection and single-camera tracking.

Saves for each requested camera:
    <output_dir>/<cam>_detections.mp4   — YOLO detections after ROI + car-class filter
    <output_dir>/<cam>_tracking.mp4     — single-camera tracks with colored IDs and trails

Also prints a per-camera statistics summary (detection rate, track count and
duration distribution) to help pinpoint where the pipeline is losing quality.

Usage
-----
# Quick single-camera check (first 500 frames):
python visualize_pipeline.py --cameras c001 --max-frames 500

# All cameras, full video:
python visualize_pipeline.py

# Max-overlap tracker instead of SORT:
python visualize_pipeline.py --tracker max_overlap --cameras c001 c002
"""

import argparse
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

from src.bounding_box import BoundingBox
from src.dataset import AICityDataset
from src.single_camera_tracker import SORTTracker, MaxOverlapTracker

# ---------------------------------------------------------------------------
# Defaults (mirror run_offline_mtmc.py)
# ---------------------------------------------------------------------------

DATA_ROOT    = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
SEQ_ID       = "S01"
CAMERAS      = ["c001", "c002", "c003", "c004", "c005"]
YOLO_WEIGHTS = "./yolov10s_coco.pt"
OUTPUT_DIR   = Path("output/viz")

CONF         = 0.45
MAX_AGE      = 10
IOU_THRESH   = 0.45
CAR_CLASS    = 0
TRAIL_LEN    = 40   # frames to keep in the trajectory trail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detection + tracking diagnostic videos")
    p.add_argument("--data-root",     default=DATA_ROOT)
    p.add_argument("--seq",           default=SEQ_ID)
    p.add_argument("--cameras",       nargs="+", default=CAMERAS)
    p.add_argument("--yolo-weights",  default=YOLO_WEIGHTS)
    p.add_argument("--tracker",       choices=["sort", "max_overlap"], default="sort")
    p.add_argument("--conf",          type=float, default=CONF)
    p.add_argument("--max-age",       type=int,   default=MAX_AGE)
    p.add_argument("--iou-threshold", type=float, default=IOU_THRESH)
    p.add_argument("--trail-len",     type=int,   default=TRAIL_LEN,
                   help="Number of past frames to show in the tracking trail")
    p.add_argument("--max-frames",    type=int,   default=None,
                   help="Stop after this many frames per camera (useful for quick checks)")
    p.add_argument("--output-dir",    default=str(OUTPUT_DIR))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _track_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, visually distinct BGR color for a given track ID."""
    hue = int((track_id * 47) % 180)
    hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _draw_detection_frame(
    frame: np.ndarray,
    detections: list[tuple[BoundingBox, float]],
    roi_mask: np.ndarray | None,
    frame_idx: int,
) -> np.ndarray:
    out = frame.copy()

    # Tint out-of-ROI area in dark red so the ROI boundary is visible.
    if roi_mask is not None:
        overlay = out.copy()
        overlay[roi_mask == 0] = (0, 0, 120)
        cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

    for bbox, conf in detections:
        x1, y1 = int(bbox.left),  int(bbox.top)
        x2, y2 = int(bbox.right), int(bbox.bottom)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)

    label = f"Frame {frame_idx:5d}  |  Dets: {len(detections)}"
    cv2.putText(out, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _draw_tracking_frame(
    frame: np.ndarray,
    active: list[tuple[int, BoundingBox]],
    trails: dict[int, list[tuple[int, int, int]]],
    frame_idx: int,
    trail_len: int,
) -> np.ndarray:
    out = frame.copy()
    cutoff = frame_idx - trail_len

    for track_id, bbox in active:
        color = _track_color(track_id)
        x1, y1 = int(bbox.left),  int(bbox.top)
        x2, y2 = int(bbox.right), int(bbox.bottom)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"T{track_id}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        # Trail: draw line segments between successive past centers.
        pts = [(cx, cy) for fi, cx, cy in trails.get(track_id, [])
               if cutoff <= fi <= frame_idx]
        for i in range(1, len(pts)):
            alpha = i / max(len(pts) - 1, 1)          # fade older segments
            faded = tuple(int(c * (0.3 + 0.7 * alpha)) for c in color)
            cv2.line(out, pts[i - 1], pts[i], faded, 1, cv2.LINE_AA)

    label = f"Frame {frame_idx:5d}  |  Active tracks: {len(active)}"
    cv2.putText(out, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Video writer helper
# ---------------------------------------------------------------------------

def _make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    return writer


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------

def process_camera(
    camera,
    detector: YOLO,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    cam_id = camera.id
    video_path = camera.video_path
    roi_path   = camera.roi_path

    # --- Load ROI mask ---
    roi_mask = None
    if roi_path and Path(roi_path).exists():
        raw = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if raw is not None:
            _, roi_mask = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
    else:
        print(f"  [WARN] No ROI found for {cam_id}, processing full frame.")

    # --- Open video ---
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = min(total_frames, args.max_frames) if args.max_frames else total_frames
    print(f"  {cam_id}: {n_frames} frames  {width}×{height}  {fps:.1f} fps")

    # --- Build tracker ---
    if args.tracker == "sort":
        tracker = SORTTracker(max_age=args.max_age, iou_threshold=args.iou_threshold)
    else:
        tracker = MaxOverlapTracker(max_age=args.max_age, iou_threshold=args.iou_threshold)

    # --- Output paths ---
    det_path   = output_dir / f"{cam_id}_detections.mp4"
    track_path = output_dir / f"{cam_id}_tracking.mp4"

    # -----------------------------------------------------------------------
    # PASS 1 — Detection + tracking; write detection video
    # -----------------------------------------------------------------------
    det_writer = _make_writer(det_path, fps, width, height)

    # Statistics accumulators
    det_counts: list[int] = []
    all_detections: dict[int, list[tuple[BoundingBox, float]]] = {}  # frame → [(bbox, conf)]

    for frame_idx in tqdm(range(n_frames), desc=f"  {cam_id} detect+track", leave=False):
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO detection
        results  = detector(frame, verbose=False, conf=args.conf)[0]
        raw_dets = []
        for box in results.boxes:
            if int(box.cls[0]) != CAR_CLASS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())

            # ROI filter (same logic as the main pipeline)
            if roi_mask is not None:
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                h, w = roi_mask.shape
                corners = [(ix1, iy1), (ix1, iy2), (ix2, iy1), (ix2, iy2)]
                if any(0 <= cx < w and 0 <= cy < h and roi_mask[cy, cx] < 255
                       for cx, cy in corners):
                    continue

            raw_dets.append((BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf), conf))

        bboxes = [bd[0] for bd in raw_dets]
        tracker.update(bboxes, frame, frame_idx)

        all_detections[frame_idx] = raw_dets
        det_counts.append(len(raw_dets))

        det_writer.write(_draw_detection_frame(frame, raw_dets, roi_mask, frame_idx))

    det_writer.release()
    print(f"    Detection video → {det_path}")

    # Finalise tracks
    finished_tracks = tracker.finalize()

    # -----------------------------------------------------------------------
    # Build per-frame tracking lookup and trail data
    # -----------------------------------------------------------------------
    # frame_idx → list of (track_id, bbox)
    frame_to_active: dict[int, list[tuple[int, BoundingBox]]] = defaultdict(list)
    # track_id → list of (frame_idx, cx, cy) for trail rendering
    trails: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for track in finished_tracks:
        for obs in track.history:
            if obs.frame_idx >= n_frames:
                continue
            bbox = obs.bbox
            cx = int((bbox.left + bbox.right) / 2)
            cy = int((bbox.top  + bbox.bottom) / 2)
            frame_to_active[obs.frame_idx].append((track.track_id, bbox))
            trails[track.track_id].append((obs.frame_idx, cx, cy))

    # -----------------------------------------------------------------------
    # PASS 2 — Write tracking video
    # -----------------------------------------------------------------------
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    track_writer = _make_writer(track_path, fps, width, height)

    for frame_idx in tqdm(range(n_frames), desc=f"  {cam_id} tracking video", leave=False):
        ok, frame = cap.read()
        if not ok:
            break
        active = frame_to_active.get(frame_idx, [])
        track_writer.write(_draw_tracking_frame(frame, active, trails, frame_idx, args.trail_len))

    track_writer.release()
    cap.release()
    print(f"    Tracking video  → {track_path}")

    # -----------------------------------------------------------------------
    # Per-camera statistics
    # -----------------------------------------------------------------------
    durations = [len(t.history) for t in finished_tracks]
    avg_dets  = float(np.mean(det_counts)) if det_counts else 0.0
    print(f"\n    ── {cam_id} statistics ──")
    print(f"    Frames processed      : {n_frames}")
    print(f"    Avg detections/frame  : {avg_dets:.1f}")
    print(f"    Frames with 0 dets    : {sum(1 for c in det_counts if c == 0)}")
    print(f"    Total tracks          : {len(finished_tracks)}")
    if durations:
        print(f"    Track duration (frames): "
              f"min={min(durations)}  mean={np.mean(durations):.1f}  "
              f"max={max(durations)}  median={int(np.median(durations))}")
        short = sum(1 for d in durations if d < 5)
        print(f"    Tracks < 5 frames     : {short}  ({100*short/len(durations):.0f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO …")
    detector = YOLO(args.yolo_weights)

    print("Loading dataset …")
    ds  = AICityDataset(args.data_root)
    seq = ds[args.seq]

    for cam_id in args.cameras:
        if cam_id not in seq.cameras:
            print(f"[WARN] Camera {cam_id} not found in sequence {args.seq}, skipping.")
            continue
        print(f"\n─── Camera {cam_id} ───")
        process_camera(seq[cam_id], detector, args, output_dir)

    print(f"\nAll videos saved to {output_dir}/")


if __name__ == "__main__":
    main()
