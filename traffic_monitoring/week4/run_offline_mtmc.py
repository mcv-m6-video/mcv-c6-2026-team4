"""
Offline multi-camera tracking pipeline with animated world-map visualisation.

Usage examples
--------------
# Appearance-only (ReID), sequence S01, all cameras:
python run_offline_mtmc.py

# Geometry + appearance, custom sequence / cameras:
python run_offline_mtmc.py --seq S03 --cameras c013 c014 c015 \
    --w-spatial 0.3 --w-temporal 0.2 --spatial-scale 0.01 --temporal-scale 30

# Skip ReID (colour histogram only):
python run_offline_mtmc.py --extractor histogram

# SORT tracker instead of max-overlap:
python run_offline_mtmc.py --tracker sort
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless-safe; switch to "TkAgg" / "Qt5Agg" for interactive
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
if not hasattr(np, "asfarray"):          # removed in NumPy 2.0, needed by motmetrics
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from offline_multicamera_tracking import OfflineMulticameraTracker
from src.camera_graph import CameraGraph
from src.dataset import AICityDataset
from src.multi_camera_associator import CombinedAssociator, GlobalTrack
from src.reid_feature_extractor import ColorHistogramExtractor, FtNetExtractor
from src.single_camera_tracker import MaxOverlapTracker, SORTTracker
from src.world_and_camera_tracking import ContactPoint

# Re-use helpers from the camera-map script
from try_to_create_camera_map import _iqr_filter, _palette, _roi_white_pixels

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_ROOT    = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
SEQ_ID       = "S01"
CAMERAS      = ["c001", "c002", "c003", "c004", "c005"]
YOLO_WEIGHTS = "./yolov10s_coco.pt"
REID_WEIGHTS = "./src/net_19.pth"
OUTPUT_DIR   = Path("output/tracking")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline multi-camera tracker")

    # Dataset
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--seq", default=SEQ_ID, help="Sequence id, e.g. S01")
    p.add_argument("--cameras", nargs="+", default=CAMERAS)

    # Weights
    p.add_argument("--yolo-weights", default=YOLO_WEIGHTS)
    p.add_argument("--reid-weights", default=REID_WEIGHTS)

    # SCT
    p.add_argument("--tracker", choices=["sort", "max_overlap"], default="sort")
    p.add_argument("--max-age", type=int, default=10)
    p.add_argument("--iou-threshold", type=float, default=0.45)
    p.add_argument("--conf", type=float, default=0.45)

    # Feature extractor
    p.add_argument("--extractor", choices=["reid", "histogram"], default="reid",
                   help="'reid' uses ft_net; 'histogram' uses HSV histograms (no GPU needed)")
    p.add_argument("--n-crops", type=int, default=5)

    # Association
    p.add_argument("--w-appearance", type=float, default=1.0)
    p.add_argument("--w-spatial",    type=float, default=0.0)
    p.add_argument("--w-temporal",   type=float, default=0.0)
    p.add_argument("--w-velocity",   type=float, default=0.0)
    p.add_argument("--match-threshold", type=float, default=0.4)
    p.add_argument("--spatial-scale",   type=float, default=1.0)
    p.add_argument("--temporal-scale",  type=float, default=1.0)
    p.add_argument("--no-same-camera-filter", action="store_true")
    p.add_argument("--feature-momentum", type=float, default=0.8)
    p.add_argument("--debug-costs", action="store_true",
                   help="Print per-pair cost breakdown during association (very verbose)")

    # Camera connectivity graph
    p.add_argument("--camera-graph", action="store_true",
                   help="Enable camera connectivity graph to gate implausible transitions")
    p.add_argument("--v-min", type=float, default=0.00001,
                   help="Min vehicle speed in world units/s (tune to your homography scale)")
    p.add_argument("--v-max", type=float, default=0.0002,
                   help="Max vehicle speed in world units/s")
    p.add_argument("--max-cam-distance", type=float, default=None,
                   help="Cameras whose ROI centroids are further apart than this are not connected")

    # Evaluation
    p.add_argument("--gt-path", default=None,
                   help="Path to ground-truth .txt file (AI City format). "
                        "If provided, evaluates the tracker after running.")

    # Animation
    p.add_argument("--anim-dt",     type=float, default=0.5,
                   help="Real-time seconds between animation frames")
    p.add_argument("--anim-fps",    type=int,   default=10,
                   help="Playback FPS of the output video")
    p.add_argument("--trail-secs",  type=float, default=5.0,
                   help="How many past seconds of trajectory to show")
    p.add_argument("--max-frames",  type=int,   default=None,
                   help="Render only the first N animation frames (for quick debug)")
    p.add_argument("--output-dir",  default=str(OUTPUT_DIR))

    return p.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_tracker_factory(args):
    def factory():
        if args.tracker == "sort":
            return SORTTracker(max_age=args.max_age, iou_threshold=args.iou_threshold)
        return MaxOverlapTracker(max_age=args.max_age, iou_threshold=args.iou_threshold)
    return factory


def build_feature_extractor(args, device):
    if args.extractor == "reid":
        reid_path = Path(args.reid_weights)
        if not reid_path.exists():
            print(f"[WARN] ReID weights not found at {reid_path}. Falling back to histogram.")
            return ColorHistogramExtractor(n_crops=args.n_crops)
        return FtNetExtractor(reid_path, device=device, n_crops=args.n_crops)
    return ColorHistogramExtractor(n_crops=args.n_crops)


def run_pipeline(args) -> list[GlobalTrack]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset …")
    ds = AICityDataset(args.data_root)
    seq = ds[args.seq]
    cameras = [seq[cam_id] for cam_id in args.cameras if cam_id in seq.cameras]
    if not cameras:
        raise ValueError(f"No valid cameras found in sequence {args.seq} for {args.cameras}")
    print(f"Sequence {args.seq}: {[c.id for c in cameras]}")

    print("Loading YOLO …")
    detector = YOLO(args.yolo_weights)

    print(f"Building feature extractor ({args.extractor}) …")
    extractor = build_feature_extractor(args, device)

    graph = None
    if args.camera_graph:
        print("Building camera connectivity graph …")
        graph = CameraGraph.from_cameras(
            cameras,
            v_min=args.v_min,
            v_max=args.v_max,
            max_distance=args.max_cam_distance,
        )
        print(graph.summary())

    associator = CombinedAssociator(
        w_appearance=args.w_appearance,
        w_spatial=args.w_spatial,
        w_temporal=args.w_temporal,
        w_velocity=args.w_velocity,
        match_threshold=args.match_threshold,
        same_camera_filter=not args.no_same_camera_filter,
        camera_graph=graph,
        feature_momentum=args.feature_momentum,
        spatial_scale=args.spatial_scale,
        temporal_scale=args.temporal_scale,
        debug_costs=args.debug_costs,
    )

    tracker = OfflineMulticameraTracker(
        tracker_factory=build_tracker_factory(args),
        detector=detector,
        associator=associator,
        contact_strategy=ContactPoint.BOTTOM_CENTER,
        feature_extractor=extractor,
        confidence=args.conf,
    )

    print("\nRunning per-camera tracking …")
    global_tracks = tracker.track(cameras)
    print(f"\nGlobal tracks: {len(global_tracks)}")
    return global_tracks, seq


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def global_tracks_to_predictions_df(global_tracks: list[GlobalTrack]) -> pd.DataFrame:
    """
    Convert GlobalTracks to the AI City Challenge submission format.

    Columns: CameraId (int), Id, FrameId (1-based), X, Y, Width, Height, Xworld, Yworld.
    """
    rows = []
    for gt in global_tracks:
        for wt in gt.tracklets:
            # "c001" → 1
            try:
                cam_int = int(wt.camera_id.lstrip("c"))
            except ValueError:
                cam_int = hash(wt.camera_id) % 100000

            for wo in wt.world_observations:
                bbox = wo.bbox
                rows.append({
                    "CameraId": cam_int,
                    "Id":       gt.global_id,
                    "FrameId":  wo.frame_idx + 1,           # 0-based → 1-based
                    "X":        int(bbox.left),              # eval indexes roi[y, x] → must be int
                    "Y":        int(bbox.top),
                    "Width":    int(bbox.right - bbox.left),
                    "Height":   int(bbox.bottom - bbox.top),
                    "Xworld":   wo.world_point.x,
                    "Yworld":   wo.world_point.y,
                })

    cols = ["CameraId", "Id", "FrameId", "X", "Y", "Width", "Height", "Xworld", "Yworld"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def evaluate(global_tracks: list[GlobalTrack], args, output_dir: Path) -> None:
    """
    Evaluate global tracks against the ground truth using src/eval.py.

    The eval infrastructure:
      - Filters predictions to multi-camera tracks automatically (removeOutliersSingleCam)
      - Filters by ROI (removeOutliersROI) using data_root/<seq>/c<NNN>/roi.jpg
      - Computes IDF1, IDP, IDR + optionally HOTA/DetA/AssA
    """
    from src.eval import readData, eval as aicity_eval, print_results, get_results

    print("\n--- Evaluation ---")
    gt_df = readData(args.gt_path)

    pred_df = global_tracks_to_predictions_df(global_tracks)
    if pred_df.empty:
        print("[WARN] No predictions to evaluate.")
        return

    # Save predictions for inspection / offline re-evaluation
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / f"{args.seq}_predictions.txt"
    pred_df.to_csv(str(pred_path), sep=" ", index=False, header=False)
    print(f"Predictions saved → {pred_path}")
    print(f"  {len(pred_df)} raw detection-level rows  |  "
          f"{pred_df['Id'].nunique()} global IDs  |  "
          f"{pred_df['CameraId'].nunique()} cameras")

    try:
        summary = aicity_eval(
            gt_df,
            pred_df,
            dstype=f"train/{args.seq}",  # → roidir/train/S01/c001/roi.jpg
            roidir=args.data_root,
        )
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}")
        return

    print()
    print_results(summary)

    metrics = get_results(summary)
    if metrics:
        print("\nSummary:")
        for k, v in metrics.items():
            if v is not None:
                print(f"  {k.upper():6s}: {v:6.2f}")


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _build_scatter_background(seq, camera_ids: list[str]) -> dict:
    """
    Project each camera's ROI pixels to world coordinates (same convention as
    try_to_create_camera_map.py: uses inv(H) for pixel→world).
    Returns {cam_id: (world_pts, color)}.
    """
    cam_items = [(cid, seq.cameras[cid]) for cid in camera_ids if cid in seq.cameras]
    colors = _palette(len(cam_items))
    scatter: dict = {}

    for (cam_id, cam), color in zip(cam_items, colors):
        img_pts = _roi_white_pixels(cam.roi_path, step=5)
        if len(img_pts) == 0:
            continue
        H_inv = np.linalg.inv(cam.calibration.homography)
        ones = np.ones((len(img_pts), 1))
        pts_h = np.hstack([img_pts, ones])
        world_h = (H_inv @ pts_h.T).T
        w = world_h[:, 2]
        valid = np.abs(w) > 1e-9
        world_pts = world_h[valid, :2] / w[valid, np.newaxis]
        world_pts = _iqr_filter(world_pts, 1.5)
        if len(world_pts) > 0:
            scatter[cam_id] = (world_pts, color)

    return scatter


def _precompute_fused_positions(
    global_tracks: list[GlobalTrack],
    times: np.ndarray,
) -> dict[int, list]:
    """
    For each global track, precompute the fused world position at every
    animation time step using GlobalTrack.fused_position_at(t).

    Using the fused position (inverse-variance-weighted average across all
    camera tracklets) rather than raw per-observation positions prevents the
    trail from zigzagging when multiple cameras with overlapping FOVs see the
    same car at slightly different projected world coordinates.

    Returns {global_id: [WorldPoint | None, ...]} aligned to `times`.
    None entries mean no camera covers the car at that time step (gap between
    cameras); these are rendered as breaks in the trail line.
    """
    result = {}
    for gt in global_tracks:
        positions = [gt.fused_position_at(float(t)) for t in times]
        if any(p is not None for p in positions):
            result[gt.global_id] = positions
    return result


def _collect_velocities(
    global_tracks: list[GlobalTrack],
) -> dict[int, list[tuple]]:
    """
    For each global track, collect (timestamp, vx, vy) from raw observations
    (within each tracklet) for the velocity arrow display.
    """
    result = {}
    for gt in global_tracks:
        entries = []
        for wt in gt.tracklets:
            for i, wo in enumerate(wt.world_observations):
                if wo.timestamp is None:
                    continue
                vel = wt.velocity_at(i)
                if vel is not None:
                    entries.append((wo.timestamp, float(vel[0]), float(vel[1])))
        entries.sort(key=lambda e: e[0])
        if entries:
            result[gt.global_id] = entries
    return result


def _axis_limits(scatter: dict, margin: float = 0.05):
    all_pts = np.vstack([pts for pts, _ in scatter.values()])
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate(
    global_tracks: list[GlobalTrack],
    seq,
    camera_ids: list[str],
    output_dir: Path,
    seq_id: str,
    dt: float = 0.5,
    fps: int = 10,
    trail_secs: float = 5.0,
    max_frames: int | None = None,
) -> None:
    print("\nBuilding world-map background …")
    scatter = _build_scatter_background(seq, camera_ids)
    if not scatter:
        print("[ERROR] No ROI scatter data — cannot animate.")
        return

    # Collect raw timestamps to build the time grid
    raw_velocities = _collect_velocities(global_tracks)
    all_ts = [ts for entries in raw_velocities.values() for ts, _, _ in entries]
    if not all_ts:
        print("[WARN] No timestamped observations found — cannot animate.")
        return

    t_min, t_max = min(all_ts), max(all_ts)
    times = np.arange(t_min, t_max + dt, dt)
    if max_frames is not None:
        times = times[:max_frames]

    # Precompute fused positions at every time step (avoids multi-camera zigzag)
    print(f"Precomputing fused positions for {len(global_tracks)} tracks × {len(times)} steps …")
    fused_positions = _precompute_fused_positions(global_tracks, times)
    if not fused_positions:
        print("[WARN] No fused positions computed — cannot animate.")
        return

    # Per-track colours (cycle tab20)
    track_ids = sorted(fused_positions.keys())
    palette = _palette(max(len(track_ids), 1))
    track_color = {gid: palette[i % len(palette)] for i, gid in enumerate(track_ids)}

    # --------------- figure setup ---------------
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect("equal")
    xlim, ylim = _axis_limits(scatter)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("World X (dataset units)", fontsize=9)
    ax.set_ylabel("World Y (dataset units)", fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    # Static scatter (drawn once, not touched by animation loop)
    for cam_id, (world_pts, color) in scatter.items():
        ax.scatter(
            world_pts[:, 0], world_pts[:, 1],
            s=0.3, color=color[:3], alpha=0.2,
            linewidths=0, rasterized=True, zorder=1,
        )

    title = ax.set_title("", fontsize=11)
    dynamic: list = []   # artists cleared each frame

    # --------------- per-frame update ---------------
    def update(frame_idx: int):
        nonlocal dynamic
        for art in dynamic:
            try:
                art.remove()
            except Exception:
                pass
        dynamic = []

        t = times[frame_idx]
        title.set_text(
            f"Multi-Camera Tracking — Sequence {seq_id}   t = {t:.1f} s"
            f"   ({frame_idx + 1}/{len(times)})"
        )

        for gid, positions in fused_positions.items():
            color = track_color[gid][:3]

            # -- Trajectory trail (fused positions, nan = gap between cameras) --
            trail_xs, trail_ys = [], []
            current_wp = None
            for j in range(frame_idx + 1):
                if times[j] < t - trail_secs:
                    continue
                wp = positions[j]
                if wp is None:
                    trail_xs.append(np.nan)
                    trail_ys.append(np.nan)
                else:
                    trail_xs.append(wp.x)
                    trail_ys.append(wp.y)
                    current_wp = wp

            if current_wp is None:
                continue  # track not yet visible at this time

            # Strip leading nans before drawing
            while trail_xs and np.isnan(trail_xs[0]):
                trail_xs.pop(0)
                trail_ys.pop(0)

            if len(trail_xs) >= 2:
                (line,) = ax.plot(trail_xs, trail_ys, color=color, linewidth=1.5,
                                  alpha=0.85, zorder=2)
                dynamic.append(line)

            cx, cy = current_wp.x, current_wp.y

            # -- Current position dot --
            (dot,) = ax.plot(cx, cy, "o", color=color, markersize=7,
                             markeredgecolor="white", markeredgewidth=0.8,
                             zorder=4)
            dynamic.append(dot)

            # -- Velocity arrow: most recent raw velocity at or before t --
            vx, vy = 0.0, 0.0
            vel_entries = raw_velocities.get(gid, [])
            past_vel = [(ts, vx_, vy_) for ts, vx_, vy_ in vel_entries if ts <= t]
            if past_vel:
                _, vx, vy = past_vel[-1]
            speed = np.hypot(vx, vy)
            if speed > 0:
                q = ax.quiver(
                    cx, cy, vx, vy,
                    color=color, angles="xy", scale_units="xy", scale=1.0,
                    width=0.003, headwidth=4, headlength=4,
                    zorder=3,
                )
                dynamic.append(q)

            # -- ID label + speed --
            label = f" {gid}"
            if speed > 0:
                label += f"\n {speed:.3f} u/s"
            txt = ax.text(
                cx, cy, label,
                fontsize=7, color=color, fontweight="bold",
                va="bottom", zorder=5,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"),
            )
            dynamic.append(txt)

        return dynamic

    anim = animation.FuncAnimation(
        fig, update, frames=len(times),
        interval=1000 / fps, blit=False,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = output_dir / f"{seq_id}_tracking.mp4"
    gif_path = output_dir / f"{seq_id}_tracking.gif"

    print(f"Rendering {len(times)} frames …")
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000,
                                        extra_args=["-pix_fmt", "yuv420p"])
        anim.save(str(mp4_path), writer=writer,
                  progress_callback=lambda i, n: print(f"  frame {i+1}/{n}", end="\r"))
        print(f"\nSaved → {mp4_path}")
    except Exception as e:
        print(f"FFmpeg unavailable ({e}), falling back to GIF …")
        writer = animation.PillowWriter(fps=fps)
        anim.save(str(gif_path), writer=writer,
                  progress_callback=lambda i, n: print(f"  frame {i+1}/{n}", end="\r"))
        print(f"\nSaved → {gif_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)

    global_tracks, seq = run_pipeline(args)

    if args.gt_path:
        evaluate(global_tracks, args, output_dir)

    animate(
        global_tracks,
        seq,
        args.cameras,
        output_dir,
        seq_id=args.seq,
        dt=args.anim_dt,
        fps=args.anim_fps,
        trail_secs=args.trail_secs,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
