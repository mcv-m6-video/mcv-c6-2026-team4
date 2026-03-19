"""
World-space photo mosaic for a sequence.

For each pixel in the output canvas (world/GPS space), the script:
  1. Converts the canvas pixel to world coordinates (lat, lon)
  2. Projects that world point to every camera via H (world → pixel)
  3. Checks bounds and ROI mask in the camera image
  4. Among valid cameras, picks the one whose ROI centroid is closest
     to the world point ("closest camera wins")
  5. Samples the mean video frame at that camera pixel

The world extent is derived from the 95th-percentile bounding box of
all projected ROI pixels, cutting off vanishing-line outliers.

Output
------
  output/world_viz/<seq>_world_mosaic.png       raw image  (large, croppable)
  output/world_viz/<seq>_world_mosaic_fig.png   same + GPS axes for reference

Usage
-----
    python make_world_mosaic.py                    # S01, 3000px long side
    python make_world_mosaic.py --seq S03
    python make_world_mosaic.py --seq S04 --size 5000
    python make_world_mosaic.py --seq S01 --size 6000 --n-frames 80
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import matplotlib.patches as mpatches

from src.dataset import AICityDataset
from try_to_create_camera_map import _iqr_filter, _palette, _roi_white_pixels

# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_ROOT        = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
OUTPUT_DIR       = Path("output/world_viz")
N_FRAMES         = 60     # frames to average per camera
CANVAS_LONG_SIDE = 3000   # pixels on the longest side
PERCENTILE       = 98.0   # clip far-field outliers beyond this percentile
BBOX_MARGIN      = 0.10   # extra margin around the percentile bounding box


# ── Helpers ───────────────────────────────────────────────────────────────────

def _project_roi_to_world(cam, step: int = 5) -> np.ndarray:
    """
    Return (N, 2) world (lat, lon) points for every white ROI pixel.
    H in calibration.txt is world→pixel, so we invert it here.
    No IQR filter applied — used for bounding box computation.
    """
    img_pts = _roi_white_pixels(cam.roi_path, step)   # (M, 2)  (col, row)
    if len(img_pts) == 0:
        return np.empty((0, 2))

    H_inv = np.linalg.inv(cam.calibration.homography)
    ones  = np.ones((len(img_pts), 1))
    wh    = (H_inv @ np.hstack([img_pts, ones]).T).T   # (M, 3)
    d     = wh[:, 2]
    valid = np.abs(d) > 1e-9
    return wh[valid, :2] / d[valid, np.newaxis]


def _roi_center_image(cam) -> tuple[float, float]:
    """
    Return the (u, v) centroid of the ROI white pixels in image space.
    Used as the "camera aim point" for the closest-camera metric.
    """
    roi = cv2.imread(str(cam.roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return 0.0, 0.0
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    rows, cols = np.where(binary > 0)
    if len(rows) == 0:
        return float(roi.shape[1] / 2), float(roi.shape[0] / 2)
    return float(cols.mean()), float(rows.mean())   # (u=col, v=row)


def _mean_frame(video_path: Path, n: int) -> np.ndarray | None:
    """Average n evenly-spaced frames. Returns BGR uint8 or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    acc, count = None, 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        acc = frame.astype(np.float32) if acc is None else acc + frame.astype(np.float32)
        count += 1
    cap.release()
    if acc is None:
        return None
    return np.clip(acc / count, 0, 255).astype(np.uint8)


# ── Main ──────────────────────────────────────────────────────────────────────

def make_world_mosaic(
    seq,
    output_dir: Path = OUTPUT_DIR,
    n_frames:         int   = N_FRAMES,
    canvas_long_side: int   = CANVAS_LONG_SIDE,
    percentile:       float = PERCENTILE,
    bbox_margin:      float = BBOX_MARGIN,
    roi_step:         int   = 5,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Project ROI pixels to world, compute bounding box ─────────────────
    print(f"\n[{seq.id}] Projecting ROI pixels to world space …")
    valid_cams: list[str] = []

    # Per-camera bounding boxes (99th-pct clip to remove vanishing-line outliers)
    # then union across all cameras.  No IQR filter here — that was the source
    # of double-filtering that made the bbox too small.
    lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []

    for cam_id, cam in sorted(seq.cameras.items()):
        pts = _project_roi_to_world(cam, step=roi_step)
        if len(pts) < 10:
            print(f"  {cam_id}: too few ROI points, skipping")
            continue
        valid_cams.append(cam_id)
        # IQR filter per camera before computing bbox — removes far-field
        # perspective blowups (pixels near the vanishing line that project
        # hundreds of metres away).  Factor 2.0 is looser than the default
        # 1.5 to preserve legitimate far-field scene content.
        pts_filt = _iqr_filter(pts, 2.0)
        if len(pts_filt) < 10:
            pts_filt = pts   # fallback if filter removes too much
        lo = 100.0 - percentile
        la_min, la_max = np.percentile(pts_filt[:, 0], [lo, percentile])
        lo_min, lo_max = np.percentile(pts_filt[:, 1], [lo, percentile])
        lat_mins.append(la_min);  lat_maxs.append(la_max)
        lon_mins.append(lo_min);  lon_maxs.append(lo_max)
        print(f"  {cam_id}: {len(pts):,} pts  "
              f"lat=[{la_min:.6f},{la_max:.6f}]  lon=[{lo_min:.6f},{lo_max:.6f}]")

    if not valid_cams:
        print("[ERROR] No world points found."); return

    # Union of per-camera boxes + margin
    lat_min = min(lat_mins);  lat_max = max(lat_maxs)
    lon_min = min(lon_mins);  lon_max = max(lon_maxs)
    margin_lat = (lat_max - lat_min) * bbox_margin
    margin_lon = (lon_max - lon_min) * bbox_margin
    lat_min -= margin_lat;  lat_max += margin_lat
    lon_min -= margin_lon;  lon_max += margin_lon
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    print(f"\n  Bounding box (union of per-camera {percentile:.0f}th-pct + {bbox_margin*100:.0f}% margin):")
    print(f"    lat [{lat_min:.6f}, {lat_max:.6f}]  Δ={lat_range:.6f}°")
    print(f"    lon [{lon_min:.6f}, {lon_max:.6f}]  Δ={lon_range:.6f}°")

    # ── 2. Canvas size (preserving metric aspect ratio) ───────────────────────
    lat_ref  = (lat_min + lat_max) / 2
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * np.cos(np.radians(lat_ref))

    lat_m = lat_range * m_per_lat
    lon_m = lon_range * m_per_lon

    if lon_m >= lat_m:                        # wider than tall
        canvas_w = canvas_long_side
        canvas_h = max(1, int(round(canvas_long_side * lat_m / lon_m)))
    else:                                     # taller than wide
        canvas_h = canvas_long_side
        canvas_w = max(1, int(round(canvas_long_side * lon_m / lat_m)))

    res_m = lon_m / canvas_w                  # metres per pixel (approx)
    print(f"\n  Canvas: {canvas_w}×{canvas_h} px  "
          f"({lon_m:.0f}m × {lat_m:.0f}m)  "
          f"≈ {res_m:.2f} m/px")

    # Canvas layout: north-up, col = east, row = south
    #   col cx → lon: lon = lon_min + cx * lon_range / canvas_w
    #   row cy → lat: lat = lat_max - cy * lat_range / canvas_h

    # ── 3. Flat world-coordinate arrays for every canvas pixel ───────────────
    print("\n  Building coordinate grid …")
    cx_arr = np.arange(canvas_w, dtype=np.float64)
    cy_arr = np.arange(canvas_h, dtype=np.float64)
    CX, CY = np.meshgrid(cx_arr, cy_arr)   # (H, W)
    N = canvas_h * canvas_w

    LAT = (lat_max - CY * (lat_range / canvas_h)).ravel()  # (N,)
    LON = (lon_min + CX * (lon_range / canvas_w)).ravel()  # (N,)

    # Homogeneous world coords: shape (3, N)
    pts_world = np.stack([LAT, LON, np.ones(N, dtype=np.float64)])

    # ── 4. Load mean frames and ROI masks ─────────────────────────────────────
    print("\n  Loading mean frames …")
    frames:    dict[str, np.ndarray] = {}
    roi_masks: dict[str, np.ndarray] = {}

    for cam_id in valid_cams:
        cam = seq.cameras[cam_id]
        print(f"    {cam_id} … ", end="", flush=True)
        avg = _mean_frame(cam.video_path, n_frames)
        if avg is None:
            print("video read failed, skipping")
            continue
        roi_raw = cv2.imread(str(cam.roi_path), cv2.IMREAD_GRAYSCALE)
        if roi_raw is None:
            print("ROI missing, skipping")
            continue
        _, roi_bin = cv2.threshold(roi_raw, 127, 255, cv2.THRESH_BINARY)
        frames[cam_id]    = avg
        roi_masks[cam_id] = roi_bin
        print(f"{avg.shape[1]}×{avg.shape[0]}")

    if not frames:
        print("[ERROR] No valid cameras."); return

    valid_cam_ids = [c for c in valid_cams if c in frames]

    # ── 5. Project each camera and fill canvas (average of all valid cameras) ───
    print("\n  Filling canvas …")
    acc   = np.zeros((N, 3), dtype=np.float32)
    count = np.zeros(N,      dtype=np.int32)

    for cam_id in tqdm(valid_cam_ids, desc="  Cameras"):
        H    = seq.cameras[cam_id].calibration.homography   # world → pixel
        proj = H @ pts_world           # (3, N)  [u*w, v*w, w]
        w     = proj[2]
        valid_w = np.abs(w) > 1e-9
        safe_w  = np.where(valid_w, w, 1.0)
        u = np.where(valid_w, proj[0] / safe_w, -1.0)
        v = np.where(valid_w, proj[1] / safe_w, -1.0)

        img_h, img_w = frames[cam_id].shape[:2]

        in_bounds = valid_w & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        u_int = u.astype(int).clip(0, img_w - 1)
        v_int = v.astype(int).clip(0, img_h - 1)
        in_roi = roi_masks[cam_id][v_int, u_int] > 0
        valid  = in_bounds & in_roi

        if not valid.any():
            continue

        bgr = frames[cam_id][v_int[valid], u_int[valid]]
        acc[valid]   += bgr[:, ::-1].astype(np.float32)   # BGR → RGB
        count[valid] += 1
        print(f"    {cam_id}: {valid.sum():,} pixels")

    # Average: covered pixels get the mean colour; uncovered pixels stay white.
    covered = count > 0
    canvas  = np.full((N, 3), 255, dtype=np.uint8)
    canvas[covered] = np.clip(
        acc[covered] / count[covered, np.newaxis], 0, 255
    ).astype(np.uint8)

    # ── 6. Reshape and save ───────────────────────────────────────────────────
    img = canvas.reshape(canvas_h, canvas_w, 3)

    raw_path = output_dir / f"{seq.id}_world_mosaic.png"
    cv2.imwrite(str(raw_path), img[:, :, ::-1])   # RGB → BGR
    print(f"\n  Saved raw image → {raw_path}")

    # Reference figure with GPS axes
    fig_path = output_dir / f"{seq.id}_world_mosaic_fig.png"
    fig_w = 12.0
    fig_h = fig_w * canvas_h / canvas_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(
        img,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        aspect="equal",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"World mosaic — Sequence {seq.id}   "
        f"{canvas_w}×{canvas_h} px   {res_m:.2f} m/px   "
        f"(mean of {n_frames} frames, {len(valid_cam_ids)} cameras)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure    → {fig_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="World-space photo mosaic")
    p.add_argument("--data-root",  default=DATA_ROOT)
    p.add_argument("--seq",        default="S01",
                   help="Sequence ID, or 'all' for every sequence")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--n-frames",   type=int,   default=N_FRAMES,
                   help="Frames to average per camera")
    p.add_argument("--size",       type=int,   default=CANVAS_LONG_SIDE,
                   help="Pixels on the longest canvas side")
    p.add_argument("--percentile", type=float, default=PERCENTILE,
                   help="Percentile for world bounding box (default 95)")
    args = p.parse_args()

    dataset = AICityDataset(args.data_root)
    seq_ids = sorted(dataset.sequences) if args.seq == "all" else [args.seq]

    for sid in seq_ids:
        if sid not in dataset.sequences:
            print(f"[WARN] Sequence {sid!r} not found, skipping."); continue
        make_world_mosaic(
            dataset.sequences[sid],
            output_dir       = Path(args.output_dir),
            n_frames         = args.n_frames,
            canvas_long_side = args.size,
            percentile       = args.percentile,
        )


if __name__ == "__main__":
    main()
