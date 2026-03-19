"""
World-space camera-coverage map for a sequence.

Same backward-projection canvas as make_world_mosaic.py, but each pixel
is coloured by which camera(s) see it.  Overlapping regions are blended
(average of the cameras' colours), so shared coverage shows as a mix.

Output
------
  output/world_viz/<seq>_camera_map.png       raw image
  output/world_viz/<seq>_camera_map_fig.png   same + GPS axes + legend

Usage
-----
    python make_camera_map.py                    # S01, 3000px long side
    python make_camera_map.py --seq S03
    python make_camera_map.py --seq S04 --size 5000
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

from src.dataset import AICityDataset
from try_to_create_camera_map import _iqr_filter, _palette, _roi_white_pixels

# Re-use bbox + canvas helpers from make_world_mosaic
from make_world_mosaic import _project_roi_to_world

# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_ROOT        = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
OUTPUT_DIR       = Path("output/world_viz")
CANVAS_LONG_SIDE = 3000
PERCENTILE       = 98.0
BBOX_MARGIN      = 0.10


# ── Main ──────────────────────────────────────────────────────────────────────

def make_camera_map(
    seq,
    output_dir:       Path  = OUTPUT_DIR,
    canvas_long_side: int   = CANVAS_LONG_SIDE,
    percentile:       float = PERCENTILE,
    bbox_margin:      float = BBOX_MARGIN,
    roi_step:         int   = 5,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Bounding box (identical to make_world_mosaic) ─────────────────────
    print(f"\n[{seq.id}] Projecting ROI pixels to world space …")
    valid_cams: list[str] = []
    lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []

    for cam_id, cam in sorted(seq.cameras.items()):
        pts = _project_roi_to_world(cam, step=roi_step)
        if len(pts) < 10:
            print(f"  {cam_id}: too few ROI points, skipping")
            continue
        valid_cams.append(cam_id)
        pts_filt = _iqr_filter(pts, 2.0)
        if len(pts_filt) < 10:
            pts_filt = pts
        lo = 100.0 - percentile
        la_min, la_max = np.percentile(pts_filt[:, 0], [lo, percentile])
        lo_min, lo_max = np.percentile(pts_filt[:, 1], [lo, percentile])
        lat_mins.append(la_min);  lat_maxs.append(la_max)
        lon_mins.append(lo_min);  lon_maxs.append(lo_max)
        print(f"  {cam_id}: {len(pts):,} pts")

    if not valid_cams:
        print("[ERROR] No world points found."); return

    lat_min = min(lat_mins);  lat_max = max(lat_maxs)
    lon_min = min(lon_mins);  lon_max = max(lon_maxs)
    margin_lat = (lat_max - lat_min) * bbox_margin
    margin_lon = (lon_max - lon_min) * bbox_margin
    lat_min -= margin_lat;  lat_max += margin_lat
    lon_min -= margin_lon;  lon_max += margin_lon
    lat_range = lat_max - lat_min
    lon_range = lon_max - lon_min

    # ── 2. Canvas size ────────────────────────────────────────────────────────
    lat_ref   = (lat_min + lat_max) / 2
    m_per_lat = 111_320.0
    m_per_lon = 111_320.0 * np.cos(np.radians(lat_ref))
    lat_m = lat_range * m_per_lat
    lon_m = lon_range * m_per_lon

    if lon_m >= lat_m:
        canvas_w = canvas_long_side
        canvas_h = max(1, int(round(canvas_long_side * lat_m / lon_m)))
    else:
        canvas_h = canvas_long_side
        canvas_w = max(1, int(round(canvas_long_side * lon_m / lat_m)))

    res_m = lon_m / canvas_w
    print(f"\n  Canvas: {canvas_w}×{canvas_h} px  ({lon_m:.0f}m × {lat_m:.0f}m)  ≈ {res_m:.2f} m/px")

    # ── 3. World-coordinate grid ──────────────────────────────────────────────
    print("\n  Building coordinate grid …")
    CX, CY = np.meshgrid(np.arange(canvas_w, dtype=np.float64),
                         np.arange(canvas_h, dtype=np.float64))
    N = canvas_h * canvas_w
    LAT = (lat_max - CY * (lat_range / canvas_h)).ravel()
    LON = (lon_min + CX * (lon_range / canvas_w)).ravel()
    pts_world = np.stack([LAT, LON, np.ones(N, dtype=np.float64)])

    # ── 4. Load ROI masks ─────────────────────────────────────────────────────
    roi_masks: dict[str, np.ndarray] = {}
    for cam_id in valid_cams:
        cam = seq.cameras[cam_id]
        roi_raw = cv2.imread(str(cam.roi_path), cv2.IMREAD_GRAYSCALE)
        if roi_raw is None:
            print(f"  {cam_id}: ROI missing, skipping")
            continue
        _, roi_bin = cv2.threshold(roi_raw, 127, 255, cv2.THRESH_BINARY)
        roi_masks[cam_id] = roi_bin

    valid_cam_ids = [c for c in valid_cams if c in roi_masks]

    # ── 5. Assign camera colours ──────────────────────────────────────────────
    print("\n  Filling canvas …")
    colors = _palette(len(valid_cam_ids))   # list of (r,g,b,a) in [0,1]
    cam_color = {cam_id: np.array(colors[i][:3]) * 255
                 for i, cam_id in enumerate(valid_cam_ids)}

    acc   = np.zeros((N, 3), dtype=np.float32)
    count = np.zeros(N,      dtype=np.int32)

    for cam_id in tqdm(valid_cam_ids, desc="  Cameras"):
        H    = seq.cameras[cam_id].calibration.homography
        proj = H @ pts_world
        w     = proj[2]
        valid_w = np.abs(w) > 1e-9
        safe_w  = np.where(valid_w, w, 1.0)
        u = np.where(valid_w, proj[0] / safe_w, -1.0)
        v = np.where(valid_w, proj[1] / safe_w, -1.0)

        img_h, img_w = roi_masks[cam_id].shape[:2]
        in_bounds = valid_w & (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        u_int  = u.astype(int).clip(0, img_w - 1)
        v_int  = v.astype(int).clip(0, img_h - 1)
        in_roi = roi_masks[cam_id][v_int, u_int] > 0
        valid  = in_bounds & in_roi

        if not valid.any():
            continue

        acc[valid]   += cam_color[cam_id].astype(np.float32)
        count[valid] += 1
        print(f"    {cam_id}: {valid.sum():,} pixels")

    # Average blends overlapping cameras; uncovered pixels stay white
    covered = count > 0
    canvas  = np.full((N, 3), 255, dtype=np.uint8)
    canvas[covered] = np.clip(
        acc[covered] / count[covered, np.newaxis], 0, 255
    ).astype(np.uint8)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    img = canvas.reshape(canvas_h, canvas_w, 3)

    raw_path = output_dir / f"{seq.id}_camera_map.png"
    cv2.imwrite(str(raw_path), img[:, :, ::-1])   # RGB → BGR
    print(f"\n  Saved raw image → {raw_path}")

    fig_path = output_dir / f"{seq.id}_camera_map_fig.png"
    fig_w = 12.0
    fig_h = fig_w * canvas_h / canvas_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.imshow(img, extent=[lon_min, lon_max, lat_min, lat_max],
              origin="upper", aspect="equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Camera coverage map — Sequence {seq.id}   "
        f"{canvas_w}×{canvas_h} px   {res_m:.2f} m/px",
        fontsize=11,
    )
    legend = [mpatches.Patch(facecolor=np.array(colors[i][:3]), label=cam_id)
              for i, cam_id in enumerate(valid_cam_ids)]
    ax.legend(handles=legend, loc="upper right", fontsize=9, framealpha=0.85)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure    → {fig_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="World-space camera coverage map")
    p.add_argument("--data-root",  default=DATA_ROOT)
    p.add_argument("--seq",        default="S01",
                   help="Sequence ID, or 'all' for every sequence")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--size",       type=int,   default=CANVAS_LONG_SIDE)
    p.add_argument("--percentile", type=float, default=PERCENTILE)
    args = p.parse_args()

    dataset = AICityDataset(args.data_root)
    seq_ids = sorted(dataset.sequences) if args.seq == "all" else [args.seq]

    for sid in seq_ids:
        if sid not in dataset.sequences:
            print(f"[WARN] Sequence {sid!r} not found, skipping."); continue
        make_camera_map(
            dataset.sequences[sid],
            output_dir       = Path(args.output_dir),
            canvas_long_side = args.size,
            percentile       = args.percentile,
        )


if __name__ == "__main__":
    main()
