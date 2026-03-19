"""
World-space projection-uncertainty map for a sequence.

For each canvas pixel the script:
  1. Projects it to every camera via H (world → pixel)
  2. Keeps only pixels that land inside the camera's image and ROI mask
  3. Propagates the per-camera calibration error (σ_pixel) through the
     analytic homography Jacobian to get σ_world in metres
  4. Assigns each canvas pixel the *minimum* σ across valid cameras
     (i.e. the best precision available at that world location)

The result shows how calibration uncertainty blows up near the vanishing
line (far field) and is small close to the camera nadir.

Output
------
  output/world_viz/<seq>_uncertainty_map.png       raw colourmap image
  output/world_viz/<seq>_uncertainty_map_fig.png   same + axes + colourbar

Usage
-----
    python make_uncertainty_map.py                    # S01, 3000px
    python make_uncertainty_map.py --seq S03
    python make_uncertainty_map.py --seq all --size 4000
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.dataset import AICityDataset
from try_to_create_camera_map import _iqr_filter
from make_world_mosaic import _project_roi_to_world

# ── Defaults ──────────────────────────────────────────────────────────────────

DATA_ROOT        = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
OUTPUT_DIR       = Path("output/world_viz")
CANVAS_LONG_SIDE = 3000
PERCENTILE       = 98.0
BBOX_MARGIN      = 0.10

# Metres-per-degree constants (rough, sufficient for colouring)
M_PER_LAT = 111_320.0


# ── Vectorised Jacobian ────────────────────────────────────────────────────────

def _jacobian_frob_sq_batch(H_inv: np.ndarray,
                             u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorised Frobenius-norm² of the homography Jacobian ∂(world)/∂(pixel)
    at every pixel (u[i], v[i]).

    The Jacobian is the 2×2 matrix:
        J[0,0] = (Hi[0,0]*p2 - p0*Hi[2,0]) / p2²
        J[0,1] = (Hi[0,1]*p2 - p0*Hi[2,1]) / p2²
        J[1,0] = (Hi[1,0]*p2 - p1*Hi[2,0]) / p2²
        J[1,1] = (Hi[1,1]*p2 - p1*Hi[2,1]) / p2²

    where  [p0, p1, p2] = H_inv @ [u, v, 1].

    Returns an (N,) array of ‖J‖²_F values (in (°/px)²).
    """
    ones = np.ones_like(u)
    pts  = np.stack([u, v, ones])           # (3, N)
    p    = H_inv @ pts                      # (3, N)
    p0, p1, p2 = p[0], p[1], p[2]
    p2_sq = p2 ** 2

    Hi = H_inv
    frob_sq = (
        (Hi[0, 0] * p2 - p0 * Hi[2, 0]) ** 2 +
        (Hi[0, 1] * p2 - p0 * Hi[2, 1]) ** 2 +
        (Hi[1, 0] * p2 - p1 * Hi[2, 0]) ** 2 +
        (Hi[1, 1] * p2 - p1 * Hi[2, 1]) ** 2
    ) / (p2_sq ** 2)

    return frob_sq


# ── Main ──────────────────────────────────────────────────────────────────────

def make_uncertainty_map(
    seq,
    output_dir:       Path  = OUTPUT_DIR,
    canvas_long_side: int   = CANVAS_LONG_SIDE,
    percentile:       float = PERCENTILE,
    bbox_margin:      float = BBOX_MARGIN,
    roi_step:         int   = 5,
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Bounding box ───────────────────────────────────────────────────────
    print(f"\n[{seq.id}] Projecting ROI pixels to world space …")
    valid_cams: list[str] = []
    lat_mins, lat_maxs, lon_mins, lon_maxs = [], [], [], []

    for cam_id, cam in sorted(seq.cameras.items()):
        pts = _project_roi_to_world(cam, step=roi_step)
        if len(pts) < 10:
            print(f"  {cam_id}: too few ROI points, skipping"); continue
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
    m_per_lon = M_PER_LAT * np.cos(np.radians(lat_ref))
    lat_m     = lat_range * M_PER_LAT
    lon_m     = lon_range * m_per_lon

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
        roi_raw = cv2.imread(str(seq.cameras[cam_id].roi_path), cv2.IMREAD_GRAYSCALE)
        if roi_raw is None:
            print(f"  {cam_id}: ROI missing, skipping"); continue
        _, roi_bin = cv2.threshold(roi_raw, 127, 255, cv2.THRESH_BINARY)
        roi_masks[cam_id] = roi_bin

    valid_cam_ids = [c for c in valid_cams if c in roi_masks]

    # ── 5. Compute per-pixel minimum σ (metres) ───────────────────────────────
    print("\n  Computing uncertainty …")

    # Average m/° at this latitude (used to convert σ in °→metres)
    m_per_deg = (M_PER_LAT + m_per_lon) / 2.0

    # Start with +inf; take element-wise minimum across cameras
    sigma_min = np.full(N, np.inf, dtype=np.float64)

    for cam_id in tqdm(valid_cam_ids, desc="  Cameras"):
        cam  = seq.cameras[cam_id]
        H    = cam.calibration.homography          # world → pixel
        H_inv = np.linalg.inv(H)                  # pixel → world
        sigma_pixel_sq = cam.calibration.reprojection_error ** 2

        # Forward-project world → camera pixel
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

        # Vectorised Jacobian for valid pixels
        frob_sq = _jacobian_frob_sq_batch(H_inv, u[valid], v[valid])

        # σ²_world in °²; σ_world in ° → metres
        sigma_sq_deg = sigma_pixel_sq * frob_sq / 2.0
        sigma_m      = np.sqrt(sigma_sq_deg) * m_per_deg

        # Take element-wise minimum
        current = sigma_min[valid]
        sigma_min[valid] = np.minimum(current, sigma_m)

        finite = np.isfinite(sigma_m)
        if finite.any():
            print(f"    {cam_id}: σ ∈ [{sigma_m[finite].min():.3f}, {sigma_m[finite].max():.3f}] m  "
                  f"({valid.sum():,} pixels)")

    # ── 6. Colour map ─────────────────────────────────────────────────────────
    covered = np.isfinite(sigma_min)
    print(f"\n  Covered pixels: {covered.sum():,} / {N:,}")
    if not covered.any():
        print("[ERROR] No covered pixels."); return

    vals = sigma_min[covered]
    print(f"  σ range: [{vals.min():.3f}, {vals.max():.3f}] m  "
          f"(median {np.median(vals):.3f} m)")

    # Log scale so both near (<0.1 m) and far (>10 m) are visible
    log_vals    = np.log10(np.clip(vals, 1e-6, None))
    log_min, log_max = np.percentile(log_vals, [1, 99])

    norm_vals = np.clip((log_vals - log_min) / (log_max - log_min + 1e-12), 0, 1)
    cmap      = plt.get_cmap("plasma")
    colors    = (cmap(norm_vals)[:, :3] * 255).astype(np.uint8)   # RGB

    canvas = np.full((N, 3), 255, dtype=np.uint8)   # white = no data
    canvas[covered] = colors

    img = canvas.reshape(canvas_h, canvas_w, 3)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    raw_path = output_dir / f"{seq.id}_uncertainty_map.png"
    cv2.imwrite(str(raw_path), img[:, :, ::-1])
    print(f"\n  Saved raw image → {raw_path}")

    fig_path = output_dir / f"{seq.id}_uncertainty_map_fig.png"
    fig_w = 12.0
    fig_h = fig_w * canvas_h / canvas_w
    fig, ax = plt.subplots(figsize=(fig_w, fig_h + 0.6), dpi=150)
    im = ax.imshow(
        img,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper", aspect="equal",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Projection uncertainty (best camera) — Sequence {seq.id}   "
        f"{canvas_w}×{canvas_h} px   {res_m:.2f} m/px",
        fontsize=11,
    )

    # Colourbar with real-metre ticks
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=log_min, vmax=log_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("σ  (metres,  log₁₀ scale)", fontsize=9)

    # Pick a few nice tick positions in log space
    tick_log = np.arange(np.ceil(log_min * 2) / 2, np.floor(log_max * 2) / 2 + 0.1, 0.5)
    tick_log = tick_log[(tick_log >= log_min) & (tick_log <= log_max)]
    cbar.set_ticks(tick_log)
    cbar.set_ticklabels([f"{10**t:.2g} m" for t in tick_log])

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure    → {fig_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="World-space projection-uncertainty map")
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
        make_uncertainty_map(
            dataset.sequences[sid],
            output_dir       = Path(args.output_dir),
            canvas_long_side = args.size,
            percentile       = args.percentile,
        )


if __name__ == "__main__":
    main()
