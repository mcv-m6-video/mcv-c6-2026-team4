"""
World-space projection diagnostic.

For every GT vehicle that appears in 2+ cameras, uses the camera homographies
to project ground-contact points (bottom-centre of GT bounding boxes) to world
coordinates, then computes:

  - Same-vehicle, cross-camera world distance at co-temporal observations (should be SMALL)
  - Different-vehicle, cross-camera world distance at co-temporal observations (should be LARGE)

"Co-temporal" is defined using ABSOLUTE timestamps (camera.start_timestamp +
(frame_id - 1) / fps), NOT by matching equal FrameIds across cameras.  This
is critical for unsynchronised sequences like S04, where camera c016 starts at
t=0 s and camera c040 starts at t=175.8 s — frame 100 in each camera refers to
completely different moments in time.

Reports:
  - Distribution statistics for both groups
  - Suggested SAME_THRESHOLD and DIFF_THRESHOLD for the hard co-temporal gate
  - Separation quality metric (gap between distributions)
  - Histogram saved to <output_dir>/world_space_distributions.png

These values can be fed directly into ClusteringAssociator via
--same-threshold and --diff-threshold once those parameters are added.

Usage
-----
python world_space_diagnostic.py
python world_space_diagnostic.py --seq S03 --cameras c010 c011 c012 c013 c014 c015
python world_space_diagnostic.py --seq S01 --n-diff-pairs 5000
python world_space_diagnostic.py --seq S04 --fps 10 --fps-override c015:8
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset import AICityDataset
from src.eval import readData
from src.world_and_camera_tracking import _project_point

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DATA_ROOT  = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
SEQ_ID     = "S01"
CAMERAS    = ["c001", "c002", "c003", "c004", "c005"]
GT_PATH    = f"{DATA_ROOT}/eval/ground_truth_train.txt"
OUTPUT_DIR = Path("output/world_diagnostic")

N_DIFF_PAIRS       = 3000   # random different-vehicle co-temporal pairs
MIN_OVERLAP_SECS   = 0.5    # minimum temporal overlap (seconds) to count as co-temporal
MAX_DT_SECS        = 0.15   # two observations are "simultaneous" if |t_a - t_b| ≤ this
DEFAULT_FPS        = 10.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="World-space projection diagnostic")
    p.add_argument("--data-root",       default=DATA_ROOT)
    p.add_argument("--seq",             default=SEQ_ID)
    p.add_argument("--cameras",         nargs="+", default=CAMERAS)
    p.add_argument("--gt-path",         default=GT_PATH)
    p.add_argument("--n-diff-pairs",    type=int, default=N_DIFF_PAIRS)
    p.add_argument("--min-overlap",     type=float, default=MIN_OVERLAP_SECS,
                   help="Minimum temporal overlap in seconds to include a pair")
    p.add_argument("--max-dt",          type=float, default=MAX_DT_SECS,
                   help="Max time difference (s) for two observations to be 'simultaneous'")
    p.add_argument("--fps",             type=float, default=DEFAULT_FPS,
                   help="Default camera frame rate (used when video header cannot be read)")
    p.add_argument("--fps-override",    nargs="*", default=[],
                   metavar="CAM:FPS",
                   help="Per-camera fps overrides, e.g. --fps-override c015:8")
    p.add_argument("--output-dir",      default=str(OUTPUT_DIR))
    return p.parse_args()


def _build_fps_map(cameras: list[str], seq, default_fps: float,
                   overrides: list[str]) -> dict[str, float]:
    """
    Returns {cam_id: fps} for each camera.

    Priority: --fps-override > video header > --fps default.
    Reading the video header requires the video file to exist; if it doesn't,
    falls back to the default.
    """
    override_map = {}
    for token in overrides:
        cam, fps_str = token.split(":")
        override_map[cam.strip()] = float(fps_str.strip())

    fps_map = {}
    for cam_id in cameras:
        if cam_id in override_map:
            fps_map[cam_id] = override_map[cam_id]
            continue

        cam = seq[cam_id]
        video_path = str(cam.video_path)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            fps_map[cam_id] = fps if fps > 0 else default_fps
        else:
            fps_map[cam_id] = default_fps

    return fps_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt_world_positions(
    gt_path: str,
    camera_ids: list[str],
    seq: object,           # AICityDataset Sequence
    fps_map: dict[str, float],
) -> dict[int, dict[str, dict[float, tuple[float, float]]]]:
    """
    Returns:
        { vehicle_id: { cam_id: { abs_timestamp: (world_x, world_y) } } }

    Timestamps are ABSOLUTE (sequence clock), computed as:
        abs_timestamp = camera.start_timestamp + (frame_id - 1) / fps

    Only vehicles appearing in ≥ 2 of the requested cameras are included.
    World positions are computed by projecting the bottom-centre of each GT
    bounding box through the camera's homography.
    """
    df = readData(gt_path)

    cam_int_to_str = {int(c.lstrip("c")): c for c in camera_ids}
    df = df[df["CameraId"].isin(cam_int_to_str.keys())]

    # Build world-position lookup per vehicle per camera per absolute timestamp
    data: dict[int, dict[str, dict[float, tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for row in df.itertuples(index=False):
        cam_str = cam_int_to_str[row.CameraId]
        camera  = seq[cam_str]
        fps     = fps_map[cam_str]

        # Absolute timestamp on the shared sequence clock
        abs_t = camera.start_timestamp + (row.FrameId - 1) / fps

        # calibration.txt stores world→pixel; invert to get pixel→world
        H_inv = np.linalg.inv(camera.calibration.homography)

        # Bottom-centre contact point
        cx = row.X + row.Width  / 2.0
        cy = row.Y + row.Height         # bottom edge

        wx, wy = _project_point(H_inv, cx, cy)
        data[row.Id][cam_str][abs_t] = (wx, wy)

    # Keep only multi-camera vehicles
    return {
        vid: dict(cam_data)
        for vid, cam_data in data.items()
        if len(cam_data) >= 2
    }


# ---------------------------------------------------------------------------
# Distance computation
# ---------------------------------------------------------------------------

def _mean_world_dist_during_overlap(
    pos_i: dict[float, tuple[float, float]],
    pos_j: dict[float, tuple[float, float]],
    min_overlap_secs: float,
    max_dt: float,
) -> float | None:
    """
    Mean Euclidean world distance between two tracklets at timestamps where
    both have a GT annotation within `max_dt` seconds of each other.

    Returns None if the accumulated matched interval is shorter than
    `min_overlap_secs` (i.e. there is not enough true temporal overlap).

    Algorithm: for each observation in the smaller tracklet, find the nearest
    timestamp in the other tracklet.  If the gap ≤ max_dt, count it as a
    matched pair and accumulate the distance.
    """
    ts_i = sorted(pos_i.keys())
    ts_j = sorted(pos_j.keys())

    if not ts_i or not ts_j:
        return None

    # Quick temporal-overlap check using the tracklet time spans
    t_i0, t_i1 = ts_i[0],  ts_i[-1]
    t_j0, t_j1 = ts_j[0],  ts_j[-1]
    overlap_start = max(t_i0, t_j0)
    overlap_end   = min(t_i1, t_j1)
    if overlap_end - overlap_start < min_overlap_secs:
        return None

    # Match each observation in i to the nearest observation in j
    ts_j_arr = np.array(ts_j)
    dists = []
    for t in ts_i:
        idx = int(np.argmin(np.abs(ts_j_arr - t)))
        if abs(ts_j_arr[idx] - t) <= max_dt:
            xi, yi = pos_i[t]
            xj, yj = pos_j[ts_j[idx]]
            dists.append(np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2))

    if not dists:
        return None

    # Require enough matched observations to represent min_overlap_secs
    # (at 10 fps, 0.5 s = 5 frames minimum)
    min_matches = max(1, int(min_overlap_secs * 10 * 0.5))  # conservative lower bound
    if len(dists) < min_matches:
        return None

    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset …")
    ds  = AICityDataset(args.data_root)
    seq = ds[args.seq]

    print("Detecting camera frame rates …")
    fps_map = _build_fps_map(args.cameras, seq, args.fps, args.fps_override)
    for cam_id, fps in fps_map.items():
        print(f"  {cam_id}: {fps:.1f} fps  (start_t={seq[cam_id].start_timestamp:.3f}s)")

    print("Loading GT and projecting to world …")
    gt = load_gt_world_positions(args.gt_path, args.cameras, seq, fps_map)
    print(f"  Multi-camera vehicles: {len(gt)}")

    # ------------------------------------------------------------------
    # Same-vehicle cross-camera co-temporal distances
    # ------------------------------------------------------------------
    same_dists: list[float] = []
    same_meta:  list[tuple] = []   # (vehicle_id, cam_i, cam_j, n_matched_obs)

    for vid, cam_data in gt.items():
        cams = list(cam_data.keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                d = _mean_world_dist_during_overlap(
                    cam_data[cams[i]], cam_data[cams[j]],
                    args.min_overlap, args.max_dt,
                )
                if d is not None:
                    # Count how many obs were matched (approximate)
                    ts_i = np.array(sorted(cam_data[cams[i]].keys()))
                    ts_j = np.array(sorted(cam_data[cams[j]].keys()))
                    n_matched = sum(
                        np.min(np.abs(ts_j - t)) <= args.max_dt for t in ts_i
                    )
                    same_dists.append(d)
                    same_meta.append((vid, cams[i], cams[j], n_matched))

    print(f"  Same-vehicle co-temporal pairs: {len(same_dists)}")

    if not same_dists:
        print("\n[WARN] No same-vehicle co-temporal pairs found.")
        print("       Either the cameras have no real temporal overlap,")
        print("       or all overlaps are shorter than --min-overlap.")
        print("       This is expected for sequences where cameras cover")
        print("       completely different time windows (no co-temporal coverage).")
        return

    # ------------------------------------------------------------------
    # Different-vehicle cross-camera co-temporal distances
    # ------------------------------------------------------------------
    all_cam_pos: list[tuple[int, str, dict]] = []   # (vid, cam_id, pos_dict)
    for vid, cam_data in gt.items():
        for cam_id, pos_dict in cam_data.items():
            all_cam_pos.append((vid, cam_id, pos_dict))

    diff_dists: list[float] = []
    rng = random.Random(42)
    attempts = 0
    while len(diff_dists) < args.n_diff_pairs and attempts < args.n_diff_pairs * 20:
        attempts += 1
        vid_a, cam_a, pos_a = rng.choice(all_cam_pos)
        vid_b, cam_b, pos_b = rng.choice(all_cam_pos)
        if vid_a == vid_b:
            continue
        d = _mean_world_dist_during_overlap(pos_a, pos_b, args.min_overlap, args.max_dt)
        if d is not None:
            diff_dists.append(d)

    print(f"  Different-vehicle co-temporal pairs: {len(diff_dists)}")

    if not diff_dists:
        print("\n[WARN] No different-vehicle co-temporal pairs found either.")
        return

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    same_arr = np.array(same_dists)
    diff_arr = np.array(diff_dists)

    print("\n" + "=" * 60)
    print("WORLD-SPACE DIAGNOSTIC RESULTS")
    print("=" * 60)

    print(f"\nSame-vehicle, co-temporal ({len(same_arr)} pairs):")
    print(f"  mean   = {same_arr.mean():.6f}")
    print(f"  std    = {same_arr.std():.6f}")
    print(f"  min    = {same_arr.min():.6f}  max = {same_arr.max():.6f}")
    print(f"  p50    = {np.median(same_arr):.6f}")
    print(f"  p90    = {np.percentile(same_arr, 90):.6f}")
    print(f"  p95    = {np.percentile(same_arr, 95):.6f}")
    print(f"  p99    = {np.percentile(same_arr, 99):.6f}")

    print(f"\nDifferent-vehicle, co-temporal ({len(diff_arr)} pairs):")
    print(f"  mean   = {diff_arr.mean():.6f}")
    print(f"  std    = {diff_arr.std():.6f}")
    print(f"  min    = {diff_arr.min():.6f}  max = {diff_arr.max():.6f}")
    print(f"  p1     = {np.percentile(diff_arr, 1):.6f}")
    print(f"  p5     = {np.percentile(diff_arr, 5):.6f}")
    print(f"  p10    = {np.percentile(diff_arr, 10):.6f}")
    print(f"  p50    = {np.median(diff_arr):.6f}")

    # Suggested thresholds
    same_threshold = float(np.percentile(same_arr, 95))
    diff_threshold = float(np.percentile(diff_arr, 5))
    gap = diff_threshold - same_threshold

    print(f"\nSuggested thresholds:")
    print(f"  SAME_THRESHOLD  (same_p95)  = {same_threshold:.6f}")
    print(f"    → merge if world distance < this (Tier-1 force merge)")
    print(f"  DIFF_THRESHOLD  (diff_p5)   = {diff_threshold:.6f}")
    print(f"    → block if world distance > this (Tier-1 hard block)")
    print(f"  Gap between thresholds      = {gap:.6f}")

    if gap > 0:
        print(f"  → Clean separation: hard co-temporal gate will work well")
        overlap_frac = np.mean(same_arr > diff_threshold) + np.mean(diff_arr < same_threshold)
        print(f"  → Overlap fraction (both sides): {100*overlap_frac/2:.1f}%")
    else:
        print(f"  → OVERLAP between distributions (gap < 0)")
        print(f"  → Hard gate will make errors — homography may be too noisy")
        print(f"  → Consider using soft cost instead (w_geo + geo_scale={same_arr.mean():.6f})")

    # Reprojection error context
    print(f"\nCamera reprojection errors:")
    for cam_id in args.cameras:
        if cam_id in seq.cameras:
            err = seq[cam_id].calibration.reprojection_error
            print(f"  {cam_id}: {err:.4f} world units")

    # ------------------------------------------------------------------
    # Worst same-vehicle pairs (biggest projection error)
    # ------------------------------------------------------------------
    print(f"\nWorst 10 same-vehicle co-temporal pairs (largest world distance):")
    for dist, (vid, ci, cj, nf) in sorted(
        zip(same_dists, same_meta), reverse=True
    )[:10]:
        print(f"  vehicle={vid:5d}  {ci} ↔ {cj}  dist={dist:.6f}  matched_obs={nf}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    # Cap the x-axis at 3× same_threshold to keep the plot readable
    # (diff distances can be arbitrarily large)
    x_max = max(same_threshold * 3, np.percentile(diff_arr, 90))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    bins_same = np.linspace(0, same_arr.max() * 1.05, 60)
    bins_diff = np.linspace(0, min(diff_arr.max() * 1.05, x_max * 2), 60)

    ax = axes[0]
    ax.hist(same_arr, bins=60, alpha=0.6, color="steelblue",
            label=f"Same vehicle ({len(same_arr)} pairs)", density=True)
    ax.hist(np.clip(diff_arr, 0, x_max * 2), bins=60, alpha=0.6, color="tomato",
            label=f"Different vehicle ({len(diff_arr)} pairs)", density=True)
    ax.axvline(same_threshold, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"same_p95 = {same_threshold:.5f}")
    ax.axvline(diff_threshold, color="tomato", linestyle="--", linewidth=1.5,
               label=f"diff_p5  = {diff_threshold:.5f}")
    ax.set_xlabel("World distance (homogeneous units)")
    ax.set_ylabel("Density")
    ax.set_title(f"Full range — {args.seq}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Zoomed view around the threshold region
    ax2 = axes[1]
    zoom_max = x_max
    same_zoom = same_arr[same_arr <= zoom_max]
    diff_zoom = diff_arr[diff_arr <= zoom_max]
    bins_zoom = np.linspace(0, zoom_max, 80)
    ax2.hist(same_zoom, bins=bins_zoom, alpha=0.6, color="steelblue",
             label=f"Same vehicle", density=True)
    ax2.hist(diff_zoom, bins=bins_zoom, alpha=0.6, color="tomato",
             label=f"Different vehicle", density=True)
    ax2.axvline(same_threshold, color="steelblue", linestyle="--", linewidth=1.5,
                label=f"same_p95 = {same_threshold:.5f}")
    ax2.axvline(diff_threshold, color="tomato", linestyle="--", linewidth=1.5,
                label=f"diff_p5  = {diff_threshold:.5f}")
    if gap > 0:
        ax2.axvspan(same_threshold, diff_threshold, alpha=0.12, color="green",
                    label=f"Gate window (gap={gap:.5f})")
    ax2.set_xlabel("World distance (homogeneous units)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Zoomed (0 to {zoom_max:.5f}) — {args.seq}")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"World-space distance distributions — {args.seq}\n"
        f"SAME_THRESHOLD={same_threshold:.5f}  DIFF_THRESHOLD={diff_threshold:.5f}  "
        f"gap={'POSITIVE ✓' if gap > 0 else 'NEGATIVE ✗'}"
    )
    fig.tight_layout()

    plot_path = out_dir / f"world_space_distributions_{args.seq}.png"
    fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {plot_path}")

    # ------------------------------------------------------------------
    # Summary recommendation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if gap > 0:
        print(f"""
The distributions are separable. Add these flags to run_offline_mtmc.py:

  --w-geo 1.0
  --same-threshold {same_threshold:.6f}
  --diff-threshold {diff_threshold:.6f}

Expected effect:
  - Co-temporal pairs within {same_threshold:.6f} world units → cost=0 (force merge)
  - Co-temporal pairs beyond {diff_threshold:.6f} world units → cost=inf (hard block)
  - Pairs in the uncertain zone [{same_threshold:.5f}, {diff_threshold:.5f}] → use ReID

This replaces the uncalibrated w_geo soft cost with a hard binary gate.
""")
    else:
        print(f"""
The distributions OVERLAP (gap={gap:.6f}).
The homography reprojection error is too large for a hard gate to be reliable.

Options:
  1. Use a soft geo cost with geo_scale={same_arr.mean():.6f} (the mean same-vehicle distance)
     This at least makes the scale meaningful even if distributions overlap.
  2. Accept that spatial cost will add noise and keep w_geo=0.
  3. Investigate whether homography calibration can be improved.
""")


if __name__ == "__main__":
    main()
