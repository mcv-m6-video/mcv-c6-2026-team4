"""
World-space visualizations for slide decks.

Directly adapted from the scatter background + animation code in
run_offline_mtmc.py  (same projection: pixel → world via inv(H)).

Three visualizations
--------------------
1. draw_scene_map(seq)
   ROI pixels projected to world, coloured by camera.
   Identical to the animation background.

2. draw_photo_mosaic(seq)
   Same projection, but each point is coloured by its actual
   pixel value in the mean video frame (first N frames averaged).
   Only ROI-masked pixels are shown.

3. draw_gt_trajectories(seq)
   Scene map background + ground-truth vehicle paths in world space.

Usage
-----
    python visualize_world_space.py
    python visualize_world_space.py --seq S03
    python visualize_world_space.py --seq S01 --viz scene photo gt
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.dataset import AICityDataset

# Re-use helpers from the existing camera-map script
from try_to_create_camera_map import _palette, _roi_white_pixels, _iqr_filter

DATA_ROOT  = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
OUTPUT_DIR = Path("output/world_viz")
PIXEL_STEP = 5    # sub-sample ROI pixels (5 = every 5th pixel in each dim)
N_FRAMES   = 60   # frames to average for photo mosaic


# ---------------------------------------------------------------------------
# Core helpers  (same as animation code in run_offline_mtmc.py)
# ---------------------------------------------------------------------------

def _project_roi_to_world(cam, step: int = PIXEL_STEP):
    """
    Project all ROI white pixels to world space.
    H in calibration.txt is world→pixel, so we invert it.
    Returns (N, 2) float64 world coords, IQR-filtered.
    """
    img_pts = _roi_white_pixels(cam.roi_path, step)
    if len(img_pts) == 0:
        return np.empty((0, 2))

    H_inv = np.linalg.inv(cam.calibration.homography)
    ones   = np.ones((len(img_pts), 1))
    pts_h  = np.hstack([img_pts, ones])       # (N, 3)
    w_h    = (H_inv @ pts_h.T).T              # (N, 3)
    denom  = w_h[:, 2]
    valid  = np.abs(denom) > 1e-9
    world  = w_h[valid, :2] / denom[valid, np.newaxis]
    return _iqr_filter(world, 1.5)


def _project_roi_to_world_with_pixels(cam, avg_frame_bgr, step: int = PIXEL_STEP):
    """
    Like _project_roi_to_world but also returns the BGR colour of each pixel
    sampled from avg_frame_bgr.

    Returns (world_pts (N,2), colors (N,3) float 0-1 RGB).
    """
    img_pts = _roi_white_pixels(cam.roi_path, step)   # (M, 2) as (col, row)
    if len(img_pts) == 0:
        return np.empty((0, 2)), np.empty((0, 3))

    # Sample pixel colours before projection (keep index alignment)
    cols = img_pts[:, 0].astype(int).clip(0, avg_frame_bgr.shape[1] - 1)
    rows = img_pts[:, 1].astype(int).clip(0, avg_frame_bgr.shape[0] - 1)
    bgr  = avg_frame_bgr[rows, cols].astype(np.float32) / 255.0
    rgb  = bgr[:, ::-1]   # BGR → RGB

    H_inv = np.linalg.inv(cam.calibration.homography)
    ones  = np.ones((len(img_pts), 1))
    pts_h = np.hstack([img_pts, ones])
    w_h   = (H_inv @ pts_h.T).T
    denom = w_h[:, 2]
    valid = np.abs(denom) > 1e-9

    world = w_h[valid, :2] / denom[valid, np.newaxis]
    rgb   = rgb[valid]

    # IQR filter (need to apply same mask to colours)
    if len(world) < 4:
        return world, rgb

    mask = np.ones(len(world), dtype=bool)
    for axis in range(2):
        vals = world[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            mask &= (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)

    return world[mask], rgb[mask]


def _mean_frame(video_path: Path, n: int = N_FRAMES):
    """Average n evenly-spaced frames. Returns BGR uint8, or None on failure."""
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


def _axis_limits(all_world_pts, margin: float = 0.08, percentile: float = 96.0):
    """
    Use percentile-based limits to clip the axis to the dense central region,
    cutting off far-field streaks caused by perspective distortion near the
    vanishing line.
    """
    lo = 100.0 - percentile
    xmin, xmax = np.percentile(all_world_pts[:, 0], [lo, percentile])
    ymin, ymax = np.percentile(all_world_pts[:, 1], [lo, percentile])
    dx = (xmax - xmin) * margin
    dy = (ymax - ymin) * margin
    return (xmin - dx, xmax + dx), (ymin - dy, ymax + dy)


def _setup_figure(xlim, ylim, title, figsize=None):
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    if figsize is None:
        w = 12.0
        h = max(5.0, w * dy / max(dx, 1e-12))
        figsize = (w, h)
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("World X (GPS latitude)")
    ax.set_ylabel("World Y (GPS longitude)")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)
    return fig, ax


# ---------------------------------------------------------------------------
# 1. Scene map  (camera colours, identical to animation background)
# ---------------------------------------------------------------------------

def draw_scene_map(seq, output_dir: Path = OUTPUT_DIR, step: int = PIXEL_STEP):
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_items = sorted(seq.cameras.items())
    colors    = _palette(len(cam_items))
    scatter   = {}

    for (cam_id, cam), color in zip(cam_items, colors):
        pts = _project_roi_to_world(cam, step)
        if len(pts):
            scatter[cam_id] = (pts, color)
            print(f"  {cam_id}: {len(pts):,} points")

    if not scatter:
        print("[WARN] No data"); return

    all_pts = np.vstack([p for p, _ in scatter.values()])
    xlim, ylim = _axis_limits(all_pts)

    fig, ax = _setup_figure(xlim, ylim,
                            f"Camera ROI footprints — Sequence {seq.id}")

    legend = []
    for cam_id, (pts, color) in scatter.items():
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=0.5, color=color[:3], alpha=0.4,
                   linewidths=0, rasterized=True)
        legend.append(mpatches.Patch(facecolor=color[:3], label=cam_id))

    ax.legend(handles=legend, loc="upper right", fontsize=8,
              framealpha=0.85, markerscale=8)
    plt.tight_layout()
    out = output_dir / f"{seq.id}_scene_map.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Photo mosaic  (actual pixel colours)
# ---------------------------------------------------------------------------

def draw_photo_mosaic(seq, output_dir: Path = OUTPUT_DIR,
                      step: int = PIXEL_STEP, n_frames: int = N_FRAMES):
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_items = sorted(seq.cameras.items())
    all_data  = []   # (world_pts, rgb_colors, cam_id, color)

    for cam_id, cam in cam_items:
        print(f"  {cam_id}: averaging {n_frames} frames …")
        avg = _mean_frame(cam.video_path, n_frames)
        if avg is None:
            print(f"    [WARN] could not read video")
            continue
        pts, rgb = _project_roi_to_world_with_pixels(cam, avg, step)
        if len(pts):
            all_data.append((pts, rgb, cam_id))
            print(f"    {len(pts):,} points projected")

    if not all_data:
        print("[WARN] No data"); return

    all_pts = np.vstack([p for p, _, _ in all_data])
    xlim, ylim = _axis_limits(all_pts)

    fig, ax = _setup_figure(xlim, ylim,
                            f"Photo mosaic in world space — Sequence {seq.id}  "
                            f"(mean of {n_frames} frames, {len(all_data)} cameras)")

    for pts, rgb, cam_id in all_data:
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=rgb, s=0.8, linewidths=0,
                   alpha=0.9, rasterized=True)

    plt.tight_layout()
    out = output_dir / f"{seq.id}_photo_mosaic.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. GT trajectories  (scene map + vehicle paths)
# ---------------------------------------------------------------------------

def _load_gt(cam):
    """Parse gt.txt → {track_id: [(frame, x, y, w, h), ...]}."""
    if cam.gt_path is None or not cam.gt_path.exists():
        return {}
    tracks = {}
    with open(cam.gt_path) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                fid = int(parts[0]); tid = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            except ValueError:
                continue
            tracks.setdefault(tid, []).append((x, y, w, h))
    return tracks


def draw_gt_trajectories(seq, output_dir: Path = OUTPUT_DIR, step: int = PIXEL_STEP):
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_items = sorted(seq.cameras.items())
    colors    = _palette(len(cam_items))
    scatter   = {}
    all_traj  = []   # (world_pts_Nx2, color)

    for (cam_id, cam), color in zip(cam_items, colors):
        # Background scatter
        pts = _project_roi_to_world(cam, step)
        if len(pts):
            scatter[cam_id] = (pts, color)

        # GT trajectories
        H_inv = np.linalg.inv(cam.calibration.homography)
        tracks = _load_gt(cam)
        for tid, detections in tracks.items():
            world_path = []
            for (bx, by, bw, bh) in detections:
                u = bx + bw / 2.0   # bottom-centre contact point
                v = by + bh
                p = H_inv @ np.array([u, v, 1.0])
                if abs(p[2]) < 1e-9:
                    continue
                xw, yw = p[0] / p[2], p[1] / p[2]
                if np.isfinite(xw) and np.isfinite(yw):
                    world_path.append([xw, yw])
            if len(world_path) >= 2:
                all_traj.append((np.array(world_path), color))

    if not scatter:
        print("[WARN] No data"); return

    # Axis limits from ROI scatter (stable; trajectories can extend a bit)
    all_pts = np.vstack([p for p, _ in scatter.values()])
    xlim, ylim = _axis_limits(all_pts, margin=0.08)

    fig, ax = _setup_figure(xlim, ylim,
                            f"GT vehicle trajectories — Sequence {seq.id}")

    # Background
    for cam_id, (pts, color) in scatter.items():
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=0.3, color=color[:3], alpha=0.15,
                   linewidths=0, rasterized=True)

    # Trajectories
    for world_path, color in all_traj:
        # Light IQR filter per track to remove projection outliers
        mask = np.ones(len(world_path), dtype=bool)
        for axis in range(2):
            vals = world_path[:, axis]
            q1, q3 = np.percentile(vals, [10, 90])
            iqr = q3 - q1
            if iqr > 0:
                mask &= (vals >= q1 - 3*iqr) & (vals <= q3 + 3*iqr)
        wp = world_path[mask]
        if len(wp) >= 2:
            ax.plot(wp[:, 0], wp[:, 1],
                    color=color[:3], linewidth=0.9, alpha=0.6)

    # Legend
    cam_colors = [(cid, scatter[cid][1]) for cid in sorted(scatter)]
    legend = [mpatches.Patch(facecolor=c[:3], label=cid) for cid, c in cam_colors]
    ax.legend(handles=legend, loc="upper right", fontsize=8,
              framealpha=0.85, markerscale=8)

    plt.tight_layout()
    out = output_dir / f"{seq.id}_gt_trajectories.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--seq",       default=None, help="e.g. S01 (default: all)")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--viz", nargs="+", default=["scene", "photo", "gt"],
                   choices=["scene", "photo", "gt"])
    p.add_argument("--step",     type=int, default=PIXEL_STEP)
    p.add_argument("--n-frames", type=int, default=N_FRAMES)
    args = p.parse_args()

    dataset = AICityDataset(args.data_root)
    seq_ids = [args.seq] if args.seq else sorted(dataset.sequences)
    out     = Path(args.output_dir)

    for sid in seq_ids:
        if sid not in dataset.sequences:
            print(f"[WARN] {sid} not found"); continue
        seq = dataset.sequences[sid]
        print(f"\n{'='*50}\nSequence {sid}  ({len(seq.cameras)} cameras)\n{'='*50}")

        if "scene" in args.viz:
            print("\n[scene map]")
            draw_scene_map(seq, out, step=args.step)

        if "photo" in args.viz:
            print("\n[photo mosaic]")
            draw_photo_mosaic(seq, out, step=args.step, n_frames=args.n_frames)

        if "gt" in args.viz:
            print("\n[GT trajectories]")
            draw_gt_trajectories(seq, out, step=args.step)


if __name__ == "__main__":
    main()
