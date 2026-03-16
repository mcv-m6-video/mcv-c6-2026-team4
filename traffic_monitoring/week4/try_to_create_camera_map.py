"""
Camera FOV map generator for the AI City Challenge 2022 dataset.

For each sequence, produces a two-panel figure:
  • Overview (top)  – all cameras, auto-scaled axes so the full extent fits.
  • Zoom     (bot)  – equal-aspect close-up of the dense cluster of cameras,
                      defined by the inter-percentile range of footprint centroids.

A dashed rectangle on the overview shows exactly what the zoom panel covers.

Usage:
    python try_to_create_camera_map.py

Output is saved to  output/camera_maps/<SEQ>_camera_map.png
and also displayed interactively.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon, Rectangle

from src.dataset import AICityDataset, Camera, Sequence

# ── Tunable parameters ────────────────────────────────────────────────────────

# How aggressively to remove perspective-blown-up outlier world points.
# Lower = stricter (1.5 keeps the "core" footprint; 3.0 keeps more far-field).
OUTLIER_IQR_FACTOR: float = 1.5

# Every Nth contour pixel is projected (speed vs. hull resolution trade-off).
CONTOUR_STEP: int = 5

# Zoom panel: centroid percentile range used to define the dense-cluster window.
# 75 means the zoom spans from the 25th to the 75th percentile of all footprint
# centroids, so cameras with very distant footprints don't dominate the view.
ZOOM_CENTROID_PERCENTILE: int = 75

# Output directory (relative to the script location)
OUTPUT_DIR = Path("output/camera_maps")


# ── Geometry ──────────────────────────────────────────────────────────────────

def _roi_contour_pixels(roi_path: Path, step: int) -> np.ndarray:
    """
    Load roi.jpg, threshold away JPEG compression artefacts, collect ALL outer
    contour pixels from ALL disconnected ROI regions, and sub-sample.

    Returns (N, 2) float64 array in (x=col, y=row) pixel order.
    """
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return np.empty((0, 2), dtype=np.float64)

    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)

    pts = np.vstack([c.reshape(-1, 2) for c in contours])
    return pts[::step].astype(np.float64)


def _project_to_world(image_pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Project image pixel coordinates to world coordinates.

    NOTE: calibration.txt stores H as world→pixel, so we invert it here
    to get the pixel→world mapping.
    Points at or beyond the vanishing line (|w| < ε) are discarded.

    Returns (M, 2) float64 array of (x_world, y_world).
    """
    H_inv = np.linalg.inv(H)
    ones = np.ones((len(image_pts), 1))
    pts_h = np.hstack([image_pts, ones])          # (N, 3)
    world_h = (H_inv @ pts_h.T).T                  # (N, 3)
    w = world_h[:, 2]
    valid = np.abs(w) > 1e-6
    return world_h[valid, :2] / w[valid, np.newaxis]


def _iqr_filter(pts: np.ndarray, factor: float) -> np.ndarray:
    """
    Remove points that are outliers on either axis using the IQR rule.
    Keeps points in [Q1 - factor*IQR, Q3 + factor*IQR] for x and y.
    """
    mask = np.ones(len(pts), dtype=bool)
    for axis in range(2):
        vals = pts[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask &= (vals >= q1 - factor * iqr) & (vals <= q3 + factor * iqr)
    return pts[mask]


def camera_footprint(
    cam: Camera,
    step: int = CONTOUR_STEP,
    iqr_factor: float = OUTLIER_IQR_FACTOR,
) -> Optional[np.ndarray]:
    """
    Compute the convex hull of a camera's observable ground footprint in world
    coordinates.

    Returns (K, 2) float64 array of hull vertices, or None on failure.
    """
    img_pts = _roi_contour_pixels(cam.roi_path, step)
    if len(img_pts) < 3:
        return None

    world_pts = _project_to_world(img_pts, cam.calibration.homography)
    world_pts = _iqr_filter(world_pts, iqr_factor)
    if len(world_pts) < 3:
        return None

    hull = cv2.convexHull(world_pts.astype(np.float32))  # (K, 1, 2)
    return hull.reshape(-1, 2).astype(np.float64)


# ── Colours ───────────────────────────────────────────────────────────────────

def _palette(n: int) -> list:
    if n <= 10:
        return [plt.cm.tab10(i / 10) for i in range(n)]
    if n <= 20:
        return [plt.cm.tab20(i / 20) for i in range(n)]
    # For very large sequences (S04 has 25 cameras) cycle through tab20
    base = [plt.cm.tab20(i / 20) for i in range(20)]
    return [base[i % 20] for i in range(n)]


# ── Visualisation ─────────────────────────────────────────────────────────────

def _zoom_limits(
    footprints: dict,
    centroid_percentile: int,
) -> tuple[tuple, tuple]:
    """
    Derive the zoom-panel axis limits from the inter-percentile range of
    footprint centroids plus a margin equal to half the average footprint span.
    This keeps wide-angle cameras (whose footprints extend far) from dominating
    the zoom window.
    """
    centroids = np.array([fp.mean(axis=0) for fp, _ in footprints.values()])
    lo = 100 - centroid_percentile
    hi = centroid_percentile
    x_lo, x_hi = np.percentile(centroids[:, 0], [lo, hi])
    y_lo, y_hi = np.percentile(centroids[:, 1], [lo, hi])

    # Expand by half the average footprint span so full shapes are visible
    avg_span = np.mean([np.ptp(fp, axis=0) for fp, _ in footprints.values()], axis=0)
    x_margin = 0.10 * max(x_hi - x_lo, 1) + 0.5 * avg_span[0]
    y_margin = 0.10 * max(y_hi - y_lo, 1) + 0.5 * avg_span[1]

    zm_xlim = (x_lo - x_margin, x_hi + x_margin)
    zm_ylim = (y_lo - y_margin, y_hi + y_margin)

    # Cap the aspect ratio of the zoom window so elongated road scenes stay
    # readable.  If x:y > max_aspect, expand y symmetrically around its centre.
    max_aspect = 10.0
    x_span = zm_xlim[1] - zm_xlim[0]
    y_span = zm_ylim[1] - zm_ylim[0]
    if y_span > 0 and x_span / y_span > max_aspect:
        y_center = (zm_ylim[0] + zm_ylim[1]) / 2
        y_half = x_span / (2 * max_aspect)
        zm_ylim = (y_center - y_half, y_center + y_half)

    return zm_xlim, zm_ylim


def _add_footprints(ax, footprints: dict, label_xlim=None, label_ylim=None,
                    label_fontsize: int = 7) -> list:
    """
    Draw all footprint polygons onto *ax*.  Labels are only placed when the
    centroid falls inside the optional (label_xlim, label_ylim) window.
    Returns a list of legend patch handles.
    """
    legend_handles = []
    for cam_id, (hull_pts, color) in footprints.items():
        patch = MplPolygon(
            hull_pts, closed=True,
            facecolor=(*color[:3], 0.20),
            edgecolor=(*color[:3], 1.00),
            linewidth=1.5,
        )
        ax.add_patch(patch)

        cx, cy = hull_pts.mean(axis=0)
        in_window = (
            label_xlim is None or (label_xlim[0] <= cx <= label_xlim[1])
        ) and (
            label_ylim is None or (label_ylim[0] <= cy <= label_ylim[1])
        )
        if in_window:
            ax.text(
                cx, cy, cam_id,
                ha="center", va="center",
                fontsize=label_fontsize, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"),
            )

        legend_handles.append(
            mpatches.Patch(facecolor=color[:3], edgecolor=color[:3], label=cam_id)
        )
    return legend_handles


def draw_sequence_map(
    seq: Sequence,
    output_dir: Path = OUTPUT_DIR,
    *,
    step: int = CONTOUR_STEP,
    iqr_factor: float = OUTLIER_IQR_FACTOR,
    zoom_centroid_percentile: int = ZOOM_CENTROID_PERCENTILE,
    figsize: tuple = (16, 11),
) -> None:
    """
    Draw and save a two-panel bird's-eye-view camera map for one sequence.

    Top panel  — Overview: all cameras, auto-scaled axes.
                 A dashed rectangle marks the zoom region.
    Bottom panel — Zoom: equal-aspect close-up of the dense camera cluster.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{seq.id}_camera_map.png"

    cam_items = sorted(seq.cameras.items())
    colors = _palette(len(cam_items))

    # ── compute footprints ────────────────────────────────────────────────────
    footprints: dict[str, tuple[np.ndarray, tuple]] = {}
    for (cam_id, cam), color in zip(cam_items, colors):
        fp = camera_footprint(cam, step=step, iqr_factor=iqr_factor)
        if fp is not None:
            footprints[cam_id] = (fp, color)
        else:
            print(f"  [WARN] {seq.id}/{cam_id}: could not compute footprint, skipping.")

    if not footprints:
        print(f"[ERROR] No valid footprints for sequence {seq.id}.")
        return

    # ── axis limits ───────────────────────────────────────────────────────────
    all_pts = np.vstack([fp for fp, _ in footprints.values()])
    ov_margin = 0.03 * max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
    ov_xlim = (all_pts[:, 0].min() - ov_margin, all_pts[:, 0].max() + ov_margin)
    ov_ylim = (all_pts[:, 1].min() - ov_margin, all_pts[:, 1].max() + ov_margin)

    zm_xlim, zm_ylim = _zoom_limits(footprints, zoom_centroid_percentile)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Camera FOV Map — Sequence {seq.id}  ({len(footprints)} cameras)",
        fontsize=14, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2.5], hspace=0.45)
    ax_ov = fig.add_subplot(gs[0])
    ax_zm = fig.add_subplot(gs[1])

    # ── overview (top) ────────────────────────────────────────────────────────
    ax_ov.set_title("Overview — all cameras (axes auto-scaled, shapes distorted)",
                    fontsize=9, style="italic")
    ax_ov.set_xlim(*ov_xlim)
    ax_ov.set_ylim(*ov_ylim)
    ax_ov.set_xlabel("World X (dataset units)", fontsize=8)
    ax_ov.set_ylabel("World Y", fontsize=8)
    ax_ov.tick_params(labelsize=7)
    ax_ov.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    _add_footprints(ax_ov, footprints, label_fontsize=5)

    # Dashed rectangle marking the zoom region
    rect = Rectangle(
        (zm_xlim[0], zm_ylim[0]),
        zm_xlim[1] - zm_xlim[0],
        zm_ylim[1] - zm_ylim[0],
        linewidth=1.8, edgecolor="black", facecolor="black",
        alpha=0.08, linestyle="--", zorder=5,
    )
    ax_ov.add_patch(rect)
    ax_ov.text(zm_xlim[0], zm_ylim[1], "  ▼ zoom",
               fontsize=7, va="bottom", color="black", fontweight="bold")

    # ── zoom (bottom) ─────────────────────────────────────────────────────────
    ax_zm.set_title(
        f"Zoom — dense cluster  "
        f"(equal aspect · centroid {zoom_centroid_percentile}th-pct window)",
        fontsize=9, style="italic",
    )
    ax_zm.set_aspect("equal", adjustable="box")
    ax_zm.set_xlim(*zm_xlim)
    ax_zm.set_ylim(*zm_ylim)
    ax_zm.set_xlabel("World X (dataset units)", fontsize=8)
    ax_zm.set_ylabel("World Y (dataset units)", fontsize=8)
    ax_zm.tick_params(labelsize=7)
    ax_zm.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)

    legend_handles = _add_footprints(
        ax_zm, footprints,
        label_xlim=zm_xlim, label_ylim=zm_ylim,
        label_fontsize=7,
    )

    ncol = max(1, len(legend_handles) // 15 + 1)
    ax_zm.legend(handles=legend_handles, loc="upper right",
                 fontsize=7, ncol=ncol, framealpha=0.85)

    # ── save & show ───────────────────────────────────────────────────────────
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


# ── ROI pixel scatter plot ────────────────────────────────────────────────────

def _roi_white_pixels(roi_path: Path, step: int) -> np.ndarray:
    """
    Return (N, 2) float64 array of (x=col, y=row) coordinates for every white
    pixel in the thresholded ROI mask, subsampled by *step* in both dimensions.
    """
    roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
    if roi is None:
        return np.empty((0, 2), dtype=np.float64)
    _, binary = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)
    # Subsample the mask before extracting pixel coords (fast)
    binary_sub = binary[::step, ::step]
    rows, cols = np.where(binary_sub > 0)
    # Scale back to original pixel coords
    return np.column_stack([cols * step, rows * step]).astype(np.float64)


def draw_roi_pixel_scatter(
    seq: Sequence,
    output_dir: Path = OUTPUT_DIR,
    *,
    pixel_step: int = 5,
    iqr_factor: float = OUTLIER_IQR_FACTOR,
    point_size: float = 0.5,
    alpha: float = 0.3,
    figsize: tuple = (16, 10),
) -> None:
    """
    For each camera project every (subsampled) white ROI pixel to world space
    via inv(H) and scatter-plot the result.  Equal aspect ratio, no distortion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{seq.id}_roi_scatter.png"

    cam_items = sorted(seq.cameras.items())
    colors = _palette(len(cam_items))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_title(
        f"ROI pixel projection — Sequence {seq.id}  "
        f"({len(cam_items)} cameras, 1/{pixel_step}² subsampling)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("World X (dataset units)")
    ax.set_ylabel("World Y (dataset units)")
    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5)

    legend_handles = []
    for (cam_id, cam), color in zip(cam_items, colors):
        img_pts = _roi_white_pixels(cam.roi_path, step=pixel_step)
        if len(img_pts) == 0:
            print(f"  [WARN] {cam_id}: empty ROI, skipping.")
            continue

        world_pts = _project_to_world(img_pts, cam.calibration.homography)
        world_pts = _iqr_filter(world_pts, iqr_factor)
        if len(world_pts) == 0:
            print(f"  [WARN] {cam_id}: all points filtered, skipping.")
            continue

        ax.scatter(
            world_pts[:, 0], world_pts[:, 1],
            s=point_size, color=color[:3], alpha=alpha, linewidths=0,
            rasterized=True,
        )
        print(f"  {cam_id}: {len(world_pts):,} points plotted")

        legend_handles.append(
            mpatches.Patch(facecolor=color[:3], edgecolor=color[:3], label=cam_id)
        )

    ncol = max(1, len(legend_handles) // 15 + 1)
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=7, ncol=ncol, framealpha=0.85,
              markerscale=10)  # enlarge legend markers (scatter points are tiny)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.show()
    plt.close(fig)


# ── Entry points ──────────────────────────────────────────────────────────────

def main_scatter():
    dataset = AICityDataset("../data/AI_CITY_CHALLENGE_2022_TRAIN")

    for seq_id, seq in sorted(dataset.sequences.items()):
        print(f"\nSequence {seq_id}  ({len(seq.cameras)} cameras)")
        draw_roi_pixel_scatter(seq)


def main():
    dataset = AICityDataset("../data/AI_CITY_CHALLENGE_2022_TRAIN")

    for seq_id, seq in sorted(dataset.sequences.items()):
        print(f"\nSequence {seq_id}  ({len(seq.cameras)} cameras)")
        draw_sequence_map(seq)


if __name__ == "__main__":
    main_scatter()
