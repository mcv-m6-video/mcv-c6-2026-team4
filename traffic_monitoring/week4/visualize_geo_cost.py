"""
Visualizations for the two geo-cost regimes in ClusteringAssociator.

Figure 1 — Co-temporal (overlapping) tracks:
  Both cameras observe the car at the same time. Concurrent observations
  are linked with thin lines to show the spatial distance being measured.

Figure 2 — Sequential (non-overlapping) tracks:
  Track A ends before Track B starts. The endpoint distance (last obs of A
  to first obs of B) is shown as a dashed line.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

RNG = np.random.default_rng(42)

# ── colour palette ──────────────────────────────────────────────────────────
CA_COLOR  = "#2563EB"   # blue  — camera A track
CB_COLOR  = "#DC2626"   # red   — camera B track
OBS_COLOR = "white"
LINK_COLOR = "#6B7280"  # gray  — concurrent-observation links
SEQ_COLOR  = "#16A34A"  # green — endpoint-to-endpoint distance line

# ── helpers ─────────────────────────────────────────────────────────────────

def smooth_path(pts):
    """Return a dense smooth path through sparse waypoints via linear interp."""
    t = np.linspace(0, 1, len(pts))
    t_dense = np.linspace(0, 1, 300)
    x = np.interp(t_dense, t, pts[:, 0])
    y = np.interp(t_dense, t, pts[:, 1])
    return x, y


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Co-temporal tracks
# ═══════════════════════════════════════════════════════════════════════════

def fig_cotemporal():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Layout (all x-positions in "time" order, left → right):
    #
    #   A: [a0  a1] [a2  a3  a4] (A starts 2 nodes before overlap, ends at overlap)
    #   B:          [b0  b1  b2] [b3  b4]  (B starts at overlap, ends 2 nodes after)
    #
    # Overlap covers indices a2–a4 ↔ b0–b2 (3 concurrent pairs).

    # Track A — enters scene first, slightly above centre
    obs_a = np.array([
        [0.8, 3.2],   # a0 — A only
        [1.8, 3.0],   # a1 — A only
        [2.8, 2.75],  # a2 — overlap start
        [3.8, 2.5],   # a3 — overlap
        [4.8, 2.25],  # a4 — overlap end (A finishes here)
    ])
    # Track B — enters scene during A's overlap, slightly below centre
    obs_b = np.array([
        [2.8, 2.1],   # b0 — overlap start
        [3.8, 1.9],   # b1 — overlap
        [4.8, 1.7],   # b2 — overlap end
        [5.8, 1.55],  # b3 — B only
        [6.8, 1.45],  # b4 — B only
    ])

    obs_a += RNG.normal(0, 0.05, obs_a.shape)
    obs_b += RNG.normal(0, 0.05, obs_b.shape)

    # Dense paths
    xa, ya = smooth_path(obs_a)
    xb, yb = smooth_path(obs_b)

    ax.plot(xa, ya, color=CA_COLOR, lw=2.0, zorder=2)
    ax.plot(xb, yb, color=CB_COLOR, lw=2.0, zorder=2)

    # Mark all observation points
    ax.scatter(obs_a[:, 0], obs_a[:, 1], s=60, color=CA_COLOR,
               edgecolors="white", linewidths=1.0, zorder=4)
    ax.scatter(obs_b[:, 0], obs_b[:, 1], s=60, color=CB_COLOR,
               edgecolors="white", linewidths=1.0, zorder=4)

    # Concurrent pairs: a2–a4 ↔ b0–b2
    overlap_a = obs_a[2:]   # indices 2,3,4
    overlap_b = obs_b[:3]   # indices 0,1,2
    for pa, pb in zip(overlap_a, overlap_b):
        ax.plot(
            [pa[0], pb[0]], [pa[1], pb[1]],
            color=LINK_COLOR, lw=0.9, ls="--", alpha=0.75, zorder=3,
        )

    # Annotations
    ax.text(obs_a[0, 0] - 0.1, obs_a[0, 1] + 0.22, "Track A",
            color=CA_COLOR, fontsize=10, fontweight="bold")
    ax.text(obs_b[-1, 0] + 0.1, obs_b[-1, 1] - 0.22, "Track B",
            color=CB_COLOR, fontsize=10, fontweight="bold")

    # Overlap bracket
    x_overlap_start = overlap_a[0, 0] - 0.1
    x_overlap_end   = overlap_a[-1, 0] + 0.1
    y_bracket = 1.1
    ax.annotate(
        "", xy=(x_overlap_end, y_bracket), xytext=(x_overlap_start, y_bracket),
        arrowprops=dict(arrowstyle="<->", color=LINK_COLOR, lw=1.2),
    )
    ax.text(
        (x_overlap_start + x_overlap_end) / 2, y_bracket - 0.22,
        "temporal overlap\n(concurrent observations linked)",
        color=LINK_COLOR, fontsize=8, ha="center", va="top",
    )

    ax.set_title(
        "Geo cost — Co-temporal regime\n"
        r"$\mathit{mean\ world{-}distance\ at\ sampled\ timestamps}$",
        fontsize=11, pad=10,
    )
    ax.set_xlim(0.2, 8.0)
    ax.set_ylim(0.5, 4.0)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Sequential (non-overlapping) tracks
# ═══════════════════════════════════════════════════════════════════════════

def fig_sequential():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_aspect("equal")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Track A — appears first in time, left side of scene
    obs_a = np.array([[0.8, 3.5], [1.4, 3.2], [2.0, 2.8], [2.5, 2.4]])
    obs_a += RNG.normal(0, 0.05, obs_a.shape)

    # Track B — appears later, car has crossed to a different camera zone
    obs_b = np.array([[4.2, 1.8], [4.9, 1.6], [5.6, 1.5], [6.2, 1.6]])
    obs_b += RNG.normal(0, 0.05, obs_b.shape)

    xa, ya = smooth_path(obs_a)
    xb, yb = smooth_path(obs_b)

    ax.plot(xa, ya, color=CA_COLOR, lw=2.0, zorder=2)
    ax.plot(xb, yb, color=CB_COLOR, lw=2.0, zorder=2)

    # Observation dots
    ax.scatter(obs_a[:, 0], obs_a[:, 1], s=60, color=CA_COLOR,
               edgecolors="white", linewidths=1.0, zorder=4)
    ax.scatter(obs_b[:, 0], obs_b[:, 1], s=60, color=CB_COLOR,
               edgecolors="white", linewidths=1.0, zorder=4)

    # Endpoint distance line: last obs of A → first obs of B
    end_a  = obs_a[-1]
    start_b = obs_b[0]
    ax.plot(
        [end_a[0], start_b[0]],
        [end_a[1], start_b[1]],
        color=SEQ_COLOR, lw=1.5, ls="--", zorder=3,
    )
    # Midpoint label
    mid = (end_a + start_b) / 2
    ax.text(
        mid[0], mid[1] + 0.2, "endpoint\ndistance  d",
        color=SEQ_COLOR, fontsize=8.5, ha="center", va="bottom",
    )

    # Highlight the specific endpoint and start point
    ax.scatter(*end_a,   s=90, color=CA_COLOR,  edgecolors=SEQ_COLOR,
               linewidths=1.8, zorder=5)
    ax.scatter(*start_b, s=90, color=CB_COLOR,  edgecolors=SEQ_COLOR,
               linewidths=1.8, zorder=5)

    # Time arrow below
    ax.annotate(
        "time →",
        xy=(6.5, 0.8), fontsize=9, color="#374151",
        ha="right", style="italic",
    )

    # Track labels
    ax.text(obs_a[0, 0], obs_a[0, 1] + 0.28, "Track A  (ends at $t_1$)",
            color=CA_COLOR, fontsize=10, fontweight="bold")
    ax.text(obs_b[-1, 0] + 0.1, obs_b[-1, 1] + 0.2, "Track B  (starts at $t_2 > t_1$)",
            color=CB_COLOR, fontsize=10, fontweight="bold", ha="right")

    # Gap annotation
    y_gap = 0.95
    ax.annotate(
        "", xy=(start_b[0], y_gap), xytext=(end_a[0], y_gap),
        arrowprops=dict(arrowstyle="<->", color="#6B7280", lw=1.2),
    )
    ax.text(
        (end_a[0] + start_b[0]) / 2, y_gap - 0.22,
        r"$\Delta t = t_2 - t_1$  (time gap)",
        color="#6B7280", fontsize=8, ha="center", va="top",
    )

    ax.set_title(
        "Geo cost — Sequential regime\n"
        r"$\mathit{cost} = \min\!\left(\dfrac{d}{v_{\max} \cdot \Delta t},\; 1\right)$",
        fontsize=11, pad=10,
    )
    ax.set_xlim(0.2, 7.0)
    ax.set_ylim(0.5, 4.2)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    f1 = fig_cotemporal()
    f2 = fig_sequential()

    f1.savefig("geo_cost_cotemporal.png", dpi=150, bbox_inches="tight")
    f2.savefig("geo_cost_sequential.png", dpi=150, bbox_inches="tight")
    print("Saved: geo_cost_cotemporal.png, geo_cost_sequential.png")

    plt.show()
