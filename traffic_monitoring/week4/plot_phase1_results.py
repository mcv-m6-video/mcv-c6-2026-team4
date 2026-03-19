"""
Plot phase-1 experiment results.

Layout: 4 metrics × 2 baselines = 8 subplots (4 rows, 2 cols).
Each subplot: grouped bars (3 bars = S01, S03, S04) per distance_threshold,
              plus a dashed average line.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV = "output/experiments_clustering/phase1/results.csv"
OUT = "output/experiments_clustering/phase1/phase1_results.png"

METRICS_COMBINED = ["IDP", "IDR", "MOTA"]   # go into the grid figure
SEQS      = ["S01", "S03", "S04"]
BASELINES = ["iou02", "iou05"]
BL_LABELS = {"iou02": "SORT  IoU=0.2", "iou05": "SORT  IoU=0.5"}

SEQ_COLORS = {"S01": "#2563EB", "S03": "#DC2626", "S04": "#16A34A"}
AVG_COLOR  = "#1F2937"

OUT_DIR = "output/experiments_clustering/phase1"

# ── load ────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV)

thresholds = sorted(df["distance_threshold"].unique())
n_dt  = len(thresholds)
n_seq = len(SEQS)

bar_w   = 0.20
offsets = np.linspace(-(n_seq - 1) / 2, (n_seq - 1) / 2, n_seq) * bar_w
x       = np.arange(n_dt)


def plot_metric(ax, metric, baseline, first_legend=True):
    sub  = df[df["baseline"] == baseline]
    avgs = []
    for di, dt in enumerate(thresholds):
        vals = []
        for si, seq in enumerate(SEQS):
            v = sub.loc[(sub["distance_threshold"] == dt) &
                        (sub["seq"] == seq), metric]
            val = float(v.iloc[0]) if len(v) and not pd.isna(v.iloc[0]) else None
            if val is not None:
                ax.bar(x[di] + offsets[si], val, bar_w,
                       color=SEQ_COLORS[seq], alpha=0.85,
                       label=seq if di == 0 else None)
                vals.append(val)
        avgs.append(np.mean(vals) if vals else np.nan)

    ax.plot(x, avgs, color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4,
            label="avg" if first_legend else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{dt:.2f}" for dt in thresholds], fontsize=8)
    ax.set_xlabel("distance_threshold", fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(metric, fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "MOTA":
        ax.axhline(0, color="gray", lw=0.8, ls=":")


def shared_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ_COLORS[s]) for s in SEQS]
    handles += [plt.Line2D([0], [0], color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4)]
    fig.legend(handles, SEQS + ["avg"],
               loc="upper center", ncol=len(SEQS) + 1,
               fontsize=9, bbox_to_anchor=(0.5, 1.0), framealpha=0.9)


# ── IDF1: one standalone figure per baseline ────────────────────────────────
for baseline in BASELINES:
    fig, ax = plt.subplots(figsize=(7, 4))
    # fig.suptitle("IDF1 — distance_threshold sweep\n"
    #              "(appearance-only, linkage=avg, min_frames=1)",
    #              fontsize=10, y=1.03)
    fig.suptitle("IDF1 — distance_threshold sweep", fontsize=10, y=1.03)
    plot_metric(ax, "IDF1", baseline, first_legend=True)
    shared_legend(fig)
    fig.tight_layout()
    path = f"{OUT_DIR}/idf1_{baseline}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")

# ── Remaining metrics: grid figure (metrics × baselines) ────────────────────
fig, axes = plt.subplots(
    len(METRICS_COMBINED), len(BASELINES),
    figsize=(14, 4 * len(METRICS_COMBINED)),
    sharey="row",
)
fig.suptitle("Phase 1 — distance_threshold sweep (appearance-only, linkage=avg, min_frames=1)",
             fontsize=11, y=1.01)

for row, metric in enumerate(METRICS_COMBINED):
    for col, baseline in enumerate(BASELINES):
        plot_metric(axes[row][col], metric, baseline, first_legend=(row == 0 and col == 0))
        axes[row][col].set_title(f"{metric}", fontsize=9)

shared_legend(fig)
fig.tight_layout()
path = f"{OUT_DIR}/other_metrics.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── print best config ────────────────────────────────────────────────────────
print("\nAverage IDF1 across sequences:")
print(f"  {'Baseline':<8}  {'d_thr':>5}  {'S01':>6}  {'S03':>6}  {'S04':>6}  {'avg':>6}")
print("  " + "-" * 46)

best_avg, best_row = -1, None
for baseline in BASELINES:
    for dt in thresholds:
        vals = []
        row_vals = {}
        for seq in SEQS:
            v = df.loc[(df["baseline"] == baseline) &
                       (df["distance_threshold"] == dt) &
                       (df["seq"] == seq), "IDF1"]
            if len(v) and not pd.isna(v.iloc[0]):
                vals.append(float(v.iloc[0]))
                row_vals[seq] = float(v.iloc[0])
        avg = np.mean(vals) if vals else float("nan")
        marker = "  ◄ BEST" if avg > best_avg and not np.isnan(avg) else ""
        if avg > best_avg and not np.isnan(avg):
            best_avg = avg
            best_row = (baseline, dt)
        s01 = f"{row_vals.get('S01', float('nan')):.2f}"
        s03 = f"{row_vals.get('S03', float('nan')):.2f}"
        s04 = f"{row_vals.get('S04', float('nan')):.2f}"
        print(f"  {baseline:<8}  {dt:>5.2f}  {s01:>6}  {s03:>6}  {s04:>6}  {avg:>6.2f}{marker}")

print(f"\nBest overall: baseline={best_row[0]}, distance_threshold={best_row[1]:.2f}  (avg IDF1={best_avg:.2f})")
