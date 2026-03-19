"""Plot phase-2 results: linkage sweep and min_track_frames sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV     = "output/experiments_clustering/phase2/results.csv"
OUT_DIR = "output/experiments_clustering/phase2"

SEQS       = ["S01", "S03", "S04"]
METRICS    = ["IDF1", "IDP", "IDR", "MOTA"]
SEQ_COLORS = {"S01": "#2563EB", "S03": "#DC2626", "S04": "#16A34A"}
AVG_COLOR  = "#1F2937"

df = pd.read_csv(CSV)

bar_w = 0.22


def grouped_bars(ax, metric, sub, x_vals, x_labels, group_col):
    n_seq = len(SEQS)
    offsets = np.linspace(-(n_seq - 1) / 2, (n_seq - 1) / 2, n_seq) * bar_w
    x = np.arange(len(x_vals))
    avgs = []
    for di, xv in enumerate(x_vals):
        vals = []
        for si, seq in enumerate(SEQS):
            row = sub[(sub[group_col] == xv) & (sub["seq"] == seq)]
            val = float(row[metric].iloc[0]) if len(row) and not pd.isna(row[metric].iloc[0]) else None
            if val is not None:
                ax.bar(x[di] + offsets[si], val, bar_w,
                       color=SEQ_COLORS[seq], alpha=0.85,
                       label=seq if di == 0 else None)
                vals.append(val)
        avgs.append(np.mean(vals) if vals else np.nan)

    ax.plot(x, avgs, color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4, label="avg")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "MOTA":
        ax.axhline(0, color="gray", lw=0.8, ls=":")
    return avgs


def shared_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ_COLORS[s]) for s in SEQS]
    handles += [plt.Line2D([0], [0], color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4)]
    fig.legend(handles, SEQS + ["avg"],
               loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), framealpha=0.9)


lk_sub  = df[df["name"].str.startswith("iou05_dt45_lk")]
mtf_sub = df[df["name"].str.startswith("iou05_dt45_mtf")]

lk_vals   = ["average", "complete"]
mtf_vals  = [1, 3, 5]

# ── IDF1: two individual figures ─────────────────────────────────────────────
for sub, x_vals, x_labels, group_col, fname, xlabel in [
    (lk_sub,  lk_vals,  lk_vals,             "linkage",         "idf1_linkage.png",    "linkage"),
    (mtf_sub, mtf_vals, [str(v) for v in mtf_vals], "min_track_frames", "idf1_min_track_frames.png", "min_track_frames"),
]:
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped_bars(ax, "IDF1", sub, x_vals, x_labels, group_col)
    ax.set_xlabel(xlabel, fontsize=9)
    shared_legend(fig)
    fig.tight_layout()
    path = f"{OUT_DIR}/{fname}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")

# ── All metrics: 2-col grid (linkage | min_track_frames) ────────────────────
fig, axes = plt.subplots(len(METRICS), 2, figsize=(12, 4 * len(METRICS)), sharey="row")
fig.suptitle("Phase 2 — iou05, dt=0.45", fontsize=11, y=1.01)

for row, metric in enumerate(METRICS):
    for col, (sub, x_vals, x_labels, group_col, xlabel) in enumerate([
        (lk_sub,  lk_vals,  lk_vals,             "linkage",         "linkage"),
        (mtf_sub, mtf_vals, [str(v) for v in mtf_vals], "min_track_frames", "min_track_frames"),
    ]):
        ax = axes[row][col]
        grouped_bars(ax, metric, sub, x_vals, x_labels, group_col)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(f"{metric}", fontsize=9)

shared_legend(fig)
fig.tight_layout()
path = f"{OUT_DIR}/all_metrics.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── Text report ───────────────────────────────────────────────────────────────
def report_sweep(sub, group_col, group_vals):
    for metric in ["IDF1", "HOTA"]:
        print(f"\n  {metric}:")
        print(f"    {group_col:<12}  {'S01':>6}  {'S03':>6}  {'S04':>6}  {'avg':>6}")
        for gv in group_vals:
            vals = []
            for s in SEQS:
                v = sub[(sub[group_col] == gv) & (sub["seq"] == s)][metric]
                vals.append(float(v.iloc[0]) if len(v) and not pd.isna(v.iloc[0]) else float("nan"))
            avg = np.nanmean(vals)
            print(f"    {str(gv):<12}  {vals[0]:>6.2f}  {vals[1]:>6.2f}  {vals[2]:>6.2f}  {avg:>6.2f}")


print("\n── Linkage sweep ──")
report_sweep(lk_sub, "linkage", lk_vals)

print("\n── min_track_frames sweep ──")
report_sweep(mtf_sub, "min_track_frames", mtf_vals)

best_avg, best_cfg = -1, None
for mtf in mtf_vals:
    vals = [float(mtf_sub[(mtf_sub["min_track_frames"] == mtf) & (mtf_sub["seq"] == s)]["IDF1"].iloc[0])
            for s in SEQS]
    avg = np.mean(vals)
    if avg > best_avg:
        best_avg, best_cfg = avg, mtf

print(f"\nBest overall: linkage=average, min_track_frames={best_cfg}, dt=0.45, iou05  (avg IDF1={best_avg:.2f})")
