"""Plot phase-3 results: appearance vs geo weight sweep."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV     = "output/experiments_clustering/phase3/results.csv"
OUT_DIR = "output/experiments_clustering/phase3"

SEQS       = ["S01", "S03", "S04"]
METRICS    = ["IDF1", "IDP", "IDR", "MOTA"]
SEQ_COLORS = {"S01": "#2563EB", "S03": "#DC2626", "S04": "#16A34A"}
AVG_COLOR  = "#1F2937"

df = pd.read_csv(CSV)
w_geo_vals = sorted(df["w_geo"].unique())
x = np.arange(len(w_geo_vals))
x_labels = [f"({1-w:.1f}, {w:.1f})" for w in w_geo_vals]

bar_w   = 0.22
offsets = np.linspace(-(len(SEQS)-1)/2, (len(SEQS)-1)/2, len(SEQS)) * bar_w


def plot_metric(ax, metric):
    avgs = []
    for di, wg in enumerate(w_geo_vals):
        vals = []
        for si, seq in enumerate(SEQS):
            row = df[(df["w_geo"] == wg) & (df["seq"] == seq)]
            val = float(row[metric].iloc[0]) if len(row) and not pd.isna(row[metric].iloc[0]) else None
            if val is not None:
                ax.bar(x[di] + offsets[si], val, bar_w,
                       color=SEQ_COLORS[seq], alpha=0.85,
                       label=seq if di == 0 else None)
                vals.append(val)
        avgs.append(np.mean(vals) if vals else np.nan)

    ax.plot(x, avgs, color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4, label="avg")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("(w_reid, w_geo)", fontsize=8)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(metric, fontsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if metric == "MOTA":
        ax.axhline(0, color="gray", lw=0.8, ls=":")


def shared_legend(fig):
    handles = [plt.Rectangle((0,0),1,1, color=SEQ_COLORS[s]) for s in SEQS]
    handles += [plt.Line2D([0],[0], color=AVG_COLOR, lw=1.8, ls="--", marker="o", ms=4)]
    fig.legend(handles, SEQS + ["avg"],
               loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), framealpha=0.9)


# ── IDF1: standalone figure ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
plot_metric(ax, "IDF1")
shared_legend(fig)
fig.tight_layout()
path = f"{OUT_DIR}/idf1_geo_sweep.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── All metrics grid ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, metric in zip(axes.flat, METRICS):
    plot_metric(ax, metric)
shared_legend(fig)
fig.tight_layout()
path = f"{OUT_DIR}/all_metrics_geo_sweep.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {path}")

# ── Text report ───────────────────────────────────────────────────────────────
print("\n── Geo weight sweep (IDF1 + HOTA) ──")
print(f"  {'w_reid':>6} {'w_geo':>6}  {'S01':>6}  {'S03':>6}  {'S04':>6}  {'avg IDF1':>9}  {'avg HOTA':>9}")
print("  " + "-" * 60)
for wg in w_geo_vals:
    wr = round(1.0 - wg, 1)
    idf1_vals, hota_vals = [], []
    for seq in SEQS:
        row = df[(df["w_geo"] == wg) & (df["seq"] == seq)]
        if len(row):
            v = row["IDF1"].iloc[0]
            if not pd.isna(v): idf1_vals.append(float(v))
            v = row["HOTA"].iloc[0]
            if not pd.isna(v): hota_vals.append(float(v))
    s01_idf1 = df[(df["w_geo"]==wg)&(df["seq"]=="S01")]["IDF1"].iloc[0]
    s03_idf1 = df[(df["w_geo"]==wg)&(df["seq"]=="S03")]["IDF1"].iloc[0]
    s04_idf1 = df[(df["w_geo"]==wg)&(df["seq"]=="S04")]["IDF1"].iloc[0]
    avg_idf1 = np.mean(idf1_vals) if idf1_vals else float("nan")
    avg_hota = np.mean(hota_vals) if hota_vals else float("nan")
    print(f"  {wr:>6.1f} {wg:>6.1f}  {s01_idf1:>6.2f}  {s03_idf1:>6.2f}  {s04_idf1:>6.2f}  {avg_idf1:>9.2f}  {avg_hota:>9.2f}")
