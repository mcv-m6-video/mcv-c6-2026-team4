"""
Three-phase experiment runner for the clustering associator.

Phase 1 — distance_threshold sweep across both SCT baselines and all 3 sequences.
Phase 2 — linkage and min_track_frames swept independently at the best config
           found in phase 1 (specify with --best-baseline / --best-threshold).
Phase 3 — appearance vs geo weight sweep (w_reid + w_geo = 1, from 1/0 to 0.5/0.5)
           using per-sequence calibrated geo_scale values from world_space_diagnostic.py.

Output
------
Each phase writes to its own directory:
    output/experiments_clustering/phase{N}/
        results.csv        ← one row per (seq × experiment), all param columns
        results.md         ← markdown table
        {seq}_{name}/stdout.txt

Usage
-----
python run_experiments_clustering.py --phase 1
python run_experiments_clustering.py --phase 1 --dry-run
python run_experiments_clustering.py --phase 1 --only S01 S03

python run_experiments_clustering.py --phase 2 --best-baseline iou05 --best-threshold 0.45
python run_experiments_clustering.py --phase 3
python run_experiments_clustering.py --phase 3 --dry-run
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

SEQUENCES = {
    "S01": ["c001", "c002", "c003", "c004", "c005"],
    "S03": ["c010", "c011", "c012", "c013", "c014", "c015"],
    "S04": [f"c{i:03d}" for i in range(16, 41)],  # c016 … c040
}

GT_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt"
PYTHON  = str(Path(sys.executable))

# ---------------------------------------------------------------------------
# Fixed base args (common to every run in both phases)
# ---------------------------------------------------------------------------

BASE_ARGS = [
    "--gt-path",     GT_PATH,
    "--no-anim",
    "--associator",  "clustering",
    "--extractor",   "reid",
    "--tracker",     "sort",
    "--max-age",     "12",
    "--n-crops",     "5",
    "--w-reid",      "1.0",
    "--w-geo",       "0.0",
]

# The two SCT baselines — only differ in IoU threshold
BASELINES = {
    "iou02": {"desc": "SORT iou=0.2 conf=0.65", "args": ["--iou-threshold", "0.2", "--conf", "0.65"]},
    "iou05": {"desc": "SORT iou=0.5 conf=0.65", "args": ["--iou-threshold", "0.5", "--conf", "0.65"]},
}

# Per-sequence geo_scale from world_space_diagnostic.py (mean same-vehicle world distance)
GEO_SCALE = {
    "S01": 0.000084,
    "S03": 0.001051,
    "S04": 0.000238,
}

# Best config from phases 1 + 2
PHASE3_BASELINE      = "iou05"
PHASE3_THRESHOLD     = 0.45
PHASE3_LINKAGE       = "average"
PHASE3_MIN_FRAMES    = 5

# ---------------------------------------------------------------------------
# Phase 1 — distance_threshold sweep
# ---------------------------------------------------------------------------

DISTANCE_THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]

def phase1_experiments():
    exps = []
    for bl_key, bl in BASELINES.items():
        for dt in DISTANCE_THRESHOLDS:
            exps.append({
                "name":               f"{bl_key}_dt{int(dt*100):02d}",
                "desc":               f"{bl['desc']} | dt={dt:.2f} linkage=avg min_frames=1",
                "baseline":           bl_key,
                "distance_threshold": dt,
                "linkage":            "average",
                "min_track_frames":   1,
                "extra": bl["args"] + [
                    "--distance-threshold", str(dt),
                    "--linkage",            "average",
                    "--min-track-frames",   "1",
                ],
            })
    return exps

# ---------------------------------------------------------------------------
# Phase 2 — linkage sweep  +  min_track_frames sweep (independent)
# ---------------------------------------------------------------------------

LINKAGES        = ["average", "complete"]
MIN_TRACK_FRAMES = [1, 3, 5]

def phase2_experiments(best_baseline: str, best_threshold: float):
    bl = BASELINES[best_baseline]
    base = bl["args"] + ["--distance-threshold", str(best_threshold)]

    exps = []

    # ── linkage sweep (min_track_frames fixed at 1) ──────────────────────────
    for lk in LINKAGES:
        exps.append({
            "name":               f"{best_baseline}_dt{int(best_threshold*100):02d}_lk_{lk}",
            "desc":               f"{bl['desc']} | dt={best_threshold:.2f} linkage={lk} min_frames=1",
            "baseline":           best_baseline,
            "distance_threshold": best_threshold,
            "linkage":            lk,
            "min_track_frames":   1,
            "extra": base + ["--linkage", lk, "--min-track-frames", "1"],
        })

    # ── min_track_frames sweep (linkage fixed at average) ───────────────────
    for mtf in MIN_TRACK_FRAMES:
        exps.append({
            "name":               f"{best_baseline}_dt{int(best_threshold*100):02d}_mtf{mtf}",
            "desc":               f"{bl['desc']} | dt={best_threshold:.2f} linkage=avg min_frames={mtf}",
            "baseline":           best_baseline,
            "distance_threshold": best_threshold,
            "linkage":            "average",
            "min_track_frames":   mtf,
            "extra": base + ["--linkage", "average", "--min-track-frames", str(mtf)],
        })

    return exps

# ---------------------------------------------------------------------------
# Phase 3 — appearance vs geo weight sweep
# ---------------------------------------------------------------------------

# (w_reid, w_geo) pairs, sum = 1, from full appearance to 50/50
WEIGHT_PAIRS = [(round(1.0 - w, 1), round(w, 1)) for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]]

def phase3_experiments():
    bl = BASELINES[PHASE3_BASELINE]
    base = (bl["args"] +
            ["--distance-threshold", str(PHASE3_THRESHOLD),
             "--linkage",            PHASE3_LINKAGE,
             "--min-track-frames",   str(PHASE3_MIN_FRAMES)])
    exps = []
    for w_reid, w_geo in WEIGHT_PAIRS:
        exps.append({
            "name":               f"geo_wr{int(w_reid*10)}_wg{int(w_geo*10)}",
            "desc":               (f"{bl['desc']} | dt={PHASE3_THRESHOLD} "
                                   f"w_reid={w_reid} w_geo={w_geo}"),
            "baseline":           PHASE3_BASELINE,
            "distance_threshold": PHASE3_THRESHOLD,
            "linkage":            PHASE3_LINKAGE,
            "min_track_frames":   PHASE3_MIN_FRAMES,
            "w_reid":             w_reid,
            "w_geo":              w_geo,
            # geo_scale injected per-sequence in run_one; base args set weights
            "extra": base + [
                "--w-reid", str(w_reid),
                "--w-geo",  str(w_geo),
            ],
        })
    return exps

# ---------------------------------------------------------------------------
# Metric parsing  (mirrors run_experiments_s03.py)
# ---------------------------------------------------------------------------

_SUMMARY_RE = re.compile(r"^\s+(\w+)\s*:\s*([\-\d.]+)\s*$", re.MULTILINE)
_METRIC_RE  = {
    "IDF1": re.compile(r"IDF1\s*[:\|]\s*([\d.]+)"),
    "IDP":  re.compile(r"IDP\s*[:\|]\s*([\d.]+)"),
    "IDR":  re.compile(r"IDR\s*[:\|]\s*([\d.]+)"),
    "MOTA": re.compile(r"MOTA\s*[:\|]\s*([\-\d.]+)"),
    "MOTP": re.compile(r"MOTP\s*[:\|]\s*([\d.]+)"),
    "HOTA": re.compile(r"HOTA\s*[:\|]\s*([\d.]+)"),
    "DetA": re.compile(r"DetA\s*[:\|]\s*([\d.]+)"),
    "AssA": re.compile(r"AssA\s*[:\|]\s*([\d.]+)"),
}
METRIC_COLS = ["IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "MOTA"]
PARAM_COLS  = ["seq", "baseline", "distance_threshold", "linkage", "min_track_frames",
               "w_reid", "w_geo"]

def parse_metrics(text: str) -> dict:
    metrics = {k: None for k in _METRIC_RE}
    for m in _SUMMARY_RE.finditer(text):
        key = m.group(1).upper()
        if key in metrics:
            try:
                metrics[key] = float(m.group(2))
            except ValueError:
                pass
    for key, pat in _METRIC_RE.items():
        if metrics[key] is None:
            m = pat.search(text)
            if m:
                try:
                    metrics[key] = float(m.group(1))
                except ValueError:
                    pass
    return metrics

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_one(seq: str, exp: dict, out_dir: Path, dry_run: bool) -> dict:
    cameras  = SEQUENCES[seq]
    run_dir  = out_dir / f"{seq}_{exp['name']}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Inject per-sequence geo_scale when geo cost is active
    seq_extra = []
    if exp.get("w_geo", 0.0) > 0.0 and seq in GEO_SCALE:
        seq_extra = ["--geo-scale", str(GEO_SCALE[seq])]

    cmd = [
        PYTHON, "run_offline_mtmc.py",
        "--seq",      seq,
        "--cameras",  *cameras,
        "--output-dir", str(run_dir),
        *BASE_ARGS,
        *exp["extra"],
        *seq_extra,
    ]

    label = f"{seq} / {exp['name']}"
    print(f"\n{'='*64}\n  {label}\n  {exp['desc']}\n{'='*64}")
    print("  CMD:", " ".join(cmd))

    if dry_run:
        print("  [DRY RUN — skipped]")
        return _empty_row(seq, exp)

    t0     = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=Path(__file__).parent)
    elapsed = time.monotonic() - t0

    output = result.stdout + "\n" + result.stderr
    (run_dir / "stdout.txt").write_text(output)

    if result.returncode != 0:
        print(f"  [ERROR] exit={result.returncode}  ({elapsed:.0f}s)")
        print(output[-2000:])
        return _empty_row(seq, exp, success=False, runtime=elapsed)

    for line in output.strip().splitlines()[-20:]:
        print(f"  {line}")
    print(f"  ({elapsed:.0f}s)")

    metrics = parse_metrics(output)
    return {
        "name":               exp["name"],
        "desc":               exp["desc"],
        "seq":                seq,
        "baseline":           exp["baseline"],
        "distance_threshold": exp["distance_threshold"],
        "linkage":            exp["linkage"],
        "min_track_frames":   exp["min_track_frames"],
        "w_reid":             exp.get("w_reid", 1.0),
        "w_geo":              exp.get("w_geo",  0.0),
        "success":            True,
        "runtime_s":          round(elapsed),
        **metrics,
    }


def _empty_row(seq, exp, success=True, runtime=0):
    return {
        "name":               exp["name"],
        "desc":               exp["desc"],
        "seq":                seq,
        "baseline":           exp["baseline"],
        "distance_threshold": exp["distance_threshold"],
        "linkage":            exp["linkage"],
        "min_track_frames":   exp["min_track_frames"],
        "w_reid":             exp.get("w_reid", 1.0),
        "w_geo":              exp.get("w_geo",  0.0),
        "success":            success,
        "runtime_s":          round(runtime),
        **{k: None for k in METRIC_COLS},
    }

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

ALL_COLS = ["name", "desc"] + PARAM_COLS + ["success", "runtime_s"] + METRIC_COLS

def fmt(v):
    return "—" if v is None else f"{v:.2f}"

def write_csv(rows: list, path: Path):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def write_markdown(rows: list, path: Path):
    header = ("| Seq | Baseline | d_thr | Linkage | min_fr "
              "| IDF1 | IDP | IDR | HOTA | DetA | AssA | MOTA |")
    sep    = ("|-----|----------|------:|---------|-------:"
              "|-----:|----:|----:|-----:|-----:|-----:|-----:|")
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['seq']} | {r['baseline']} "
            f"| {r['distance_threshold']:.2f} | {r['linkage']} | {r['min_track_frames']} "
            f"| {fmt(r.get('IDF1'))} | {fmt(r.get('IDP'))} | {fmt(r.get('IDR'))} "
            f"| {fmt(r.get('HOTA'))} | {fmt(r.get('DetA'))} | {fmt(r.get('AssA'))} "
            f"| {fmt(r.get('MOTA'))} |"
        )
    path.write_text("\n".join(lines) + "\n")

def print_summary(rows: list):
    h = f"{'Seq':<4} {'Baseline':<8} {'d_thr':>5} {'Linkage':<9} {'mtf':>3}  {'IDF1':>6} {'HOTA':>6} {'DetA':>6} {'AssA':>6}"
    print(h)
    print("-" * len(h))
    for r in rows:
        print(
            f"{r['seq']:<4} {r['baseline']:<8} {r['distance_threshold']:>5.2f} "
            f"{r['linkage']:<9} {r['min_track_frames']:>3}  "
            f"{fmt(r.get('IDF1')):>6} {fmt(r.get('HOTA')):>6} "
            f"{fmt(r.get('DetA')):>6} {fmt(r.get('AssA')):>6}"
        )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Clustering associator experiment runner")
    p.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--output-dir", default="output/experiments_clustering")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--only",      nargs="+", metavar="SEQ",
                   help="Run only these sequences (e.g. --only S01 S03)")
    # Phase 2 only
    p.add_argument("--best-baseline",  default="iou02",
                   choices=list(BASELINES.keys()),
                   help="[phase 2] Best baseline from phase 1")
    p.add_argument("--best-threshold", type=float, default=0.55,
                   help="[phase 2] Best distance_threshold from phase 1")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args    = _parse_args()
    out_dir = Path(args.output_dir) / f"phase{args.phase}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = list(SEQUENCES.keys())
    if args.only:
        seqs = [s for s in seqs if s in args.only]
        if not seqs:
            print(f"[ERROR] No matching sequences in: {args.only}")
            sys.exit(1)

    if args.phase == 1:
        experiments = phase1_experiments()
        phase_label = "Phase 1 — distance_threshold sweep"
    elif args.phase == 2:
        experiments = phase2_experiments(args.best_baseline, args.best_threshold)
        phase_label = (f"Phase 2 — linkage + min_track_frames sweep "
                       f"(baseline={args.best_baseline}, dt={args.best_threshold:.2f})")
    else:
        experiments = phase3_experiments()
        phase_label = (f"Phase 3 — appearance vs geo weight sweep "
                       f"(baseline={PHASE3_BASELINE}, dt={PHASE3_THRESHOLD}, "
                       f"linkage={PHASE3_LINKAGE}, min_frames={PHASE3_MIN_FRAMES})")

    total = len(seqs) * len(experiments)
    print(f"\n{phase_label}")
    print(f"Sequences : {seqs}")
    print(f"Runs      : {len(experiments)} experiments × {len(seqs)} seqs = {total} runs")
    print(f"Output    : {out_dir}/")
    print(f"Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    rows = []
    for exp in experiments:
        for seq in seqs:
            row = run_one(seq, exp, out_dir, args.dry_run)
            rows.append(row)
            write_csv(rows, out_dir / "results.csv")
            write_markdown(rows, out_dir / "results.md")

    print(f"\n{'='*64}\nSUMMARY — {phase_label}\n{'='*64}")
    print_summary(rows)
    print(f"\nResults → {out_dir}/results.csv  |  {out_dir}/results.md")
    print(f"Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
