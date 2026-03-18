"""
Automated experiment runner for the offline MTMC pipeline.

Runs a predefined set of hyperparameter combinations, captures stdout,
parses the key metrics, and writes:
    output/experiments/results.csv
    output/experiments/results.md
    output/experiments/<run_name>/stdout.txt

Usage
-----
python run_experiments.py
python run_experiments.py --output-dir output/my_sweep
python run_experiments.py --dry-run          # print commands only, don't execute
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

PYTHON = str(Path(sys.executable))

BASE_ARGS = [
    "--gt-path", "../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt",
    "--associator", "clustering",
    "--extractor", "reid",
    "--tracker", "sort",
    "--n-crops", "5",
    "--min-track-frames", "1",
    "--distance-threshold", "0.55",
    "--linkage", "average",
    "--w-reid", "1.0",
    "--w-geo", "0.0",
    "--max-frames", "1",          # animation only (no video output)
]

EXPERIMENTS_V1: list[dict] = [
    # -------------------------------------------------------------------------
    # 0. Baseline — SORT, threshold=0.55 (best found so far)
    # -------------------------------------------------------------------------
    {
        "name": "base_sort_t055",
        "desc": "Baseline: SORT, threshold=0.55",
        "extra": [],
    },

    # -------------------------------------------------------------------------
    # 1. Distance threshold sweep with SORT
    # -------------------------------------------------------------------------
    {
        "name": "thresh_040",
        "desc": "SORT, threshold=0.40",
        "extra": ["--distance-threshold", "0.40"],
    },
    {
        "name": "thresh_045",
        "desc": "SORT, threshold=0.45",
        "extra": ["--distance-threshold", "0.45"],
    },
    {
        "name": "thresh_050",
        "desc": "SORT, threshold=0.50",
        "extra": ["--distance-threshold", "0.50"],
    },
    {
        "name": "thresh_060",
        "desc": "SORT, threshold=0.60",
        "extra": ["--distance-threshold", "0.60"],
    },
    {
        "name": "thresh_065",
        "desc": "SORT, threshold=0.65",
        "extra": ["--distance-threshold", "0.65"],
    },

    # -------------------------------------------------------------------------
    # 2. More crops → better mean ReID feature
    # -------------------------------------------------------------------------
    {
        "name": "n_crops_10",
        "desc": "n-crops=10, threshold=0.55",
        "extra": ["--n-crops", "10"],
    },

    # -------------------------------------------------------------------------
    # 3. min-track-frames filter (remove short spurious tracklets)
    # -------------------------------------------------------------------------
    {
        "name": "min_frames_3",
        "desc": "min-track-frames=3, threshold=0.55",
        "extra": ["--min-track-frames", "3"],
    },
    {
        "name": "min_frames_5",
        "desc": "min-track-frames=5, threshold=0.55",
        "extra": ["--min-track-frames", "5"],
    },
    {
        "name": "min_frames_3_t060",
        "desc": "min-track-frames=3, threshold=0.60",
        "extra": ["--min-track-frames", "3", "--distance-threshold", "0.60"],
    },

    # -------------------------------------------------------------------------
    # 4. Linkage — complete vs average
    # -------------------------------------------------------------------------
    {
        "name": "linkage_complete",
        "desc": "linkage=complete, threshold=0.55",
        "extra": ["--linkage", "complete"],
    },
    {
        "name": "linkage_complete_t065",
        "desc": "linkage=complete, threshold=0.65",
        "extra": ["--linkage", "complete", "--distance-threshold", "0.65"],
    },

    # -------------------------------------------------------------------------
    # 5. Camera graph feasibility gate
    # -------------------------------------------------------------------------
    {
        "name": "camera_graph",
        "desc": "camera-graph gate, threshold=0.55",
        "extra": ["--camera-graph"],
    },

    # -------------------------------------------------------------------------
    # 6. Geometry weight (small w_geo to break ties)
    # -------------------------------------------------------------------------
    {
        "name": "w_geo_015",
        "desc": "w-geo=0.15, geo-scale=1.0, threshold=0.55",
        "extra": ["--w-geo", "0.15", "--geo-scale", "1.0"],
    },

    # -------------------------------------------------------------------------
    # 7. max_overlap tracker baseline (for comparison)
    # -------------------------------------------------------------------------
    {
        "name": "max_overlap_t055",
        "desc": "max_overlap tracker, threshold=0.55",
        "extra": ["--tracker", "max_overlap"],
    },

    # -------------------------------------------------------------------------
    # 8. Best combination (iterate as sweep results come in)
    # -------------------------------------------------------------------------
    {
        "name": "best_combo",
        "desc": "n-crops=10, min-frames=3, linkage=complete, t=0.60",
        "extra": [
            "--n-crops", "10",
            "--min-track-frames", "3",
            "--linkage", "complete",
            "--distance-threshold", "0.60",
        ],
    },
]

EXPERIMENTS_V2: list[dict] = [
    {
        "name": "min3_crops10_maxoverlap",
        "desc": "min-frames=3, n-crops=10, max_overlap, t=0.55",
        "extra": ["--min-track-frames", "3", "--n-crops", "10", "--tracker", "max_overlap"],
    },
    {
        "name": "min3_crops10_sort",
        "desc": "min-frames=3, n-crops=10, SORT, t=0.55",
        "extra": ["--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "min3_t050",
        "desc": "min-frames=3, threshold=0.50",
        "extra": ["--min-track-frames", "3", "--distance-threshold", "0.50"],
    },
    {
        "name": "min3_t045",
        "desc": "min-frames=3, threshold=0.45",
        "extra": ["--min-track-frames", "3", "--distance-threshold", "0.45"],
    },
]

# ---------------------------------------------------------------------------
# V3: Low IoU threshold + high max_age sweep
#
# Hypothesis: the S03/c10 result (HOTA=79.96) used IoU=0.14, max_age=35,
# min_hits=6 (≈ our min_track_frames). Low IoU lets Kalman/greedy matching
# handle cars that move far between frames; high max_age survives red-light
# stops. Combined with min_track_frames=3 to suppress the extra FP tracks
# a low IoU threshold would otherwise create.
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    # -------------------------------------------------------------------------
    # Best config from V2 as reference point
    # -------------------------------------------------------------------------
    {
        "name": "v3_ref",
        "desc": "Reference: max_overlap, iou=0.45, age=10, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.45",
                  "--max-age", "10", "--min-track-frames", "3", "--n-crops", "10"],
    },

    # -------------------------------------------------------------------------
    # Max-age sweep (keep iou=0.45, min3, crops10, max_overlap)
    # Higher max-age → tracks survive red-light stops and brief occlusions
    # -------------------------------------------------------------------------
    {
        "name": "age20",
        "desc": "max_overlap, iou=0.45, age=20, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.45",
                  "--max-age", "20", "--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "age35",
        "desc": "max_overlap, iou=0.45, age=35, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.45",
                  "--max-age", "35", "--min-track-frames", "3", "--n-crops", "10"],
    },

    # -------------------------------------------------------------------------
    # IoU threshold sweep (keep age=10, min3, crops10, max_overlap)
    # Lower IoU → permissive matching for cars moving fast / braking
    # -------------------------------------------------------------------------
    {
        "name": "iou030",
        "desc": "max_overlap, iou=0.30, age=10, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.30",
                  "--max-age", "10", "--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "iou015",
        "desc": "max_overlap, iou=0.15, age=10, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.15",
                  "--max-age", "10", "--min-track-frames", "3", "--n-crops", "10"],
    },

    # -------------------------------------------------------------------------
    # Combined: low IoU + high max-age (mirroring S03/c10 recipe)
    # -------------------------------------------------------------------------
    {
        "name": "iou015_age35",
        "desc": "max_overlap, iou=0.15, age=35, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.15",
                  "--max-age", "35", "--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "iou030_age20",
        "desc": "max_overlap, iou=0.30, age=20, min3, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.30",
                  "--max-age", "20", "--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "iou015_age35_min5",
        "desc": "max_overlap, iou=0.15, age=35, min5, crops10",
        "extra": ["--tracker", "max_overlap", "--iou-threshold", "0.15",
                  "--max-age", "35", "--min-track-frames", "5", "--n-crops", "10"],
    },

    # -------------------------------------------------------------------------
    # Same combos with SORT (Kalman predictions help with low IoU matching)
    # -------------------------------------------------------------------------
    {
        "name": "sort_iou015_age35",
        "desc": "SORT, iou=0.15, age=35, min3, crops10",
        "extra": ["--tracker", "sort", "--iou-threshold", "0.15",
                  "--max-age", "35", "--min-track-frames", "3", "--n-crops", "10"],
    },
    {
        "name": "sort_iou030_age20",
        "desc": "SORT, iou=0.30, age=20, min3, crops10",
        "extra": ["--tracker", "sort", "--iou-threshold", "0.30",
                  "--max-age", "20", "--min-track-frames", "3", "--n-crops", "10"],
    },
]

# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

# Patterns for the printed evaluation block
_METRIC_RE = {
    "IDF1":  re.compile(r"IDF1\s*[:\|]\s*([\d.]+)"),
    "IDP":   re.compile(r"IDP\s*[:\|]\s*([\d.]+)"),
    "IDR":   re.compile(r"IDR\s*[:\|]\s*([\d.]+)"),
    "MOTA":  re.compile(r"MOTA\s*[:\|]\s*([\-\d.]+)"),
    "MOTP":  re.compile(r"MOTP\s*[:\|]\s*([\d.]+)"),
    "HOTA":  re.compile(r"HOTA\s*[:\|]\s*([\d.]+)"),
    "DetA":  re.compile(r"DetA\s*[:\|]\s*([\d.]+)"),
    "AssA":  re.compile(r"AssA\s*[:\|]\s*([\d.]+)"),
}

# Also pick up the compact "Summary:" block: "  IDF1  : 37.58"
_SUMMARY_RE = re.compile(r"^\s+(\w+)\s*:\s*([\-\d.]+)\s*$", re.MULTILINE)


def parse_metrics(text: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {k: None for k in _METRIC_RE}

    # First try the Summary block (most reliable)
    for m in _SUMMARY_RE.finditer(text):
        key = m.group(1).upper()
        if key in metrics:
            try:
                metrics[key] = float(m.group(2))
            except ValueError:
                pass

    # Fall back to inline patterns for any still-missing metrics
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

def run_experiment(
    name: str,
    extra_args: list[str],
    output_dir: Path,
    dry_run: bool,
) -> dict[str, float | None]:
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"

    cmd = [
        PYTHON, "run_offline_mtmc.py",
        "--output-dir", str(run_dir),
        *BASE_ARGS,
        *extra_args,
    ]

    print(f"\n{'='*60}")
    print(f"  RUN: {name}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return {}

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )

    output = result.stdout + "\n" + result.stderr
    stdout_path.write_text(output)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}")
        print(output[-2000:])  # tail of output for quick diagnosis
    else:
        # Print the tail so the user sees metrics live
        lines = output.strip().splitlines()
        for line in lines[-20:]:
            print(f"  {line}")

    return parse_metrics(output)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRIC_COLS = ["IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "MOTA", "MOTP"]


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = ["name", "desc"] + METRIC_COLS
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fmt(v: float | None, pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}" + ("%" if pct else "")


def write_markdown(rows: list[dict], path: Path) -> None:
    header = "| Name | Desc | IDF1 | IDP | IDR | HOTA | DetA | AssA | MOTA | MOTP |"
    sep    = "|------|------|-----:|----:|----:|-----:|-----:|-----:|-----:|-----:|"
    lines  = [header, sep]
    for r in rows:
        row = (
            f"| {r['name']} | {r['desc']} "
            f"| {fmt(r.get('IDF1'))} | {fmt(r.get('IDP'))} | {fmt(r.get('IDR'))} "
            f"| {fmt(r.get('HOTA'))} | {fmt(r.get('DetA'))} | {fmt(r.get('AssA'))} "
            f"| {fmt(r.get('MOTA'))} | {fmt(r.get('MOTP'))} |"
        )
        lines.append(row)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTMC experiment runner")
    p.add_argument("--output-dir", default="output/experiments")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing")
    p.add_argument("--only", nargs="+", metavar="NAME",
                   help="Run only experiments with these names")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
        if not experiments:
            print(f"[ERROR] No experiments match: {args.only}")
            sys.exit(1)

    print(f"Running {len(experiments)} experiment(s)  →  {out_dir}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    rows: list[dict] = []
    for exp in experiments:
        metrics = run_experiment(
            name=exp["name"],
            extra_args=exp.get("extra", []),
            output_dir=out_dir,
            dry_run=args.dry_run,
        )
        row = {"name": exp["name"], "desc": exp["desc"], **metrics}
        rows.append(row)

        # Incremental write so partial results survive crashes
        write_csv(rows, out_dir / "results.csv")
        write_markdown(rows, out_dir / "results.md")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Name':<25} {'IDF1':>6} {'IDP':>6} {'IDR':>6} {'HOTA':>6} {'DetA':>6} {'AssA':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name']:<25} "
            f"{fmt(r.get('IDF1')):>6} "
            f"{fmt(r.get('IDP')):>6} "
            f"{fmt(r.get('IDR')):>6} "
            f"{fmt(r.get('HOTA')):>6} "
            f"{fmt(r.get('DetA')):>6} "
            f"{fmt(r.get('AssA')):>6}"
        )

    print(f"\nResults saved → {out_dir}/results.csv  |  {out_dir}/results.md")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
