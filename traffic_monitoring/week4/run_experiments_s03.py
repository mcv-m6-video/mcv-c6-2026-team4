"""
Experiment runner for sequence S03.

S03 has 6 cameras covering a street intersection + adjacent street sections.
Key differences from S01:
  - 71% of predicted IDs are single-camera → discarded by removeOutliersSingleCam
  - DetA=7.5% despite 48K detection rows (detection rows are being thrown away)
  - AssA=49.57 (excellent — association works when detections are present)
  - Root cause: cars detected in only 1 camera → no cross-camera link possible

Strategy:
  1. Lower conf + lower min_track_frames → more detections per camera → more
     opportunities for cross-camera overlap
  2. More aggressive clustering threshold → force more cross-camera merges
     (accepting some wrong merges to rescue single-camera IDs)
  3. Test both YOLO and FasterRCNN MobileNet

Usage
-----
python run_experiments_s03.py
python run_experiments_s03.py --dry-run
python run_experiments_s03.py --only s03_ref yolo_c030_min1
"""

import argparse
import csv
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PYTHON = str(Path(sys.executable))

BASE_ARGS = [
    "--seq", "S03",
    "--cameras", "c010", "c011", "c012", "c013", "c014", "c015",
    "--gt-path", "../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt",
    "--associator", "clustering",
    "--extractor", "reid",
    "--n-crops", "10",
    "--linkage", "average",
    "--w-reid", "1.0",
    "--w-geo", "0.0",
    "--max-frames", "1",
    # SCT defaults (overridden per experiment)
    "--tracker", "max_overlap",
    "--iou-threshold", "0.30",
    "--max-age", "10",
    "--min-track-frames", "5",
    "--distance-threshold", "0.55",
    "--conf", "0.45",
]

EXPERIMENTS: list[dict] = [
    # -------------------------------------------------------------------------
    # References: S01 best configs run on S03 unchanged
    # -------------------------------------------------------------------------
    {
        "name": "s03_ref_yolo",
        "desc": "S01-best (YOLO, conf=0.45, min5, t=0.55) on S03",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.45", "--min-track-frames", "5"],
    },
    {
        "name": "s03_ref_frcnn",
        "desc": "S01-best (frcnn-mob, conf=0.65, min5, t=0.55) on S03",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.65", "--min-track-frames", "5"],
    },

    # -------------------------------------------------------------------------
    # YOLO — conf + min_track_frames sweep
    # Goal: get more detections per camera so cars appear in 2+ cameras
    # -------------------------------------------------------------------------
    {
        "name": "yolo_c030_min1",
        "desc": "YOLO, conf=0.30, min1",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.30", "--min-track-frames", "1"],
    },
    {
        "name": "yolo_c035_min1",
        "desc": "YOLO, conf=0.35, min1",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.35", "--min-track-frames", "1"],
    },
    {
        "name": "yolo_c030_min3",
        "desc": "YOLO, conf=0.30, min3",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.30", "--min-track-frames", "3"],
    },
    {
        "name": "yolo_c035_min3",
        "desc": "YOLO, conf=0.35, min3",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.35", "--min-track-frames", "3"],
    },
    {
        "name": "yolo_c040_min3",
        "desc": "YOLO, conf=0.40, min3",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.40", "--min-track-frames", "3"],
    },

    # -------------------------------------------------------------------------
    # FasterRCNN MobileNet — conf + min_track_frames sweep
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_c035_min1",
        "desc": "frcnn-mob, conf=0.35, min1",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.35", "--min-track-frames", "1"],
    },
    {
        "name": "frcnn_c045_min1",
        "desc": "frcnn-mob, conf=0.45, min1",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.45", "--min-track-frames", "1"],
    },
    {
        "name": "frcnn_c045_min3",
        "desc": "frcnn-mob, conf=0.45, min3",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.45", "--min-track-frames", "3"],
    },
    {
        "name": "frcnn_c055_min3",
        "desc": "frcnn-mob, conf=0.55, min3",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.55", "--min-track-frames", "3"],
    },

    # -------------------------------------------------------------------------
    # More aggressive clustering threshold
    # Higher t → more merges → more IDs cross 2-camera barrier → better DetA
    # Risk: wrong merges hurt AssA (but AssA is already high, so trade may work)
    # -------------------------------------------------------------------------
    {
        "name": "yolo_c030_min3_t065",
        "desc": "YOLO, conf=0.30, min3, t=0.65",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.30", "--min-track-frames", "3",
                  "--distance-threshold", "0.65"],
    },
    {
        "name": "yolo_c030_min3_t070",
        "desc": "YOLO, conf=0.30, min3, t=0.70",
        "extra": ["--detector", "yolo", "--yolo-weights", "yolov10s_coco.pt",
                  "--conf", "0.30", "--min-track-frames", "3",
                  "--distance-threshold", "0.70"],
    },
    {
        "name": "frcnn_c045_min3_t065",
        "desc": "frcnn-mob, conf=0.45, min3, t=0.65",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.45", "--min-track-frames", "3",
                  "--distance-threshold", "0.65"],
    },
    {
        "name": "frcnn_c045_min3_t070",
        "desc": "frcnn-mob, conf=0.45, min3, t=0.70",
        "extra": ["--detector", "fasterrcnn", "--fasterrcnn-backbone", "mobilenet",
                  "--conf", "0.45", "--min-track-frames", "3",
                  "--distance-threshold", "0.70"],
    },
]

# ---------------------------------------------------------------------------
# Everything below is identical to run_experiments.py
# ---------------------------------------------------------------------------

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
_SUMMARY_RE = re.compile(r"^\s+(\w+)\s*:\s*([\-\d.]+)\s*$", re.MULTILINE)
METRIC_COLS = ["IDF1", "IDP", "IDR", "HOTA", "DetA", "AssA", "MOTA"]


def parse_metrics(text: str) -> dict:
    metrics: dict = {k: None for k in _METRIC_RE}
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


def run_experiment(name, extra_args, output_dir, dry_run):
    run_dir = output_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.txt"

    cmd = [PYTHON, "run_offline_mtmc.py",
           "--output-dir", str(run_dir), *BASE_ARGS, *extra_args]

    print(f"\n{'='*60}\n  RUN: {name}\n  CMD: {' '.join(cmd)}\n{'='*60}")

    if dry_run:
        print("  [DRY RUN — skipped]")
        return {}

    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=Path(__file__).parent)
    output = result.stdout + "\n" + result.stderr
    stdout_path.write_text(output)

    if result.returncode != 0:
        print(f"  [ERROR] exit code {result.returncode}")
        print(output[-2000:])
    else:
        for line in output.strip().splitlines()[-20:]:
            print(f"  {line}")

    return parse_metrics(output)


def fmt(v, pct=True):
    return "—" if v is None else f"{v:.2f}" + ("%" if pct else "")


def write_csv(rows, path):
    fieldnames = ["name", "desc"] + METRIC_COLS
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_markdown(rows, path):
    header = "| Name | Desc | IDF1 | IDP | IDR | HOTA | DetA | AssA | MOTA |"
    sep    = "|------|------|-----:|----:|----:|-----:|-----:|-----:|-----:|"
    lines  = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['desc']} "
            f"| {fmt(r.get('IDF1'))} | {fmt(r.get('IDP'))} | {fmt(r.get('IDR'))} "
            f"| {fmt(r.get('HOTA'))} | {fmt(r.get('DetA'))} | {fmt(r.get('AssA'))} "
            f"| {fmt(r.get('MOTA'))} |"
        )
    path.write_text("\n".join(lines) + "\n")


def _parse_args():
    p = argparse.ArgumentParser(description="S03 experiment runner")
    p.add_argument("--output-dir", default="output/experiments_s03")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--only", nargs="+", metavar="NAME")
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] in args.only]
        if not experiments:
            print(f"[ERROR] No experiments match: {args.only}")
            sys.exit(1)

    print(f"Running {len(experiments)} experiment(s) on S03  →  {out_dir}/")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    rows = []
    for exp in experiments:
        metrics = run_experiment(exp["name"], exp.get("extra", []),
                                 out_dir, args.dry_run)
        row = {"name": exp["name"], "desc": exp["desc"], **metrics}
        rows.append(row)
        write_csv(rows, out_dir / "results.csv")
        write_markdown(rows, out_dir / "results.md")

    print(f"\n{'='*60}\nSUMMARY (S03)\n{'='*60}")
    header = f"{'Name':<28} {'IDF1':>6} {'IDP':>6} {'IDR':>6} {'HOTA':>6} {'DetA':>6} {'AssA':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<28} {fmt(r.get('IDF1')):>6} {fmt(r.get('IDP')):>6} "
              f"{fmt(r.get('IDR')):>6} {fmt(r.get('HOTA')):>6} "
              f"{fmt(r.get('DetA')):>6} {fmt(r.get('AssA')):>6}")

    print(f"\nResults → {out_dir}/results.csv  |  {out_dir}/results.md")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
