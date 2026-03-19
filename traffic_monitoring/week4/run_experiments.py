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
# V4: Alternative detectors
#
# Best SCT config carried forward: max_overlap, iou=0.30, age=10, min3, crops10
# All experiments use --car-class 2 for standard COCO weights (car=class 2),
# except fasterrcnn which uses class 3 (torchvision COCO-91, background=0).
#
# YOLO26: update --yolo-weights to the correct filename once you have it.
#         The --car-class 2 assumes standard COCO weights; set to 0 if custom.
# ---------------------------------------------------------------------------

BEST_SCT = [
    "--tracker", "max_overlap",
    "--iou-threshold", "0.30",
    "--max-age", "10",
    "--min-track-frames", "3",
    "--n-crops", "10",
]

EXPERIMENTS: list[dict] = [
    # -------------------------------------------------------------------------
    # Reference: current best (yolov10s_coco.pt, car_class=0)
    # -------------------------------------------------------------------------
    {
        "name": "v4_ref",
        "desc": "Reference: yolov10s, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "yolo"],
    },

    # -------------------------------------------------------------------------
    # RT-DETR  (Ultralytics, COCO weights, car_class=2)
    # rtdetr-x  is the largest / most accurate; rtdetr-l is faster
    # -------------------------------------------------------------------------
    {
        "name": "rtdetr_x",
        "desc": "RT-DETR-X, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "rtdetr",
                  "--rtdetr-weights", "rtdetr-x.pt"],
    },
    {
        "name": "rtdetr_l",
        "desc": "RT-DETR-L, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "rtdetr",
                  "--rtdetr-weights", "rtdetr-l.pt"],
    },

    # -------------------------------------------------------------------------
    # Faster R-CNN  (torchvision COCO-91 weights, car_class=3 by default)
    # -------------------------------------------------------------------------
    {
        "name": "fasterrcnn_r50",
        "desc": "Faster R-CNN ResNet50-FPN-v2, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "resnet50"],
    },
    {
        "name": "fasterrcnn_mob",
        "desc": "Faster R-CNN MobileNetV3, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "mobilenet"],
    },

    # -------------------------------------------------------------------------
    # YOLO26  (update --yolo-weights once you have the filename)
    # Using --car-class 2 for standard COCO weights.
    # Change to --car-class 0 if it's a single-class custom model.
    # -------------------------------------------------------------------------
    {
        "name": "yolo26n",
        "desc": "YOLO26 nano, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26n.pt",   # <-- update filename if needed
                  "--car-class", "2"],
    },
    {
        "name": "yolo26s",
        "desc": "YOLO26 small, max_overlap iou=0.30 min3 crops10",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26s.pt",
                  "--car-class", "2"],
    },
]

# ---------------------------------------------------------------------------
# V5: Confidence threshold calibration + YOLO26 larger variants
#
# Key findings from V4:
#   - RT-DETR / FasterRCNN ResNet50 drown in FPs at conf=0.45 → need higher threshold
#   - YOLO26n is too conservative (IDR=27%) → try lower conf
#   - YOLO26s is competitive but IDR still lower than YOLOv10s → try lower conf
#   - FasterRCNN MobileNet has highest AssA ever (40.85) → tune its conf too
#   - Larger YOLO26 models (m, l, x) likely close the IDR gap
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict] = [
    # -------------------------------------------------------------------------
    # YOLO26 — larger model sizes (auto-downloaded by Ultralytics)
    # -------------------------------------------------------------------------
    {
        "name": "yolo26m",
        "desc": "YOLO26 medium, conf=0.45",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26m.pt", "--car-class", "2"],
    },
    {
        "name": "yolo26l",
        "desc": "YOLO26 large, conf=0.45",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26l.pt", "--car-class", "2"],
    },
    {
        "name": "yolo26x",
        "desc": "YOLO26 xlarge, conf=0.45",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26x.pt", "--car-class", "2"],
    },

    # -------------------------------------------------------------------------
    # YOLO26s — confidence threshold sweep (s was our best YOLO26 so far)
    # Lower conf → more detections, better IDR; filter noise with min_frames
    # -------------------------------------------------------------------------
    {
        "name": "yolo26s_c035",
        "desc": "YOLO26 small, conf=0.35",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26s.pt", "--car-class", "2", "--conf", "0.35"],
    },
    {
        "name": "yolo26s_c030",
        "desc": "YOLO26 small, conf=0.30",
        "extra": [*BEST_SCT, "--detector", "yolo",
                  "--yolo-weights", "yolo26s.pt", "--car-class", "2", "--conf", "0.30"],
    },

    # -------------------------------------------------------------------------
    # RT-DETR-L — higher confidence to cut false positives
    # V4 produced 69K rows at conf=0.45 → MOTA=-11; need to find its sweet spot
    # -------------------------------------------------------------------------
    {
        "name": "rtdetr_l_c060",
        "desc": "RT-DETR-L, conf=0.60",
        "extra": [*BEST_SCT, "--detector", "rtdetr",
                  "--rtdetr-weights", "rtdetr-l.pt", "--conf", "0.60"],
    },
    {
        "name": "rtdetr_l_c070",
        "desc": "RT-DETR-L, conf=0.70",
        "extra": [*BEST_SCT, "--detector", "rtdetr",
                  "--rtdetr-weights", "rtdetr-l.pt", "--conf", "0.70"],
    },

    # -------------------------------------------------------------------------
    # FasterRCNN ResNet50 — higher confidence to cut false positives
    # V4 produced 84K rows at conf=0.45 → MOTA=-32
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_r50_c065",
        "desc": "FasterRCNN ResNet50, conf=0.65",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "resnet50", "--conf", "0.65"],
    },
    {
        "name": "frcnn_r50_c075",
        "desc": "FasterRCNN ResNet50, conf=0.75",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "resnet50", "--conf", "0.75"],
    },

    # -------------------------------------------------------------------------
    # FasterRCNN MobileNet — best AssA so far (40.85); tune conf
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_mob_c055",
        "desc": "FasterRCNN MobileNet, conf=0.55",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "mobilenet", "--conf", "0.55"],
    },
    {
        "name": "frcnn_mob_c065",
        "desc": "FasterRCNN MobileNet, conf=0.65",
        "extra": [*BEST_SCT, "--detector", "fasterrcnn",
                  "--fasterrcnn-backbone", "mobilenet", "--conf", "0.65"],
    },
]

# ---------------------------------------------------------------------------
# V6: FasterRCNN MobileNet fine-tuning
#
# Findings from V5:
#   - frcnn_mob is the new best (IDF1=43.53, HOTA=31.49, AssA=42.77 at conf=0.65)
#   - IDR curve: 35.33 (c=0.45) → 36.27 (c=0.55) → 34.60 (c=0.65)
#     Peak IDR is at c=0.55 but IDF1 is nearly identical → find exact peak with c=0.60
#   - AssA keeps improving with higher conf (cleaner crops → better ReID)
#   - min_frames not yet tested with frcnn_mob → worth trying to raise AssA further
# ---------------------------------------------------------------------------

BEST_FRCNN = [
    "--detector", "fasterrcnn",
    "--fasterrcnn-backbone", "mobilenet",
    "--tracker", "max_overlap",
    "--iou-threshold", "0.30",
    "--max-age", "10",
    "--n-crops", "10",
]

EXPERIMENTS: list[dict] = [
    # -------------------------------------------------------------------------
    # Confidence fine-tuning: find the peak between 0.55 and 0.70
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_mob_c060",
        "desc": "FasterRCNN MobileNet, conf=0.60, min3",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.60"],
    },
    {
        "name": "frcnn_mob_c070",
        "desc": "FasterRCNN MobileNet, conf=0.70, min3",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.70"],
    },
    {
        "name": "frcnn_mob_c075",
        "desc": "FasterRCNN MobileNet, conf=0.75, min3",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.75"],
    },

    # -------------------------------------------------------------------------
    # min_frames sweep at best conf candidates
    # Higher min_frames → shorter tracklets pruned → cleaner ReID features
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_mob_c055_min5",
        "desc": "FasterRCNN MobileNet, conf=0.55, min5",
        "extra": [*BEST_FRCNN, "--min-track-frames", "5", "--conf", "0.55"],
    },
    {
        "name": "frcnn_mob_c065_min5",
        "desc": "FasterRCNN MobileNet, conf=0.65, min5",
        "extra": [*BEST_FRCNN, "--min-track-frames", "5", "--conf", "0.65"],
    },
    {
        "name": "frcnn_mob_c060_min5",
        "desc": "FasterRCNN MobileNet, conf=0.60, min5",
        "extra": [*BEST_FRCNN, "--min-track-frames", "5", "--conf", "0.60"],
    },

    # -------------------------------------------------------------------------
    # Best combo + more crops (15) — richer mean ReID feature per tracklet
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_mob_c065_crops15",
        "desc": "FasterRCNN MobileNet, conf=0.65, min3, crops=15",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.65",
                  "--n-crops", "15"],
    },

    # -------------------------------------------------------------------------
    # Clustering threshold fine-tune around the new best detector
    # (was tuned for YOLOv10s; frcnn_mob may have a different sweet spot)
    # -------------------------------------------------------------------------
    {
        "name": "frcnn_mob_c065_t050",
        "desc": "FasterRCNN MobileNet, conf=0.65, min3, t=0.50",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.65",
                  "--distance-threshold", "0.50"],
    },
    {
        "name": "frcnn_mob_c065_t060",
        "desc": "FasterRCNN MobileNet, conf=0.65, min3, t=0.60",
        "extra": [*BEST_FRCNN, "--min-track-frames", "3", "--conf", "0.65",
                  "--distance-threshold", "0.60"],
    },
]

# ---------------------------------------------------------------------------
# V8: Calibrated geo_scale sweep
#
# world_space_diagnostic.py on S01 measured:
#   same-vehicle mean world distance = 0.000130° (≈14m in GPS degrees)
#   same-vehicle p95                 = 0.000479°
#   different-vehicle p5             = 0.000117°
#
# With geo_scale=1.0 (default), d/geo_scale ≈ 0.0001 → cost ≈ 0 → useless.
# With geo_scale=0.000130, d/geo_scale ≈ 1.0 at the mean same-vehicle
# separation → meaningful soft cost.
#
# Base: frcnn-mob best (conf=0.65, min5, t=0.55) — S01 best config.
# Sweep: w_geo in {0.1, 0.2, 0.3, 0.5} × geo_scale in {0.000065, 0.000130, 0.000260}
# ---------------------------------------------------------------------------

BEST_S01 = [
    "--detector", "fasterrcnn",
    "--fasterrcnn-backbone", "mobilenet",
    "--tracker", "max_overlap",
    "--iou-threshold", "0.30",
    "--max-age", "10",
    "--n-crops", "10",
    "--conf", "0.65",
    "--min-track-frames", "5",
]

EXPERIMENTS: list[dict] = [
    # Reference with w_geo=0 so results are self-contained in this output dir
    {
        "name": "geo_ref",
        "desc": "frcnn-mob best, w_geo=0 (reference)",
        "extra": [*BEST_S01, "--w-geo", "0.0"],
    },

    # -------------------------------------------------------------------------
    # geo_scale = same_p50 (0.000078°, ≈8.7m) — tighter scale, harsher penalty
    # -------------------------------------------------------------------------
    {
        "name": "geo_s065_w01",
        "desc": "geo_scale=0.000065, w_geo=0.1",
        "extra": [*BEST_S01, "--w-geo", "0.1", "--geo-scale", "0.000065"],
    },
    {
        "name": "geo_s065_w02",
        "desc": "geo_scale=0.000065, w_geo=0.2",
        "extra": [*BEST_S01, "--w-geo", "0.2", "--geo-scale", "0.000065"],
    },
    {
        "name": "geo_s065_w03",
        "desc": "geo_scale=0.000065, w_geo=0.3",
        "extra": [*BEST_S01, "--w-geo", "0.3", "--geo-scale", "0.000065"],
    },

    # -------------------------------------------------------------------------
    # geo_scale = same_mean (0.000130°, ≈14m) — calibrated to data mean
    # -------------------------------------------------------------------------
    {
        "name": "geo_s130_w01",
        "desc": "geo_scale=0.000130 (same_mean), w_geo=0.1",
        "extra": [*BEST_S01, "--w-geo", "0.1", "--geo-scale", "0.000130"],
    },
    {
        "name": "geo_s130_w02",
        "desc": "geo_scale=0.000130 (same_mean), w_geo=0.2",
        "extra": [*BEST_S01, "--w-geo", "0.2", "--geo-scale", "0.000130"],
    },
    {
        "name": "geo_s130_w03",
        "desc": "geo_scale=0.000130 (same_mean), w_geo=0.3",
        "extra": [*BEST_S01, "--w-geo", "0.3", "--geo-scale", "0.000130"],
    },
    {
        "name": "geo_s130_w05",
        "desc": "geo_scale=0.000130 (same_mean), w_geo=0.5",
        "extra": [*BEST_S01, "--w-geo", "0.5", "--geo-scale", "0.000130"],
    },

    # -------------------------------------------------------------------------
    # geo_scale = same_p95 (0.000479°, ≈53m) — looser scale, gentler penalty
    # -------------------------------------------------------------------------
    {
        "name": "geo_s479_w02",
        "desc": "geo_scale=0.000479 (same_p95), w_geo=0.2",
        "extra": [*BEST_S01, "--w-geo", "0.2", "--geo-scale", "0.000479"],
    },
    {
        "name": "geo_s479_w03",
        "desc": "geo_scale=0.000479 (same_p95), w_geo=0.3",
        "extra": [*BEST_S01, "--w-geo", "0.3", "--geo-scale", "0.000479"],
    },
    {
        "name": "geo_s479_w05",
        "desc": "geo_scale=0.000479 (same_p95), w_geo=0.5",
        "extra": [*BEST_S01, "--w-geo", "0.5", "--geo-scale", "0.000479"],
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
