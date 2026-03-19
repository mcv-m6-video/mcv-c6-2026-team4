# Multi-Camera Multi-Target Tracking (MTMC)

This project implements a multi-camera multi-target tracking system for vehicle tracking across multiple camera views. It was developed for the AI City Challenge 2022, focusing on tracking vehicles across different camera perspectives in traffic scenarios.

Link to the Google Slides Presentation -> [MTMC](https://docs.google.com/presentation/d/1GIh7ZsBZdk1gFRL0EqgLg5Hli3SCXq974R0326KX4UI/edit?usp=sharing)

## Overview

The system performs:

1. **Vehicle Detection** using YOLO detectors
2. **Single-Camera Tracking** using SORT or Max-Overlap trackers
3. **Re-Identification (Re-ID)** feature extraction using deep learning models
4. **Multi-Camera Association** to link tracklets across different cameras
5. **Evaluation** using AI City Challenge metrics (IDF1, IDP, IDR)

## Main Scripts

### 1. `mtmc.py` - Basic Multi-Camera Tracking

Basic implementation of the MTMC pipeline with appearance-based Re-ID features.

**Features:**

- YOLOv10 detection with ROI filtering
- SORT-based single-camera tracking
- ft_net Re-ID feature extraction
- Global gallery matching across cameras
- Automatic evaluation with AI City Challenge metrics

**Usage:**

```bash
python mtmc.py
```

**Configuration:**
Edit the configuration section at the top of the file:

- `AI_CITY_BASE_PATH`: Path to AI City Challenge dataset
- `CAMERAS`: List of camera IDs to process (e.g., `["c001", "c002", "c003"]`)
- `CONF_THRESHOLD`: Detection confidence threshold (default: 0.3)
- `IOU_THRESHOLD`: IoU threshold for tracking (default: 0.3)
- `MAX_AGE`: Maximum frames to keep lost tracks (default: 10)
- `GLOBAL_MATCH_THRESH`: Re-ID similarity threshold (default: 0.6)

**Output:**

- `mtmc_predictions.txt`: Predictions in AI City format
- Console output with evaluation metrics (IDF1, IDP, IDR)

---

### 2. `run_offline_mtmc.py` - Advanced Offline Multi-Camera Tracking

Advanced implementation with extensive configuration options, multiple detector/tracker choices, and animated visualization.

**Features:**

- Multiple detector options: YOLO, RT-DETR, Faster R-CNN
- Multiple tracker options: SORT, Max-Overlap
- Re-ID extractors: ft_net (ResNet-based) or color histograms
- Greedy or clustering-based association strategies
- Camera connectivity graph for physics-based constraints
- Geometric + temporal + appearance features
- Animated world-space visualization with trajectory trails
- Comprehensive evaluation pipeline

**Basic Usage:**

```bash
# Default: Appearance-only (ReID), sequence S01, all cameras
python run_offline_mtmc.py

# Skip animation (faster)
python run_offline_mtmc.py --no-anim

# With evaluation
python run_offline_mtmc.py --gt-path ../data/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt
```

**Advanced Examples:**

```bash
# Geometry + appearance fusion
python run_offline_mtmc.py --w-spatial 0.3 --w-temporal 0.2 \
    --spatial-scale 0.01 --temporal-scale 30

# Custom sequence and cameras
python run_offline_mtmc.py --seq S03 --cameras c010 c011 c012 \
    --w-spatial 0.3 --w-temporal 0.2

# Use clustering associator (order-independent)
python run_offline_mtmc.py --associator clustering \
    --distance-threshold 0.4 --linkage average

# Use RT-DETR detector instead of YOLO
python run_offline_mtmc.py --detector rtdetr --rtdetr-weights rtdetr-x.pt

# Enable camera connectivity graph
python run_offline_mtmc.py --camera-graph --v-min 0.00001 --v-max 0.0002
```

**Key Parameters:**

Detection & Tracking:

- `--detector`: Choose detector (`yolo`, `rtdetr`, `fasterrcnn`)
- `--tracker`: Choose tracker (`sort`, `max_overlap`)
- `--conf`: Detection confidence threshold (default: 0.45)
- `--max-age`: Max frames to keep lost tracks (default: 10)
- `--iou-threshold`: IoU threshold for tracking (default: 0.45)

Feature Extraction:

- `--extractor`: Feature type (`reid` for deep features, `histogram` for color)
- `--n-crops`: Number of crops per track for averaging (default: 5)
- `--min-track-frames`: Min observations to keep tracklet (default: 1)

Association Strategy:

- `--associator`: Strategy (`greedy` or `clustering`)
- `--w-appearance`: Weight for appearance similarity (default: 1.0)
- `--w-spatial`: Weight for spatial proximity (default: 0.0)
- `--w-temporal`: Weight for temporal consistency (default: 0.0)
- `--match-threshold`: Similarity threshold for greedy matching (default: 0.4)

Clustering Options:

- `--distance-threshold`: Linkage distance for merging clusters (default: 0.4)
- `--linkage`: Criterion (`average`, `complete`, `single`)
- `--w-reid`: Weight for Re-ID cost (default: 1.0)
- `--w-geo`: Weight for geometric cost (default: 0.0)

Animation:

- `--no-anim`: Skip animation/video rendering
- `--anim-dt`: Time between animation frames in seconds (default: 0.5)
- `--anim-fps`: Playback FPS of output video (default: 10)
- `--trail-secs`: Trajectory trail duration (default: 5.0)

**Output:**

- `output/tracking/{seq}_predictions.txt`: Predictions in AI City format
- `output/tracking/{seq}_tracking.mp4`: Animated visualization video
- Console output with detailed evaluation metrics

---

### 3. Time Restriction Variants

These scripts add temporal constraints to improve tracking accuracy by exploiting camera timing information.

#### `mtmc_time_restrictions.py`

Extends the basic MTMC pipeline with temporal constraints and optional feature fusion.

**Additional Features:**

- Time-based filtering using camera timestamp files
- Two constraint modes:
  - `TIME_BETWEEN_CAMERAS_CONSTRAINT`: Limits time gap between camera appearances
  - `TWO_CAMERAS_SAME_TIME_CONSTRAINT`: Prevents simultaneous appearances in different cameras
- Optional color histogram fusion for appearance
- Optional shape signature (aspect ratio, area, size change)

**Configuration Flags:**

```python
APPLY_TIME_RESTRICTIONS = True
CAMERAS_POINTING_AT_SAME_SPOT = True  # True for sequential, False for overlapping
MAX_TIME_BETWEEN_CAMERAS = 5.0  # seconds

USE_COLOR_FUSION = False
COLOR_WEIGHT = 0.3

USE_SHAPE_SIGNATURE = False
SHAPE_WEIGHT = 0.15
```

**Usage:**

```bash
python mtmc_time_restrictions.py
```

#### `mtmc_time_restrictions_RESNET.py`

Similar to `mtmc_time_restrictions.py` but with:

- Support for multiple sequences (S01, S03, S04)
- Extended camera lists
- Optional YOLO detection JSON export
- WandB integration for experiment tracking
- Color histogram debugging visualization

**Usage:**

```bash
python mtmc_time_restrictions_RESNET.py
```

#### `mtmc_time_restrictions_VIT.py`

Uses Vision Transformer (ViT) based Re-ID model instead of ResNet.

**Key Differences:**

- TransReID model with ViT backbone
- Camera-aware feature extraction
- Different input resolution (252x252 vs 224x224)
- Loads configuration from YAML file

**Usage:**

```bash
python mtmc_time_restrictions_VIT.py
```

---

## Project Structure

```
week4/
├── mtmc.py                              # Basic MTMC pipeline
├── run_offline_mtmc.py                  # Advanced offline MTMC with visualization
├── mtmc_time_restrictions.py            # MTMC with temporal constraints
├── mtmc_time_restrictions_RESNET.py     # Extended version with ResNet
├── mtmc_time_restrictions_VIT.py        # Vision Transformer variant
├── offline_multicamera_tracking.py      # Core offline tracking logic
├── src/
│   ├── bounding_box.py                  # Bounding box utilities
│   ├── camera_graph.py                  # Camera connectivity graph
│   ├── clustering_associator.py         # Clustering-based association
│   ├── dataset.py                       # AI City dataset loader
│   ├── detector.py                      # Detection wrappers
│   ├── eval.py                          # Evaluation metrics
│   ├── model.py                         # ft_net ReID model
│   ├── multi_camera_associator.py       # Multi-camera association logic
│   ├── multi_tracker.py                 # Single-camera tracker wrapper
│   ├── reid_feature_extractor.py        # Feature extraction
│   ├── single_camera_tracker.py         # SORT/Max-Overlap trackers
│   ├── video_source.py                  # Video reading utilities
│   └── world_and_camera_tracking.py     # World coordinate tracking
└── output/                              # Output directory for results
```

## Model Weights

Required model files:

- `yolov10s_coco.pt`: YOLO detector weights
- `src/net_19.pth`: ft_net Re-ID model weights (ResNet-based)
- `src/vit_transreid_veri.pth`: ViT-based Re-ID weights (for VIT variant)

## Dependencies

Key packages:

- PyTorch
- OpenCV (cv2)
- Ultralytics (YOLO)
- NumPy
- Pandas
- Matplotlib (for visualization)
- tqdm
- wandb (optional, for experiment tracking)

## Evaluation Metrics

The system reports:

- **IDF1**: F1 score for ID assignment
- **IDP**: ID precision
- **IDR**: ID recall
- Optional: HOTA, DetA, AssA (if enabled)

## Notes

- All scripts automatically filter detections using ROI masks
- The system assumes 10 FPS video
- Camera IDs are converted from strings (e.g., "c001") to integers (1)
- World coordinates use homography transformations from camera calibration
- Time constraints improve accuracy
