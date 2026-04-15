### Master in Computer Vision (Barcelona) 2025/26

# Best model checkpoint

[checkpoint](https://drive.google.com/file/d/1ytooDwhaMjuHg9RD1Q8jI9q1leVqqkhN/view?usp=sharing)

Best model: **RegNetY-008 + BiGRU (hidden=440, 2 layers)** — mAP10 = **39.78**, mAP12 = **41.48** (35 epochs).

# Project 2 (Task 2) @ C6 - Video Analysis

Action spotting on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

The installation of dependencies, how to obtain the dataset, and instructions on running the spotting baseline are detailed next.

## Dependencies

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Getting the dataset and data preparation

Refer to the README files in the [data/soccernetball](/data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.

## Running the baseline for Task 2

The `main_spotting.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_spotting.py` using the following command:

```
python3 main_spotting.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](/config/). For example, to run the best model: `python3 main_spotting.py --model phase2_rny008_gru_deep_long`.

For additional details on configuration options using the configuration file, refer to the README in the [config](/config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files.
- Make sure to run `main_spotting.py` with `store_mode` set to `store` at least once to precompute and cache clip data. After this initial run, switch to `load` to reuse them in subsequent runs.

## Project structure

```
main_spotting.py         # Training + evaluation entry point
run_experiments.py       # Multi-GPU training scheduler
profiler.py              # GMACs / parameter count analysis
qualitative.py           # Per-video prediction visualizations
config/                  # JSON config files (one per experiment)
model/
  model_spotting.py      # Model wrapper + training/eval logic
  neck.py                # Temporal neck modules (GRU, TCN, Transformer, Identity)
  modules.py             # Base classes and FC layers
dataset/
  frame.py               # ActionSpotDataset + ActionSpotVideoDataset + FrameReader
  datasets.py            # Dataset factory (train/val/test splits)
util/
  eval_spotting.py       # NMS + SoccerNet mAP evaluation pipeline
  io.py                  # I/O helpers
  dataset.py             # Class name loading
data/soccernetball/      # Dataset metadata and class list
```

## Configuration reference

| Key                 | Type  | Description                                                    |
| ------------------- | ----- | -------------------------------------------------------------- |
| `feature_arch`      | str   | Backbone architecture (see below)                              |
| `neck_architecture` | str   | Temporal neck type: `identity`, `gru`, `tcn`, `transformer`    |
| `neck_parameters`   | dict  | Neck-specific kwargs                                           |
| `clip_len`          | int   | Frames per clip                                                |
| `stride`            | int   | Frame sampling stride (higher = lower effective FPS)           |
| `focal_gamma`       | float | Focal loss gamma; `0.0` disables focal loss (uses weighted CE) |
| `focal_alpha`       | float | Per-class weight for action classes in focal/weighted CE loss  |
| `map_eval_freq`     | int   | Evaluate mAP on validation every N epochs                      |
| `epoch_num_frames`  | int   | Total frames sampled per epoch (controls dataset size)         |
| `store_mode`        | str   | `store` to precompute clips, `load` to reuse them              |
| `warm_up_epochs`    | int   | Epochs for linear LR warm-up before cosine annealing           |

## Available backbones (`feature_arch`)

`rny002`, `rny004`, `rny008`, `rny016`, `rny032`, `rny064`, `resnet50`, `efficientnet_b3`, `convnext_tiny`, `clip_vitb32`, `clip_vitb16`, `clip_vitl14`, `clip_rn50`

CLIP backbones are always frozen. All others are fine-tuned end-to-end unless `freeze_backbone: true`.

## Available neck architectures

| `neck_architecture` | Key `neck_parameters`                                   | Notes                                               |
| ------------------- | ------------------------------------------------------- | --------------------------------------------------- |
| `identity`          | —                                                       | No temporal modeling; per-frame classification only |
| `gru`               | `hidden_dim`, `num_layers`, `bidirectional`, `dropout`  | Bidirectional by default                            |
| `tcn`               | `num_layers`, `kernel_size`, `dropout`                  | Dilated residual 1-D convolutions                   |
| `transformer`       | `num_layers`, `num_heads`, `dim_feedforward`, `dropout` | Pre-norm with sinusoidal positional encoding        |

## Evaluation metrics

Two mAP variants are reported, both at δ = 1 s temporal tolerance:

- **mAP12**: mean AP over all 12 action classes.
- **mAP10**: mean AP over 10 classes, excluding FREE KICK and GOAL. **Primary metric.**

At the end of training, all three saved checkpoints (`best_loss`, `best_map12`, `best_map10`) are evaluated independently on the test set.

## Additional flags

| Flag         | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `--dry-run`  | Disables all logging (wandb and local file saves). Useful for quick sanity checks. |
| `--seed INT` | Sets the random seed (default: 1).                                                 |

## Support

For any issues related to the code, please email [aclapes@ub.edu](mailto:aclapes@ub.edu) and CC [arturxe@gmail.com](mailto:arturxe@gmail.com).
