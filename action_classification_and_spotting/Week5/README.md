### Master in Computer Vision (Barcelona) 2025/26

# Project 2 (Task 1) @ C6 - Video Analysis

This repository provides the starter code for Task 1 of Project 2: Action classification on the SoccerNet Ball Action Spotting 2025 (SN-BAS-2025) dataset.

The installation of dependencies, how to obtain the dataset, and instructions on running the classification baseline are detailed next.

## Dependencies

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Getting the dataset and data preparation

Refer to the README files in the [data/soccernetball](/data/soccernetball) directory for instructions on how to download the SNABS2025 dataset, preparation of directories, and extraction of the video frames.

## Running the baseline for Task 1

The `main_classification.py` is designed to train and evaluate the baseline using the settings specified in a configuration file. You can run `main_classification.py` using the following command:

```
python3 main_classification.py --model <model_name>
```

Here, `<model_name>` can be chosen freely but must match the name of a configuration file (e.g. `baseline.json`) located in the config directory [config](/config/). For example, to chose the baseline model, you would run: `python3 main_classification.py --model baseline`.

For additional details on configuration options using the configuration file, refer to the README in the [config](/config/) directory.

## Important notes

- Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files.
- Make sure to run the `main_classification.py` with the `mode` parameter set to `store` at least once to generate the clips and save them. After this initial run, you can set the `mode` to `load` to reuse the same clips in subsequent executions.

## Project structure

```
main_classification.py   # Training + evaluation entry point
config/                  # JSON config files (one per experiment)
model/
  model_classification.py  # Model wrapper + training logic
  neck.py                  # Temporal aggregation modules
  modules.py               # Base classes and FC layers
dataset/
  frame.py                 # ActionSpotDataset + FrameReader
  datasets.py              # Dataset factory (train/val/test splits)
util/                    # I/O helpers, evaluation, class loading
data/soccernetball/      # Dataset metadata and class list
```

## Configuration reference

| Key                 | Type | Default    | Description                                            |
| ------------------- | ---- | ---------- | ------------------------------------------------------ |
| `feature_arch`      | str  | —          | Backbone (see below)                                   |
| `neck_architecture` | str  | `max_pool` | Temporal neck (see below)                              |
| `neck_parameters`   | dict | `{}`       | Neck-specific kwargs                                   |
| `clip_len`          | int  | —          | Frames per clip                                        |
| `stride`            | int  | `2`        | Frame sampling stride (higher = lower effective FPS)   |
| `loss`              | str  | `bce`      | `bce`, `weighted_bce`, or `focal`                      |
| `loss_parameters`   | dict | `{}`       | e.g. `{"gamma": 2.0, "alpha": 0.25}` for focal         |
| `freeze_backbone`   | bool | `false`    | Freeze backbone weights during training                |
| `map_eval_freq`     | int  | `2`        | Evaluate mAP on validation every N epochs              |
| `epoch_num_frames`  | int  | —          | Total frames sampled per epoch (controls dataset size) |
| `store_mode`        | str  | —          | `store` to precompute clips, `load` to reuse them      |

## Available backbones (`feature_arch`)

`rny002`, `rny004`, `rny008`, `resnet18`, `resnet50`, `efficientnet_b0`, `efficientnet_b3`, `convnext_tiny`, `clip_vitb32`, `clip_vitb16`, `clip_rn50`

CLIP backbones are always frozen. All others are fine-tuned unless `freeze_backbone: true`.

## Available neck architectures

| `neck_architecture` | Key `neck_parameters`                                                 | Notes                                                    |
| ------------------- | --------------------------------------------------------------------- | -------------------------------------------------------- |
| `max_pool`          | —                                                                     | Baseline; no temporal parameters                         |
| `gru`               | `hidden_dim`, `num_layers`, `bidirectional`, `dropout`                | Bidirectional by default; mean-pools hidden states       |
| `tcn`               | `num_layers`, `kernel_size`, `dropout`                                | Dilated residual 1-D convs                               |
| `tcn_unet`          | `num_levels`, `blocks_per_level`, `kernel_size`, `dropout`, `pooling` | Encoder-decoder with skip connections                    |
| `transformer`       | `num_layers`, `num_heads`, `dim_feedforward`, `dropout`               | Pre-norm transformer with sinusoidal positional encoding |

## Evaluation metrics

Two mAP variants are reported:

- **mAP12**: mean AP over all 12 action classes.
- **mAP10**: mean AP over 10 classes, excluding FREE KICK and GOAL (which are easier and inflate the overall score).

At the end of training, all three saved checkpoints (`best_loss`, `best_map12`, `best_map10`) are evaluated independently on the test set.

## Additional flags

| Flag         | Description                                                                        |
| ------------ | ---------------------------------------------------------------------------------- |
| `--dry-run`  | Disables all logging (wandb and local file saves). Useful for quick sanity checks. |
| `--seed INT` | Sets the random seed (default: 1).                                                 |

## Support

For any issues related to the code, please email [aclapes@ub.edu](mailto:aclapes@ub.edu) and CC [arturxe@gmail.com](mailto:arturxe@gmail.com).
