# Computer Vision Tracking & Optical Flow Evaluation

This repository contains the codebase for evaluating optical flow methods and implementing multi-object tracking pipelines. The project utilizes models like RAFT, FlowFormer++, and PyFlow, applying them to standard benchmarks including the KITTI vision benchmark suite and the AI City Challenge dataset.



## Folder Structure

### `src/`
Core utilities, data structures, and evaluation modules shared across the different tasks.

* **`bounding_box.py`**: Defines the foundational `BoundingBox` data structure (using `NamedTuple`) to standardize detection and tracking outputs (top, bottom, left, right, confidence).
* **`trackeval_metrics.py`**: Wraps the `trackeval` library to compute robust multi-object tracking metrics. It structures the predicted and ground truth tracks to output comprehensive metrics like HOTA, MOTA, AssA, DetA, and IDF1. 
* **`utils.py`**: Contains specialized functions for optical flow evaluation. It computes Mean Squared Error in Non-occluded areas (MSEN) and Percentage of Erroneous Pixels in Non-occluded areas (PEPN) against ground truth maps.
* **`video_source.py`**: Video processing utility featuring the `VideoPartSource` class, which allows for memory-efficient loading and iteration over specific fractional segments of OpenCV video objects.



### `task_1/`
This module is dedicated to baseline optical flow estimation and introductory multi-object tracking.

* **`models/`**
    * `FlowFormerPlusPlus/`: Contains the codebase, weights, and evaluation scripts for the FlowFormer++ optical flow model.
    * `RAFT/`: Contains the Recurrent All-Pairs Field Transforms (RAFT) architecture implementation and evaluation scripts.
* **`task_1_1_pyflow.py`**: Computes coarse-to-fine optical flow using the traditional PyFlow method. Evaluates performance against ground truth data (e.g., KITTI) calculating MSEN and PEPN.
* **`task_1_2.py`**: Implements an object tracking pipeline using YOLO for detections and integrating optical flow (RAFT) to propagate bounding boxes. 
* **`sweep_launcher.py`**: A Weights & Biases (W&B) sweep configuration script designed to run Bayesian hyperparameter optimization. It maximizes metrics like `hota_idf1` by tuning tracking parameters such as `iou_threshold`, `conf_threshold`, and `max_age`.
* **`plot.ipynb`**: A Jupyter Notebook used to generate plots and visualizations for experimental results, including execution time (efficiency) and overhead comparisons among different configurations.



### `task_2/`
This module focuses on advanced tracking applications, specifically targeting the AI City Challenge dataset.

* **`task_2.py`**: The primary tracking script for the AI City Challenge. It adapts the tracking pipeline to handle this specific dataset, filtering detections by specific vehicle classes, applying Region of Interest (ROI) constraints, and computing tracking metrics against global ground truth annotations.
* **`sweep_launcher.py`**: A dedicated W&B sweep launcher for `task_2.py`.
* **`get_gt_video.py`**: A utility script that parses the global `ground_truth_train.txt` file and generates a visualization video with overlaid ground truth bounding boxes and track IDs for specific camera sequences.
* **`visualize_roi.py`**: A visualization tool that loads a binary Region of Interest (ROI) mask and overlays it onto the first frame of a given video sequence. It dims the ignored areas to easily verify the active tracking zones.