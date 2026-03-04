# Traffic Monitoring Project - Week 2

This repository contains the implementation for **Week 2** of the Traffic Monitoring project (C6). The focus is on fine-tuning and evaluating state-of-the-art object detection models—**YOLOv10s** and **RT-DETR-L**—for high-precision vehicle detection in surveillance scenarios.

## 📂 Project Structure

The project is organized into two main directories: `src/` for core utilities and `task_1/` for experimental scripts.

```text
├── src/                        # Core utility modules
│   ├── bounding_box.py         # BoundingBox NamedTuple definition
│   ├── evaluation.py           # COCO evaluation wrapper and annotation loading
│   └── video_source.py         # Video streaming and frame extraction logic
└── task_1/                     # Experimental tasks and training scripts
    ├── task_1_1.py             # Inference and qualitative visualization
    ├── task_1_2.py             # Fine-tuning implementation (Strategy A)
    ├── task_1_3.py             # Validation strategies (Strategy B & C)
    ├── sweep_launcher.py       # W&B Hyperparameter sweep configuration
    ├── data_conversor.py       # YOLO dataset formatting utility
    ├── yolo.py                 # Standalone training with optimal params
    └── plots.ipynb             # Performance benchmarking and visualizations
```

---

## 🛠️ Module Descriptions

### **src/ (Core Library)**
* **`bounding_box.py`**: A lightweight module defining the `BoundingBox` structure to ensure consistent coordinate handling (`top`, `left`, `bottom`, `right`) across the pipeline.
* **`evaluation.py`**: Integrates `pycocotools` to provide standard COCO metrics. It handles the conversion of model predictions into JSON format and calculates **AP50**, **AP50-95**, and size-specific metrics (Small, Medium, Large).
* **`video_source.py`**: Implements `VideoPartSource`, allowing the user to stream specific portions of a video using `start_frac` and `end_frac` to adhere to training/testing split strategies.

### **task_1/ (Experiments)**
* **`task_1_1.py`**: Performs model inference and generates visualization videos, including a side-by-side view of Ground Truth vs. Predictions and a binary detection mask.
* **`task_1_2.py`**: The primary training script for fine-tuning. It includes backbone freezing logic (`freeze=11`) and integrates with Weights & Biases for experiment tracking.
* **`task_1_3.py`**: Executes the comparative validation strategies: **Strategy B** (4-Fold Sequential Cross-Validation) and **Strategy C** (Random Splits) to measure model stability and variance.
* **`sweep_launcher.py`**: A configuration script to launch a Bayesian optimization sweep on W&B, searching for the best learning rates, optimizers (e.g., NAdam, AdamW), and augmentation intensities.
* **`data_conversor.py`**: Provides the `export_to_yolo` function, which extracts frames and generates normalized YOLO `.txt` labels from XML annotations.

---

## 🚀 Usage

### 1. Installation
Install the necessary computer vision and logging libraries:
```bash
pip install ultralytics wandb pycocotools opencv-python scikit-learn tqdm
```

### 2. Hyperparameter Optimization
To find the best configuration for the YOLOv10s model using Strategy A:
```bash
python task_1/sweep_launcher.py
```

### 3. Strategy A Validation
The most simple strategy taking 25% initial frames of the sequence as training and last 75% for validation:
```bash
python task_1/task_1_2.py
```

### 4. Strategy B & C Validation
To assess model stability across different splits using the best-found hyperparameters:
```bash
python task_1/task_1_3.py
```

### 5. Qualitative Evaluation
To generate detection videos using your trained `best.pt` weights:
```bash
python task_1/task_1_1.py
```

---

## 📊 Key Results
* **Model Choice:** **YOLOv10s** was selected over RT-DETR for fine-tuning due to superior **parameter efficiency** and tighter bounding box regression (**AP50-95**).
