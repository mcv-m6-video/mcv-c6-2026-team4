import os
import sys
import random
import yaml
import shutil
import wandb
import numpy as np
from ultralytics import YOLO
import cv2

# Add your source path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import load_annotations
from src.video_source import VideoPartSource
from data_conversor import export_to_yolo

# --- CONFIGURATION ---
VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi"
XML_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"
PROJECT_NAME = "c6_week2_task_1_3"
MODEL_NAME = "yolov10s.pt" 

# BEST HYPERPARAMETERS
BEST_HYPERPARAMS = {
    "epochs": 50,
    "imgsz": 640,         
    "batch": 32,            
    "lr0": 0.004161985142776562,        
    "lrf": 0.01086124909501176,         
    "cos_lr": True,      
    "optimizer": "NAdam",  
    "momentum": 0.937,
    "weight_decay": 0.00008066452006745294,
    "warmup_epochs": 9,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "mosaic": 0.23651390958203464,      
}

def create_data_yaml(path, train_path, val_path):
    """Generates the data.yaml file required by YOLO."""
    data = {
        'path': os.path.abspath(path),
        'train': os.path.abspath(os.path.join(path, train_path)),
        'val': os.path.abspath(os.path.join(path, val_path)),
        'names': {0: 'car'}
    }
    yaml_path = os.path.join(path, 'data.yaml')
    with open(yaml_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)
    return yaml_path

def run_training(run_name, dataset_yaml, strategy_type, fold_idx):
    """Executes one training run and logs detailed metrics to W&B."""
    
    # 1. Merge Hyperparams with Run Metadata
    # This ensures 'fold' and 'strategy' are treated as Config, not Metrics
    run_config = BEST_HYPERPARAMS.copy()
    run_config['fold'] = fold_idx
    run_config['strategy'] = strategy_type
    run_config['dataset_yaml'] = dataset_yaml

    # 2. Initialize W&B Run
    run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        group=strategy_type,  # Grouping for UI
        job_type="train",
        config=run_config, 
        reinit=True
    )
    
    model = YOLO(MODEL_NAME)
    
    # 3. Train with Best Hyperparameters
    print(f"🚀 Starting training for {run_name} (Fold {fold_idx})...")
    model.train(
        data=dataset_yaml,
        project=PROJECT_NAME,
        name=run_name,
        seed=42,        # Keep seed fixed to isolate data split effects
        deterministic=True,
        save=False,     
        plots=True,     
        freeze=11,      
        **BEST_HYPERPARAMS
    )
    
    # 4. Validation for Granular Metrics (S/M/L)
    print(f"📊 Validating {run_name}...")
    
    val_results = model.val(data=dataset_yaml, split='val')
    
    # Log ONLY the performance metrics (Fold is already in config)
    wandb.log({
        "metrics": val_results.results_dict
    })
    
    print(f"✅ Run {run_name} finished. Metrics logged to W&B.")
    run.finish()

def strategy_b_kfold():
    """
    Strategy B: 4-Fold Cross Validation.
    Video is split into 4 sequential chunks. 
    Each fold uses 1 chunk (25%) for Train, 3 chunks (75%) for Val.
    """
    print("\n🚀 Starting Strategy B: 4-Fold CV (Fixed Blocks)")
    
    folds = [(0.0, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0)]
    all_gt = load_annotations(XML_PATH)

    for i, (start, end) in enumerate(folds):
        run_name = f"StratB_Fold_{i}"
        dataset_root = f"datasets/strat_b_fold_{i}"
        
        # Prepare Data if not exists
        if not os.path.exists(dataset_root):
            print(f"Generating dataset for Fold {i}...")
            
            # Training Source (Current 25% chunk)
            train_src = VideoPartSource(VIDEO_PATH, start_frac=start, end_frac=end)
            export_to_yolo(train_src, all_gt, os.path.join(dataset_root, "train"), train_src.width, train_src.height)
            
            # Validation Source (The other 75%)
            # Part 1: Before current chunk
            if start > 0:
                val_src_1 = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=start)
                export_to_yolo(val_src_1, all_gt, os.path.join(dataset_root, "val"), val_src_1.width, val_src_1.height)
            
            # Part 2: After current chunk
            if end < 1.0:
                val_src_2 = VideoPartSource(VIDEO_PATH, start_frac=end, end_frac=1.0)
                export_to_yolo(val_src_2, all_gt, os.path.join(dataset_root, "val"), val_src_2.width, val_src_2.height)

        yaml_path = create_data_yaml(dataset_root, "train", "val")
        
        # Run Training
        run_training(run_name, yaml_path, "Strategy_B", i)

def strategy_c_random():
    """
    Strategy C: Random Splits.
    Robust version: checks if pool count matches video frame count.
    """
    print("\n🚀 Starting Strategy C: Random Splits")
    
    # 1. Setup Master Pool Paths
    pool_root = os.path.abspath("datasets/pool_all_frames")
    pool_img_dir = os.path.join(pool_root, "images")
    pool_lbl_dir = os.path.join(pool_root, "labels")
    
    all_gt = load_annotations(XML_PATH)

    # --- INTELLIGENT POOL CHECK ---
    # Get actual video length
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Count existing images
    existing_images = []
    if os.path.exists(pool_img_dir):
        existing_images = [f for f in os.listdir(pool_img_dir) if f.endswith('.jpg')]
    
    print(f"🧐 Pool Status: Found {len(existing_images)} images. Video has {total_video_frames} frames.")

    # RE-EXPORT IF: Pool is empty OR Pool has fewer images than the video (corrupt run)
    if len(existing_images) < total_video_frames:
        print(f"⚠️  Pool is incomplete ({len(existing_images)}/{total_video_frames}). Deleting and Re-exporting...")
        
        # Nuke the corrupt folder
        if os.path.exists(pool_root): 
            shutil.rmtree(pool_root)
        os.makedirs(pool_root, exist_ok=True)
        
        print(f"⏳ Exporting ALL frames to pool: {pool_root}...")
        full_source = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=1.0)
        export_to_yolo(full_source, all_gt, pool_root, full_source.width, full_source.height)
        
        # Update list after export
        existing_images = [f for f in os.listdir(pool_img_dir) if f.endswith('.jpg')]
    else:
        print(f"✅ Pool is valid and complete.")

    # 3. Sanity Check
    if len(existing_images) == 0:
        raise FileNotFoundError(f"❌ Critical: Export finished but {pool_img_dir} is still empty!")
    
    all_images = existing_images
    NUM_RUNS = 3
    
    for run_idx in range(NUM_RUNS):
        run_name = f"StratC_Run_{run_idx}"
        dataset_root = os.path.abspath(f"datasets/strat_c_run_{run_idx}")
        train_img_dir = os.path.join(dataset_root, "train", "images")
        
        # Force clean if empty split found
        if os.path.exists(train_img_dir) and len(os.listdir(train_img_dir)) == 0:
            shutil.rmtree(dataset_root)
        
        if not os.path.exists(train_img_dir):
            print(f"Creating Random Split {run_idx}...")
            
            for split in ['train', 'val']:
                os.makedirs(os.path.join(dataset_root, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(dataset_root, split, "labels"), exist_ok=True)
            
            # Shuffle
            random.seed(42 + run_idx)
            current_images = all_images.copy()
            random.shuffle(current_images)
            
            split_point = int(len(current_images) * 0.25)
            train_files = current_images[:split_point]
            val_files = current_images[split_point:]
            
            def copy_files_safe(files, split_name):
                dest_img_path = os.path.join(dataset_root, split_name, "images")
                dest_lbl_path = os.path.join(dataset_root, split_name, "labels")
                
                # Using shutil.copy2 to preserve metadata (slightly safer)
                for f in files:
                    shutil.copy2(os.path.join(pool_img_dir, f), os.path.join(dest_img_path, f))
                    label_name = f.replace('.jpg', '.txt')
                    src_label = os.path.join(pool_lbl_dir, label_name)
                    if os.path.exists(src_label):
                        shutil.copy2(src_label, os.path.join(dest_lbl_path, label_name))

            copy_files_safe(train_files, "train")
            copy_files_safe(val_files, "val")

        yaml_path = create_data_yaml(dataset_root, "train", "val")
        run_training(run_name, yaml_path, "Strategy_C", run_idx)
        
if __name__ == "__main__":
    # Safety Check: If datasets folder exists and you want a fresh start, uncomment below:
    # if os.path.exists("datasets"): shutil.rmtree("datasets")
    
    #strategy_b_kfold()
    strategy_c_random()