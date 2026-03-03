import os
import cv2
import sys
from tqdm import tqdm
import wandb
from data_conversor import export_to_yolo
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import evaluate_detections, load_annotations, show_metrics
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource

SAVE_DATASET=False  

def train():
    run = wandb.init()
    config = run.config
    MODEL_NAME = "yolov10s.pt"
    
    if SAVE_DATASET:
        # --- CONFIGURATION ---
        VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi" 
        XML_PATH = "../data/ai_challenge_s03_c010-full_annotation.xml"
        DATASET_ROOT = "yolo_dataset_strategy_a"
        
        # Load all annotations
        all_gt = load_annotations(XML_PATH)

        # Strategy A: 25% Train / 75% Test
        train_source = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=0.25)
        test_source = VideoPartSource(VIDEO_PATH, start_frac=0.25, end_frac=1.0)
        
        export_to_yolo(train_source, all_gt, os.path.join(DATASET_ROOT, "train"), 
                    train_source.width, train_source.height)
        export_to_yolo(test_source, all_gt, os.path.join(DATASET_ROOT, "val"), 
                    test_source.width, test_source.height)

    # --- Training Task 1.2 ---
    # Load a pre-trained model
    model = YOLO(MODEL_NAME)

    # Fine-tuning logic: Freeze early layers and train on our dataset
    print("Fine-tuning: Freezing backbone layers...")
    results = model.train(
        data="data.yaml",
        epochs=config.epochs,
        batch=config.batch_size,
        optimizer=config.optimizer,
        lr0=config.learning_rate,
        lrf=config.lrf,
        cos_lr=config.cos_lr,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs if hasattr(config, 'warmup_epochs') else 0,
        imgsz=config.img_size if hasattr(config, 'img_size') else 640,
        patience=5,
        freeze=11, # Freezes the first 11 layers (Backbone)
        project="c6_week2",
        name=run.name,
        seed=42,
        deterministic=True,
        
        augment=config.augment if hasattr(config, 'augment') else False,
        fliplr=0.5, # 50% chance of horizontal flip
        flipud=0.0,
        hsv_v=0.4, # Random brightness
        mosaic=config.mosaic if hasattr(config, 'mosaic') else 0.0, # Enable mosaic augmentation
        
        save=True,
        plots=True,
    )
    
    save_dir = results.save_dir
    
    results_path = os.path.join(save_dir, "results.png")
    if os.path.exists(results_path):
        wandb.log({"Training Results": wandb.Image(results_path)})
    
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    best_model = YOLO(best_model_path)
    
    # Run validation on the test set (Strategy A: last 75%)
    val_results = best_model.val(data="data.yaml", split='val')
    
    target_metrics = {
        "final_mAP50": val_results.results_dict.get('metrics/mAP50(B)', -1),
        "final_mAP50_95": val_results.results_dict.get('metrics/mAP50-95(B)', -1),
        "results_dict": val_results.results_dict
    }
    
    wandb.log(target_metrics)     
    run.finish()

if __name__ == "__main__":
    train()