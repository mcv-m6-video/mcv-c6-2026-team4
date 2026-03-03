import os
import sys
from ultralytics import YOLO

MODEL_NAME = "yolov10s.pt"      # Using the Small model
DATA_YAML = "./datasets/strat_c_run_0/data.yaml"  # Path to your existing data.yaml
PROJECT_NAME = "c6_week2_task_1_3" # Local folder name for results
RUN_NAME = "best_config_run"

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
    "augment": True,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "mosaic": 0.23651390958203464,      
}

def train_final_model():
    print(f"🚀 Starting training for {MODEL_NAME} with BEST config...")
    
    # Load the pretrained model
    model = YOLO(MODEL_NAME)

    # Train using the unpacked dictionary of hyperparameters
    # We force 'freeze=11' because this is likely for your Fine-Tuning Task 1.2
    results = model.train(
        data=DATA_YAML,
        project=PROJECT_NAME,
        name=RUN_NAME,
        freeze=11,          # FREEZE BACKBONE (Essential for Task 1.2)
        seed=42,            # reproducible seed
        deterministic=True,
        save=True,          # Force save weights
        plots=True,         # Save loss plots locally
        exist_ok=True,      # Overwrite existing run if needed
        **BEST_HYPERPARAMS  # Unpacks the dict above
    )

    # Output the location of the weights
    print("\n✅ Training Complete!")
    print(f"💾 Best weights saved at: {os.path.join(os.getcwd(), PROJECT_NAME, RUN_NAME, 'weights', 'best.pt')}")

if __name__ == "__main__":
    train_final_model()