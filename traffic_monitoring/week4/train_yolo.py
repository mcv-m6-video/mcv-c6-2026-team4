import os
import sys
import json
import zipfile
import shutil
import urllib.request
from pathlib import Path
from tqdm import tqdm
import cv2
from ultralytics import YOLO

# --- Custom AI City Imports ---
# Adjust the path if your script is located somewhere else relative to 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import load_annotations
from src.video_source import VideoPartSource
from src.bounding_box import BoundingBox

# --- Configuration ---
DATA_DIR = Path("./coco_vehicle_dataset")
YOLO_DATASET_DIR = DATA_DIR / "yolo_dataset"

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",
    "val_images": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
}
# COCO IDs: 3=car, 6=bus, 8=truck. (We keep bus here for COCO if you still want it there, 
# or you can remove 6 if you want NO buses anywhere). 
TARGET_CLASSES = {3: "car", 8: "truck"} # Removed 6 (bus) based on your request
YOLO_CLASS_ID = 0 # Merging them all into a single "vehicle" class

# AI City Configuration
AICITY_VIDEO_PATH = "../data/AICity_data/train/S03/c010/vdo.avi" 
AICITY_XML_PATH = "../data/ai_challenge_s03_c010-full_annotation.xml"


# ==========================================
# 1. COCO PROCESSING FUNCTIONS
# ==========================================
def download_and_extract(url, extract_to):
    """Downloads and extracts a zip file if it doesn't already exist."""
    os.makedirs(extract_to, exist_ok=True)
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
        
    print(f"Extracting {filename}...")
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def convert_coco_bbox_to_yolo(bbox, img_width, img_height):
    """Converts COCO bbox [x_min, y_min, width, height] to YOLO format."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2) / img_width
    y_center = (y_min + h / 2) / img_height
    return x_center, y_center, w / img_width, h / img_height

def process_coco_annotations(split="train"):
    """Parses COCO JSON, filters target classes, and creates YOLO .txt labels."""
    print(f"Processing COCO {split} annotations...")
    
    json_path = DATA_DIR / f"annotations/instances_{split}2017.json"
    img_dir = DATA_DIR / f"{split}2017"
    
    out_img_dir = YOLO_DATASET_DIR / f"images/{split}"
    out_label_dir = YOLO_DATASET_DIR / f"labels/{split}"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    images_info = {img['id']: img for img in coco_data['images']}
    valid_image_ids = set()
    
    for ann in tqdm(coco_data['annotations'], desc=f"Filtering COCO {split} boxes"):
        if ann['category_id'] in TARGET_CLASSES:
            img_id = ann['image_id']
            valid_image_ids.add(img_id)
            img_info = images_info[img_id]
            
            yolo_bbox = convert_coco_bbox_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
            
            label_file = out_label_dir / f"{img_info['file_name'].replace('.jpg', '.txt')}"
            with open(label_file, 'a') as lf:
                lf.write(f"{YOLO_CLASS_ID} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")

    for img_id in tqdm(valid_image_ids, desc=f"Copying valid COCO {split} images"):
        img_filename = images_info[img_id]['file_name']
        src_img = img_dir / img_filename
        dst_img = out_img_dir / img_filename
        if not dst_img.exists():
            shutil.copy(src_img, dst_img)


# ==========================================
# 2. AI CITY PROCESSING FUNCTIONS
# ==========================================
def convert_aicity_bbox_to_yolo(bbox: BoundingBox, img_width: int, img_height: int):
    """Converts absolute bounding box coordinates to YOLO normalized format."""
    w = bbox.right - bbox.left
    h = bbox.bottom - bbox.top
    x_center = (bbox.left + w / 2) / img_width
    y_center = (bbox.top + h / 2) / img_height
    return x_center, y_center, w / img_width, h / img_height

def process_aicity_split(video_source, gt_per_frame, split_name, seq_name="S03_c010"):
    """Extracts frames and YOLO annotations for a specific video split."""
    # Append directly to the YOLO dataset directories created by the COCO step
    img_dir = YOLO_DATASET_DIR / f"images/{split_name}"
    lbl_dir = YOLO_DATASET_DIR / f"labels/{split_name}"
    
    width, height = video_source.width, video_source.height
    current_frame_id = video_source.start_frame

    print(f"Extracting AI City {split_name} split ({len(video_source)} frames)...")
    
    for frame in tqdm(video_source, total=len(video_source)):
        img_filename = f"{seq_name}_frame_{current_frame_id:05d}.jpg"
        lbl_filename = f"{seq_name}_frame_{current_frame_id:05d}.txt"

        img_path = img_dir / img_filename
        lbl_path = lbl_dir / lbl_filename

        cv2.imwrite(str(img_path), frame)

        with open(lbl_path, "w") as f:
            if current_frame_id in gt_per_frame:
                for bbox in gt_per_frame[current_frame_id]:
                    label = getattr(bbox, 'label', '').lower()
                    name = getattr(bbox, 'name', '').lower()
                    
                    # Exclude buses as requested
                    if 'bus' in label or 'bus' in name:
                        continue
                    
                    yolo_bbox = convert_aicity_bbox_to_yolo(bbox, width, height)
                    f.write(f"{YOLO_CLASS_ID} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
        
        current_frame_id += 1

def add_aicity_data():
    """Wrapper function to handle loading and splitting the AI City video."""
    print("\n--- Adding AI City Data to YOLO Dataset ---")
    gt_per_frame = load_annotations(AICITY_XML_PATH)

    # Train Split: First 25%
    train_source = VideoPartSource(AICITY_VIDEO_PATH, start_frac=0.0, end_frac=0.25)
    process_aicity_split(train_source, gt_per_frame, split_name="train")

    # Validation Split: Remaining 75%
    val_source = VideoPartSource(AICITY_VIDEO_PATH, start_frac=0.25, end_frac=1.0)
    process_aicity_split(val_source, gt_per_frame, split_name="val")


# ==========================================
# 3. YOLO TRAINING INTEGRATION
# ==========================================
def create_yaml():
    """Generates the dataset.yaml file required by Ultralytics."""
    yaml_content = f"""
path: {YOLO_DATASET_DIR.absolute()}
train: images/train
val: images/val

nc: 1
names: ['vehicle']
"""
    yaml_path = DATA_DIR / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    return yaml_path

def main():
    # Step 1: Prepare COCO Data
    print("\n[STEP 1] Preparing COCO Base Data...")
    #download_and_extract(COCO_URLS["annotations"], DATA_DIR)
    #download_and_extract(COCO_URLS["train_images"], DATA_DIR)
    #download_and_extract(COCO_URLS["val_images"], DATA_DIR)
    #process_coco_annotations(split="train")
    #process_coco_annotations(split="val")

    # Step 2: Prepare AI City Data
    print("\n[STEP 2] Injecting AI City Data...")
    add_aicity_data()

    # Step 3: Create YAML Configuration
    print("\n[STEP 3] Generating YOLO Configuration...")
    yaml_path = create_yaml()

    # Step 4: Train YOLO
    print("\n[STEP 4] Starting YOLOv10s Training...")
    model = YOLO("yolov10s.pt") 
    
    results = model.train(
        data=str(yaml_path),
        epochs=50,
        imgsz=640,
        batch=64,
        device=0,
        patience=5,
        freeze=10,  # Freezing the backbone
        project="traffic_mot",
        name="yolov10s_vehicle_single_class",
        verbose=True
    )
    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    main()