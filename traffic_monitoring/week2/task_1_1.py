import os

import xml.etree.ElementTree as ET
from typing import Dict
from ultralytics import YOLO
from tqdm import tqdm
from src.evaluation import evaluate_detections, load_annotations
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource

def main():
    ############ CONFIGURATION ############
    VIDEO_PATH = "/home/pau/Documentos/Universidad/Master/C6/mcv-c6-2026-team4/traffic_monitoring/data/AICity_data/train/S03/c010/vdo.avi" 
    XML_PATH = "/home/pau/Documentos/Universidad/Master/C6/mcv-c6-2026-team4/traffic_monitoring/data/ai_challenge_s03_c010-full_annotation.xml"
    MODEL_NAME = "yolov8n.pt"            
    CONF_THRESHOLD = 0.5
    CAR_CLASS_ID = 2 # YOLO COCO index for 'car' is 2
    
    if os.path.exists(VIDEO_PATH):
        print(f"✅ Found video file at: {VIDEO_PATH}")
    else:
        print(f"❌ Video file NOT found at: {VIDEO_PATH}")
        print("Try using the absolute path starting with /home/pau/...")
        return # Stop here to fix the path

    print(f"Loading model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print(f"Loading video: {VIDEO_PATH}...")
    video_source = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=1.0)

    print("Running inference...")
    pred_per_frame = {}  # Dictionary to hold predictions per frame, keyed by absolute frame ID
    
    # We need to track the absolute frame ID to match the annotations
    current_frame_id = video_source.start_frame

    # Iterate over the video source
    for frame in tqdm(video_source, total=len(video_source)):
        
        # Run YOLOv8 inference
        # verbose=False keeps the console clean
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        
        frame_detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                
                # Filter only for 'car' class
                if cls_id == CAR_CLASS_ID:
                    # YOLOv8 returns xyxy format
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    # Create a BoundingBox object as expected by evaluation.py
                    detection = BoundingBox(
                        left=x1,
                        top=y1,
                        right=x2,
                        bottom=y2,
                        confidence=conf
                    )
                    frame_detections.append(detection)
        
        # Store detections for this frame if any exist
        if frame_detections:
            pred_per_frame[current_frame_id] = frame_detections
            
        current_frame_id += 1

    print("Loading Ground Truth annotations...")
    # Using the load_annotations function from your evaluation.py
    # Note: Ensure the XML path corresponds to the AICityChallenge format
    gt_per_frame = load_annotations(XML_PATH)

    print("Evaluating...")
    # Filter GT to only include frames we actually processed (if using start/end_frac)
    processed_frames = set(pred_per_frame.keys())
    
    # Optional: Filter GT to match only the processed video part for fair comparison
    # gt_per_frame_filtered = {k: v for k, v in gt_per_frame.items() if k in processed_frames}
    metrics = evaluate_detections(gt_per_frame, pred_per_frame)

    print("\n--- Evaluation Results (Task 1.1) ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()