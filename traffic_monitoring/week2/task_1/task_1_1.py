import json
import os
import sys
import xml.etree.ElementTree as ET
from typing import Dict
from ultralytics import YOLO, RTDETR
from tqdm import tqdm
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import evaluate_detections, load_annotations, show_metrics
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource

# Import draw_bboxes helper if available in your project, otherwise we define a simple one
def draw_bboxes(img, bboxes, color=(0, 255, 0)):
    img_copy = img.copy()
    for bbox in bboxes:
        cv2.rectangle(img_copy, (int(bbox.left), int(bbox.top)), 
                      (int(bbox.right), int(bbox.bottom)), color, 2)
    return img_copy

def main():
    ############# CONFIGURATION #############
    VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi" 
    XML_PATH = "../../data/ai_challenge_s03_c010-full_annotation.xml"
    YOLO_MODEL_NAME = "best_rand.pt"
    DETR_MODEL_NAME = "rtdetr-l.pt"
    OUTPUT_DET_PATH = "yolo_detections.avi"
    OUTPUT_SIDE_BY_SIDE = "yolo_side_by_side.avi"
    CONF_THRESHOLD = 0.5
    CAR_CLASS_ID = 0  # COCO Car ID
    SAVE_VIDEOS = True
    USE_YOLO = True
    RESULTS_FILE = f"task_1_1_results_{YOLO_MODEL_NAME if USE_YOLO == True else DETR_MODEL_NAME}.json"
    #########################################

    print(f"Loading model: {YOLO_MODEL_NAME if USE_YOLO == True else DETR_MODEL_NAME}")
    model = YOLO(YOLO_MODEL_NAME) if USE_YOLO == True else RTDETR(DETR_MODEL_NAME)

    video_source = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=1.0)
    fps = video_source.fps
    width, height = video_source.width, video_source.height

    # Initialize Video Writers
    if SAVE_VIDEOS:
        detection_writer = cv2.VideoWriter(
            OUTPUT_DET_PATH,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps, (width, height)
        )

        side_by_side_writer = cv2.VideoWriter(
            OUTPUT_SIDE_BY_SIDE,
            cv2.VideoWriter_fourcc(*"XVID"),
            fps, (width * 2, height)
        )

    gt_per_frame = load_annotations(XML_PATH)
    pred_per_frame = {}
    current_frame_id = video_source.start_frame

    print("Processing Video and Generating Outputs...")
    for frame in tqdm(video_source, total=len(video_source)):
        # Inference
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        frame_detections = []
        
        # Create a black "mask" frame for visualization
        mask_frame = np.zeros((height, width), dtype=np.uint8)

        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == CAR_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    
                    bbox = BoundingBox(left=x1, top=y1, right=x2, bottom=y2, confidence=conf)
                    frame_detections.append(bbox)
                    
                    # Fill detected area on mask for visualization
                    cv2.rectangle(mask_frame, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)

        pred_per_frame[current_frame_id] = frame_detections

        # Visualizations
        # Draw Predicted (Green) and GT (Blue) if available
        vis_frame = draw_bboxes(frame, frame_detections, (0, 255, 0))
        if current_frame_id in gt_per_frame:
            vis_frame = draw_bboxes(vis_frame, gt_per_frame[current_frame_id], (255, 0, 0))

        # Convert grayscale mask to BGR for side-by-side
        mask_bgr = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)
        
        # Write frames
        if SAVE_VIDEOS:
            detection_writer.write(vis_frame)
            side_by_side_writer.write(np.concatenate([mask_bgr, vis_frame], axis=1))

        current_frame_id += 1

    # Cleanup
    if SAVE_VIDEOS:
        detection_writer.release()
        side_by_side_writer.release()

    # Evaluation
    metrics = evaluate_detections(gt_per_frame, pred_per_frame)
    # Prepare the data dictionary
    results_to_save = {
        "model": YOLO_MODEL_NAME if USE_YOLO == True else DETR_MODEL_NAME,
        "metrics": metrics
    }
    # Save results to JSON
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_to_save, f, indent=4)

    show_metrics(metrics)

if __name__ == "__main__":
    main()