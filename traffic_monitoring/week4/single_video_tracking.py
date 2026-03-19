import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src.bounding_box import BoundingBox
from src.multi_tracker import MultiTracker

# ---------- CONFIGURATION ----------
VIDEO_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c001/vdo.avi"
ROI_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c001/roi.jpg"
OUTPUT_PATH = "inference_output_c001.avi"
YOLO_WEIGHTS = "./yolov10s_coco.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.25
MAX_AGE = 10              
CAR_CLASS = 0 
# -----------------------------------

def get_color(track_id):
    """Generate a unique, consistent color for a given track ID."""
    np.random.seed(track_id)
    color = np.random.randint(50, 255, size=3)
    return tuple(map(int, color))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLO Detector...")
    detector = YOLO(YOLO_WEIGHTS)
    
    # Load ROI Mask
    print("Loading ROI Mask...")
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is not None:
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        print(f"Warning: No roi.jpg found at {ROI_PATH}. Processing full frames.")

    print("Initializing MultiTracker...")
    tracker = MultiTracker(method='sort', max_age=MAX_AGE, iou_threshold=IOU_THRESHOLD)

    # Setup Video Capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    current_frame_id = 1

    pbar = tqdm(total=total_frames, desc="Tracking & Rendering")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run Detection
        results = detector(frame, verbose=False, conf=CONF_THRESHOLD)[0]
        detections = []
        
        for box in results.boxes:
            if int(box.cls[0]) == CAR_CLASS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # ROI Filtering logic 
                if roi_mask is not None:
                    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                    h_mask, w_mask = roi_mask.shape
                    is_outlier = False
                    
                    if 0 <= ix1 < w_mask:
                        if 0 <= iy1 < h_mask and roi_mask[iy1, ix1] < 255: is_outlier = True
                        if 0 <= iy2 < h_mask and roi_mask[iy2, ix1] < 255: is_outlier = True
                    if 0 <= ix2 < w_mask:
                        if 0 <= iy1 < h_mask and roi_mask[iy1, ix2] < 255: is_outlier = True
                        if 0 <= iy2 < h_mask and roi_mask[iy2, ix2] < 255: is_outlier = True
                    
                    if is_outlier:
                        continue 
                        
                conf = float(box.conf[0].cpu().numpy())

                detections.append(BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf))

        # 2. Run Tracking
        tracker.update(detections, frame, current_frame_id)

        # 3. Visualization
        cv2.putText(frame, f"Frame: {current_frame_id}/{total_frames}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for track_id, info in tracker.active_tracks.items():
            bbox = info["current_bbox"]
            
            # Extract coordinates from BoundingBox object
            x1, y1 = int(bbox.left), int(bbox.top)
            x2, y2 = int(bbox.right), int(bbox.bottom)
            
            color = get_color(track_id)
            
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label
            label = f"ID: {track_id}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. Write output and update progress bar
        out.write(frame)
        current_frame_id += 1
        pbar.update(1)

    # Cleanup
    pbar.close()
    cap.release()
    out.release()
    print(f"\nInference complete! Qualitative video saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()