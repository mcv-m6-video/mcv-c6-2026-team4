import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# Import your custom classes
from src.bounding_box import BoundingBox
from src.multi_tracker import MultiTracker

# ---------- CONFIGURATION ----------
VIDEO_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c001/vdo.avi"
ROI_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c001/roi.jpg"
OUTPUT_PATH = "sort_tracking_c001.avi"

YOLO_WEIGHTS = "./yolov10s_coco.pt"
CONF_THRESHOLD = 0.6543688814661107
IOU_THRESHOLD = 0.2
MAX_AGE = 12      
CAR_CLASS = 0      
# -----------------------------------

def get_color(track_id):
    """Generates a unique, consistent color for a given track ID."""
    np.random.seed(int(track_id))
    color = np.random.randint(50, 255, size=3)
    return tuple(map(int, color))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Models & Masks
    print("Loading YOLOv10s...")
    detector = YOLO(YOLO_WEIGHTS)

    print("Loading ROI Mask...")
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is not None:
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    else:
        print(f"Warning: No roi.jpg found at {ROI_PATH}. Processing full frames.")

    print("Initializing SORT MultiTracker...")
    tracker = MultiTracker(method='sort', max_age=MAX_AGE, iou_threshold=IOU_THRESHOLD)

    # 2. Setup Video I/O
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video at {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    # 3. Processing Loop
    current_frame_id = 1
    
    for _ in tqdm(range(total_frames), desc="Tracking & Rendering"):
        ret, frame = cap.read()
        if not ret:
            break

        # A. Run Detection
        results = detector(frame, verbose=False, conf=CONF_THRESHOLD)[0]
        detections = []
        
        for box in results.boxes:
            if int(box.cls[0]) == CAR_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                
                # Apply ROI Filtering
                if roi_mask is not None:
                    is_outlier = False
                    if 0 <= x1 < width:
                        if 0 <= y1 < height and roi_mask[y1, x1] < 255: is_outlier = True
                        if 0 <= y2 < height and roi_mask[y2, x1] < 255: is_outlier = True
                    if 0 <= x2 < width:
                        if 0 <= y1 < height and roi_mask[y1, x2] < 255: is_outlier = True
                        if 0 <= y2 < height and roi_mask[y2, x2] < 255: is_outlier = True
                    
                    if is_outlier:
                        continue 
                        
                conf = float(box.conf[0].cpu().numpy())
                
                # Remember to include confidence!
                detections.append(BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf))

        # B. Update Tracker
        # This runs the Kalman Filter prediction and Hungarian matching
        tracker.update(detections, frame, current_frame_id)

        # C. Visualization
        # Draw ROI overlay for context (optional, makes for a great presentation visual)
        if roi_mask is not None:
            overlay = frame.copy()
            overlay[roi_mask < 255] = (0, 0, 255) # Red tint for ignored areas
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

        # Draw frame counter
        cv2.putText(frame, f"Frame: {current_frame_id}/{total_frames}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw active tracks
        for track_id, info in tracker.active_tracks.items():
            bbox = info["current_bbox"]
            
            x1, y1 = int(bbox.left), int(bbox.top)
            x2, y2 = int(bbox.right), int(bbox.bottom)
            
            color = get_color(track_id)
            
            # Draw Bounding Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw Label with SORT ID
            label = f"ID: {track_id}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - h_text - 10), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # D. Write output
        out.write(frame)
        current_frame_id += 1

    # Finalize
    tracker.finalize() # Good practice to close out remaining tracks
    cap.release()
    out.release()
    print(f"\nTracking complete! Video saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()