import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ---------- CONFIGURATION ----------
VIDEO_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S03/c011/vdo.avi"
ROI_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S03/c011/roi.jpg"
OUTPUT_PATH = "yolov10s_inference_roi_c001.avi"

YOLO_WEIGHTS = "./yolov10s_coco.pt"
CONF_THRESHOLD = 0.45
CAR_CLASS = 0 
# -----------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLOv10s...")
    model = YOLO(YOLO_WEIGHTS)

    print("Loading ROI Mask...")
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    if roi_mask is not None:
        # 255 is the region we keep, < 255 is the region we ignore
        _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        print("ROI mask loaded successfully.")
    else:
        print(f"Warning: No roi.jpg found at {ROI_PATH}. Processing full frames.")

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

    print(f"Starting inference. Output will be saved to: {OUTPUT_PATH}")

    for _ in tqdm(range(total_frames), desc="Running Inference"):
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run YOLO detection
        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        # 2. Process and draw bounding boxes
        for box in results.boxes:
            if int(box.cls[0]) == CAR_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Apply ROI Filtering logic
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
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Car {conf:.2f}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(frame, (x1, y1 - h_text - 5), (x1 + w_text, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 3. OVERLAY THE ROI MASK
        if roi_mask is not None:
            # Create a copy of the frame to act as the color overlay
            overlay = frame.copy()
            
            # Color the ignored regions (where mask < 255) with Red (BGR format: 0, 0, 255)
            overlay[roi_mask < 255] = (0, 0, 255)
            
            # Blend the overlay with the original frame 
            # 0.3 is the opacity of the red tint, 0.7 is the opacity of the original video
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # 4. Write frame
        out.write(frame)

    cap.release()
    out.release()
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()