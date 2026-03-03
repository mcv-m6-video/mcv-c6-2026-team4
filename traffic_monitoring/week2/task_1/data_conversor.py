import os
import cv2
import sys
from tqdm import tqdm

def export_to_yolo(video_source, annotations, output_dir, img_width, img_height):
    """Saves frames as images and annotations as YOLO txt files."""
    img_dir = os.path.join(output_dir, "images")
    lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    current_id = video_source.start_frame
    for frame_img in tqdm(video_source, desc=f"Exporting {output_dir}"):
        # 1. Save Image
        img_name = f"frame_{current_id:04d}.jpg"
        cv2.imwrite(os.path.join(img_dir, img_name), frame_img)

        # 2. Save Label (if detections exist for this frame)
        if current_id in annotations:
            with open(os.path.join(lbl_dir, f"frame_{current_id:04d}.txt"), "w") as f:
                for bbox in annotations[current_id]:
                    # YOLO format: class_id x_center y_center width height (normalized 0-1)
                    dw = 1.0 / img_width
                    dh = 1.0 / img_height
                    w = bbox.right - bbox.left
                    h = bbox.bottom - bbox.top
                    x_center = bbox.left + (w / 2.0)
                    y_center = bbox.top + (h / 2.0)
                    
                    # Using class 0 for 'car' in our custom dataset
                    f.write(f"0 {x_center*dw:.6f} {y_center*dh:.6f} {w*dw:.6f} {h*dh:.6f}\n")
        
        current_id += 1