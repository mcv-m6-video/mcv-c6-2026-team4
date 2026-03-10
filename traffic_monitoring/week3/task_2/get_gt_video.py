import cv2
import numpy as np
from tqdm import tqdm

# --- Paths ---
VIDEO_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/train/S03/c015/vdo.avi"
GLOBAL_GT_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt"
TARGET_CAMERA = "c015"
OUTPUT_VIDEO_PATH = "gt_vis_S03_c015.avi"

def get_track_color(track_id: int):
    """Generates a consistent, distinct color for a given track ID."""
    np.random.seed(track_id)
    color = np.random.randint(50, 255, size=3)
    return tuple(int(c) for c in color)

def load_global_gt_for_camera(path: str, target_camera_str: str):
    """
    Parses the global ground_truth_train.txt file for a specific camera.
    Format: camera_id track_id frame_id left top width height x_world y_world
    """
    target_camera_id = int(target_camera_str.replace('c', ''))
    gt_with_tracks = {}
    
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ') 
            if len(parts) < 7: 
                continue 
            
            camera_id = int(parts[0])
            if camera_id != target_camera_id:
                continue 
                
            track_id = int(parts[1])
            frame_id = int(parts[2])
            left = float(parts[3])
            top = float(parts[4])
            width = float(parts[5])
            height = float(parts[6])
            
            right = left + width
            bottom = top + height
            
            if frame_id not in gt_with_tracks:
                gt_with_tracks[frame_id] = []
            
            # Store as a simple dictionary for drawing
            gt_with_tracks[frame_id].append({
                "track_id": track_id,
                "left": int(left),
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom)
            })
            
    total_boxes = sum(len(v) for v in gt_with_tracks.values())
    print(f"Loaded {total_boxes} ground truth boxes for {target_camera_str}.")
    return gt_with_tracks

def main():
    print(f"Loading GT for {TARGET_CAMERA}...")
    gt_data = load_global_gt_for_camera(GLOBAL_GT_PATH, TARGET_CAMERA)
    
    print(f"Opening video {VIDEO_PATH}...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return

    # Get video properties for the writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    print(f"Generating output video: {OUTPUT_VIDEO_PATH}")
    
    # AI City Challenge frame_id is 1-indexed!
    frame_id = 1 
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Check if there are ground truth annotations for this frame
            if frame_id in gt_data:
                for obj in gt_data[frame_id]:
                    track_id = obj["track_id"]
                    left, top, right, bottom = obj["left"], obj["top"], obj["right"], obj["bottom"]
                    
                    color = get_track_color(track_id)
                    
                    # Draw Bounding Box
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw Track ID Text Header
                    text = f"GT ID: {track_id}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (left, top - text_h - 5), (left + text_w, top), color, -1)
                    cv2.putText(frame, text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(frame)
            frame_id += 1
            pbar.update(1)
            
    cap.release()
    out.release()
    print("Video generation complete!")

if __name__ == "__main__":
    main()