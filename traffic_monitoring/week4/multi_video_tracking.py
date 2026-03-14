import cv2
import numpy as np
from collections import defaultdict

def get_color(track_id):
    """Generate a unique, consistent color for a given track ID."""
    np.random.seed(track_id)
    # Generate RGB values avoiding very dark colors
    color = np.random.randint(50, 255, size=3)
    return tuple(map(int, color))

def parse_mtmc_results(results_path, target_camera):
    """
    Parses an AI City Challenge MTMC tracking file.
    Format: <camera_id>, <object_id>, <frame_id>, <x>, <y>, <w>, <h>, <x_world>, <y_world>
    """
    tracking_data = defaultdict(list)
    
    # AI City results often map "c001" to the integer 1 in the camera_id column
    try:
        target_cam_int = int(target_camera.replace('c', ''))
    except ValueError:
        target_cam_int = -1

    try:
        with open(results_path, 'r') as f:
            for line in f:
                # Handle both comma and space-separated lines
                parts = line.strip().replace(',', ' ').split()
                if len(parts) < 7:
                    continue
                
                cam_id_str = parts[0]
                
                # Check if the line belongs to the target camera (e.g., "c001" or "1")
                if cam_id_str == target_camera or cam_id_str == str(target_cam_int):
                    track_id = int(float(parts[1]))
                    frame_id = int(float(parts[2]))
                    bb_left = float(parts[3])
                    bb_top = float(parts[4])
                    bb_width = float(parts[5])
                    bb_height = float(parts[6])
                    
                    tracking_data[frame_id].append((track_id, bb_left, bb_top, bb_width, bb_height))
    except FileNotFoundError:
        print(f"Error: Tracking results file not found at {results_path}")
        
    return tracking_data

def generate_qualitative_video(video_path, results_path, output_path, target_camera):
    """
    Extracts specific camera data from an MTMC file, overlays it on the video, and saves it.
    """
    print(f"Loading tracking data for Camera: {target_camera}...")
    tracking_data = parse_mtmc_results(results_path, target_camera)
    
    if not tracking_data:
        print(f"Warning: No tracking data found for camera {target_camera} in {results_path}.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fps, fourcc, (width, height))

    frame_idx = 1
    
    print(f"Generating video for {target_camera}. Total frames: {total_frames}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display metadata
        cv2.putText(frame, f"Camera: {target_camera}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (30, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw bounding boxes and IDs for the current frame
        if frame_idx in tracking_data:
            for track_id, x, y, w, h in tracking_data[frame_idx]:
                color = get_color(track_id)
                start_point = (int(x), int(y))
                end_point = (int(x + w), int(y + h))
                
                # Draw Box
                cv2.rectangle(frame, start_point, end_point, color, 2)
                
                # Draw ID background and text
                label = f"ID: {track_id}"
                (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x), int(y) - h_text - 10), (int(x) + w_text, int(y)), color, -1)
                cv2.putText(frame, label, (int(x), int(y) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Qualitative video saved successfully to: {output_path}")

if __name__ == "__main__":
    # --- INTERNAL VARIABLES ---
    VIDEO_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/c001/vdo.avi"
    RESULTS_PATH = "mtmc_predictions_fixed.txt"
    OUTPUT_PATH = "output_video.avi"
    TARGET_CAMERA = "c001" 
    
    generate_qualitative_video(VIDEO_PATH, RESULTS_PATH, OUTPUT_PATH, TARGET_CAMERA)