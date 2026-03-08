import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

# Add src to path for VideoPartSource
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.video_source import VideoPartSource

# Add pyflow to path
pyflow_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pyflow'))
sys.path.append(pyflow_path)
import pyflow

def flow_to_bgr(u, v):
    """Converts optical flow (u, v) into an RGB image using HSV mapping."""
    h, w = u.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255 # Max saturation
    
    # Calculate magnitude and angle
    mag, ang = cv2.cartToPolar(u, v)
    
    # Hue represents direction, Value represents magnitude
    hsv[..., 0] = ang * 180 / np.pi / 2 
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def main():
    ############# CONFIGURATION #############
    VIDEO_PATH = "../../data/AICity_data/train/S03/c010/vdo.avi" 
    OUTPUT_FLOW_VIDEO = "pyflow_visualization.avi"
    
    # PyFlow Fast Config
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 1
    nInnerFPIterations = 1
    nSORIterations = 1
    colType = 1  # 1 for Grayscale
    
    # Process only 5% of the video to test it quickly. Change to 1.0 for the full video.
    END_FRAC = 1
    SCALE = 0.4  # Downscale to speed up PyFlow
    #########################################

    print(f"Loading Video: {VIDEO_PATH}")
    video_source = VideoPartSource(VIDEO_PATH, start_frac=0.0, end_frac=END_FRAC)
    fps = video_source.fps
    orig_width, orig_height = video_source.width, video_source.height

    # Initialize Video Writer (Side-by-side: Original + Flow)
    writer = cv2.VideoWriter(
        OUTPUT_FLOW_VIDEO,
        cv2.VideoWriter_fourcc(*"XVID"),
        fps, 
        (orig_width * 2, orig_height) # Double width for side-by-side
    )

    print("Computing Optical Flow and Generating Video...")
    prev_gray = None

    for frame in tqdm(video_source, total=len(video_source)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            prev_gray = gray
            continue # Skip flow computation for the very first frame

        # 1. Resize images down for speed
        prev_gray_small = cv2.resize(prev_gray, (0, 0), fx=SCALE, fy=SCALE)
        gray_small = cv2.resize(gray, (0, 0), fx=SCALE, fy=SCALE)
        
        # 2. Prepare for PyFlow (H, W, 1) float arrays
        im1 = prev_gray_small[:, :, np.newaxis].astype(float) / 255.
        im2 = gray_small[:, :, np.newaxis].astype(float) / 255.
        
        # 3. Compute flow on the small images
        u_small, v_small, _ = pyflow.coarse2fine_flow(
            im1, im2, alpha, ratio, minWidth, 
            nOuterFPIterations, nInnerFPIterations, nSORIterations, colType)
            
        # 4. Resize flow back to original 1080p size and correct the magnitude
        u = cv2.resize(u_small, (orig_width, orig_height)) / SCALE
        v = cv2.resize(v_small, (orig_width, orig_height)) / SCALE
        
        # 5. Convert flow vectors to an BGR color map
        flow_bgr = flow_to_bgr(u, v)

        # 6. Create side-by-side frame and write
        side_by_side_frame = np.concatenate([frame, flow_bgr], axis=1)
        writer.write(side_by_side_frame)

        prev_gray = gray

    writer.release()
    print(f"Video saved successfully to {OUTPUT_FLOW_VIDEO}")

if __name__ == "__main__":
    main()