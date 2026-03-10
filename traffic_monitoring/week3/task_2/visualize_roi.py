import cv2
import numpy as np

# --- Paths ---
SEQUENCE = "S03"
CAMERA = "c011"
VIDEO_PATH = f"../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/train/{SEQUENCE}/{CAMERA}/vdo.avi"
ROI_PATH = f"../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/train/{SEQUENCE}/{CAMERA}/roi.jpg"
OUTPUT_IMAGE_PATH = f"{CAMERA}_roi_overlay.jpg"

def main():
    print(f"Loading video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file at {VIDEO_PATH}")
        return
        
    # Read the very first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read the first frame.")
        return

    print(f"Loading ROI mask: {ROI_PATH}")
    roi_mask = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
    
    if roi_mask is None:
        print(f"Error: Could not read ROI mask at {ROI_PATH}")
        return
        
    # Ensure it's a strict binary mask (0 or 255)
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Create the Overlay ---
    
    # 1. Create a darkened version of the frame (30% brightness) for the ignored areas
    dark_frame = (frame * 0.3).astype(np.uint8)
    
    # 2. Expand the 1-channel mask to 3 channels to match the frame's shape
    mask_3channel = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
    
    # 3. Combine: Where mask is white (255), use original frame; otherwise use dark frame
    overlayed_frame = np.where(mask_3channel == 255, frame, dark_frame)
    
    # 4. Draw a green contour line exactly on the ROI boundary to make it pop
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlayed_frame, contours, -1, (0, 255, 0), 2)

    print(f"Saving overlay to: {OUTPUT_IMAGE_PATH}")
    cv2.imwrite(OUTPUT_IMAGE_PATH, overlayed_frame)
    print("Done!")

if __name__ == "__main__":
    main()