import os
import sys
import traceback
import cv2
import torch
import numpy as np
if not hasattr(np, "asfarray"):          # removed in NumPy 2.0, needed by motmetrics
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)
from tqdm import tqdm
import torchvision.transforms as T
from ultralytics import YOLO
import wandb


from src.eval import readData, eval as aicity_eval, print_results, get_results
from src.model import ft_net
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource
from src.multi_tracker import MultiTracker
# ---------- CONFIGURATION ----------
# Base paths
AI_CITY_BASE_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN"
AI_CITY_SEQ_PATH = f"{AI_CITY_BASE_PATH}/train/S01" 
GLOBAL_GT_PATH = f"{AI_CITY_BASE_PATH}/eval/ground_truth_train.txt" # Adjust if your GT is located elsewhere
OUTPUT_PRED_FILE = "mtmc_predictions.txt"

YOLO_WEIGHTS = "./yolov10s_coco.pt"
REID_WEIGHTS = "./src/net_19.pth"
CAMERAS = ["c001", "c002", "c003", "c004", "c005"] 

# Hyperparameters
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
MAX_AGE = 10              
NUM_CROPS_PER_TRACK = 5  
GLOBAL_MATCH_THRESH = 0.6 
CAR_CLASS = 0
# -----------------------------------

WANDB=False

def extract_reid_features(reid_model, transform, track_history, device):
    """Samples evenly spaced crops and extracts the average Re-ID feature."""
    if len(track_history) == 0: return None

    indices = np.linspace(0, len(track_history) - 1, min(len(track_history), NUM_CROPS_PER_TRACK), dtype=int)
    sampled_crops = [track_history[i]["img"] for i in indices]

    features = []
    reid_model.eval()
    with torch.no_grad():
        for crop in sampled_crops:
            if crop is None or crop.size == 0: continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensor = transform(crop_rgb).unsqueeze(0).to(device)
            _, feat = reid_model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            features.append(feat.cpu().numpy())

    if not features: return None
    avg_feature = np.mean(features, axis=0)
    return (avg_feature / np.linalg.norm(avg_feature))[0]


def main():
    if WANDB:
        run = wandb.init()
        config = run.config
    else:
        config = None
        
    confidence = CONF_THRESHOLD if not WANDB else config.conf_threshold
    max_age = MAX_AGE if not WANDB else config.max_age
    iou_threshold = IOU_THRESHOLD if not WANDB else config.iou_threshold
    global_match_threshold = GLOBAL_MATCH_THRESH if not WANDB else config.global_match_threshold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLO Detector...")
    detector = YOLO(YOLO_WEIGHTS)

    print("Loading Re-ID Feature Extractor...")
    reid_model = ft_net(class_num=34071, ibn=True, linear_num=2048, circle=True)
    reid_model.load_state_dict(torch.load(REID_WEIGHTS, map_location=device), strict=False)
    reid_model = reid_model.to(device)

    reid_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    global_gallery = [] 
    next_global_id = 1
    
    # Open file to write AI City Challenge format predictions
    out_file = open(OUTPUT_PRED_FILE, 'w')

    # --- PROCESS EACH CAMERA ---
    for cam in CAMERAS:
        cam_int = int(cam.replace('c', '')) # 'c001' -> 1
        video_path = os.path.join(AI_CITY_SEQ_PATH, cam, "vdo.avi")
        roi_path = os.path.join(AI_CITY_SEQ_PATH, cam, "roi.jpg")
        
        print(f"\n--- Processing Camera: {cam} ---")
        
        # Load ROI Mask
        roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        if roi_mask is not None:
            _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
        else:
            print(f"Warning: No roi.jpg found for {cam}. Processing full frames.")

        video = VideoPartSource(video_path, start_frac=0.0, end_frac=1.0)
        tracker = MultiTracker(method='sort', max_age=max_age, iou_threshold=iou_threshold)

        # Step A: Local Tracking & Crop Collection
        for idx, frame in enumerate(tqdm(video, desc="Tracking & Cropping")):
            current_frame_id = idx + 1
            results = detector(frame, verbose=False, conf=confidence)[0]
            detections = []
            
            for box in results.boxes:
                if int(box.cls[0]) == CAR_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # ROI Filtering
                    if roi_mask is not None:
                        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                        height, width = roi_mask.shape
                        is_outlier = False
                        
                        if 0 <= ix1 < width:
                            if 0 <= iy1 < height and roi_mask[iy1, ix1] < 255: is_outlier = True
                            if 0 <= iy2 < height and roi_mask[iy2, ix1] < 255: is_outlier = True
                        if 0 <= ix2 < width:
                            if 0 <= iy1 < height and roi_mask[iy1, ix2] < 255: is_outlier = True
                            if 0 <= iy2 < height and roi_mask[iy2, ix2] < 255: is_outlier = True
                        
                        if is_outlier:
                            continue 
                            
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append(BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf))
            
            tracker.update(detections, frame, current_frame_id)

        finished_tracks = tracker.finalize()
        print(f"Extracted {len(finished_tracks)} local tracklets.")

        # Step B: Feature Extraction & Global Matching
        for track in tqdm(finished_tracks, desc="Re-ID & Global Linking"):
            signature = extract_reid_features(reid_model, reid_transform, track["history"], device)
            if signature is None: 
                continue

            matched_global_id = None

            # 1. Filter the gallery to exclude entries from the CURRENT camera
            # This enforces the assumption that a car cannot pass the same camera twice.
            eligible_indices = [
                i for i, g in enumerate(global_gallery) 
                if g["cam_id"] != cam_int
            ]

            # 2. Perform matching ONLY if there are eligible candidates in the gallery
            if not global_gallery or not eligible_indices:
                # No gallery yet or no eligible candidates from other cameras
                matched_global_id = next_global_id
                global_gallery.append({
                    "global_id": next_global_id, 
                    "feature": signature, 
                    "cam_id": cam_int  # Store cam_id for future filtering
                })
                next_global_id += 1
            else:
                # Extract features for eligible candidates only
                eligible_features = np.array([global_gallery[i]["feature"] for i in eligible_indices])
                
                # Calculate cosine similarity (via dot product of normalized vectors)
                similarities = np.dot(eligible_features, signature)
                
                best_local_idx = np.argmax(similarities)
                best_sim = similarities[best_local_idx]
                
                # Map the local 'eligible' index back to the main gallery index
                best_global_idx = eligible_indices[best_local_idx]

                if best_sim > global_match_threshold:
                    # Match found in a DIFFERENT camera
                    matched_global_id = global_gallery[best_global_idx]["global_id"]
                    
                    # Update the gallery feature with a momentum-based average
                    updated_feat = 0.8 * global_gallery[best_global_idx]["feature"] + 0.2 * signature
                    global_gallery[best_global_idx]["feature"] = updated_feat / np.linalg.norm(updated_feat)
                    
                    # Optional: Update cam_id to the most recent camera if needed, 
                    # but keeping the original cam_id check is safer for your assumption.
                else:
                    # No high-confidence match found; treat as a new car
                    matched_global_id = next_global_id
                    global_gallery.append({
                        "global_id": next_global_id, 
                        "feature": signature, 
                        "cam_id": cam_int
                    })
                    next_global_id += 1

            # Step C: Write to output file exactly how eval.py expects it
            # Format: CameraId, Id, FrameId, X, Y, Width, Height, Xworld, Yworld
            for hist_entry in track["history"]:
                f_idx = hist_entry["frame"]
                bbox = hist_entry["bbox"]
                w = bbox.right - bbox.left
                h = bbox.bottom - bbox.top
                out_file.write(f"{cam_int},{matched_global_id},{f_idx},{int(bbox.left)},{int(bbox.top)},{int(w)},{int(h)},-1,-1\n")

    out_file.close()

    # ---------------------------------------------------------
    # Step D: Unified MTMC Evaluation using AI City's eval.py
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("RUNNING OFFICIAL AI CITY EVALUATION")
    print("="*50)
    
    try:
        # readData from your eval.py
        test_df = readData(GLOBAL_GT_PATH)
        pred_df = readData(OUTPUT_PRED_FILE)
        
        # aicity_eval from your eval.py 
        # (roidir is set to the base path so eval.py looks in "../data/AI_CITY_CHALLENGE_2022_TRAIN/train/S01/...")
        summary = aicity_eval(test_df, pred_df, mread=False, dstype="train/S01", roidir=AI_CITY_BASE_PATH)
        
        # print_results from your eval.py
        print_results(summary, mread=False)
        results=get_results(summary)
        
        if WANDB:
            wandb.log(results)
            run.finish()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()