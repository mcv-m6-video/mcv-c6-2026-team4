import os

os.environ["CUDA_VISIBLE_DEVICES"] = '5'

import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torchvision.transforms as T
from ultralytics import YOLO
import wandb
from types import SimpleNamespace


from src.eval import readData, eval as aicity_eval, print_results, get_results
from src.model import ft_net
from src.TransReID.model.make_model import make_model
from src.TransReID.config import cfg
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource
from src.multi_tracker import MultiTracker

# ---------- CONFIGURATION ----------
# Base paths
AI_CITY_BASE_PATH = "/data/113-2/users/gasbert/master/C6/AI_CITY_CHALLENGE/AI_CITY_CHALLENGE_2022_TRAIN"
AI_CITY_SEQ_PATH = f"{AI_CITY_BASE_PATH}/train/S01" 
GLOBAL_GT_PATH = f"{AI_CITY_BASE_PATH}/eval/ground_truth_train.txt" # Adjust if your GT is located elsewhere
OUTPUT_PRED_FILE = "mtmc_predictions.txt"
CAM_TIMESTAMP_FILE = os.path.join(AI_CITY_BASE_PATH, "cam_timestamp/S01.txt")

YOLO_WEIGHTS = "./yolov10s_coco.pt"
REID_WEIGHTS = "./src/net_19.pth"
VIT_WEIGHTS = "./src/vit_transreid_veri.pth"
CAMERAS = ["c001", "c002", "c003", "c004", "c005"] 

# Hyperparameters
CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.49
MAX_AGE = 12              
NUM_CROPS_PER_TRACK = 5  
GLOBAL_MATCH_THRESH = 0.68 
CAR_CLASS = 0
# -----------------------------------

# Time Constraints Configuration
APPLY_TIME_RESTRICTIONS = True       # Modify these three global variables, dont touch the other ones below in this section, they are automatically assigned.
CAMERAS_POINTING_AT_SAME_SPOT = True
MAX_TIME_BETWEEN_CAMERAS = 5.0  # seconds

if APPLY_TIME_RESTRICTIONS:
    if CAMERAS_POINTING_AT_SAME_SPOT:
        TIME_BETWEEN_CAMERAS_CONSTRAINT = True
        TWO_CAMERAS_SAME_TIME_CONSTRAINT = False
    else:
        TIME_BETWEEN_CAMERAS_CONSTRAINT = False
        TWO_CAMERAS_SAME_TIME_CONSTRAINT = True
else:
    TIME_BETWEEN_CAMERAS_CONSTRAINT = False
    TWO_CAMERAS_SAME_TIME_CONSTRAINT = False
# -----------------------------------

# Feature fusion
USE_COLOR_FUSION = False
COLOR_WEIGHT = 0.3
ROOF_REGION = False
CENTRAL_REGION = True

USE_SHAPE_SIGNATURE = False
SHAPE_WEIGHT = 0.15

if USE_COLOR_FUSION:
    if USE_SHAPE_SIGNATURE:
        REID_WEIGHT = 1.0 - COLOR_WEIGHT - SHAPE_WEIGHT
    else:
        REID_WEIGHT = 1.0 - COLOR_WEIGHT
elif USE_SHAPE_SIGNATURE:
    REID_WEIGHT = 1.0 - SHAPE_WEIGHT
else:
    REID_WEIGHT = 1.0
# -----------------------------------

# Model choice
MODEL = "VIT" 
# Possible choices: "RESNET", "VIT", "DEIT"
# -----------------------------------


WANDB=False

cfg.merge_from_file("/home-local/gasbert/master/C6/mcv-c6-2026-team4/traffic_monitoring/week4/src/TransReID/configs/VeRi/vit_transreid.yml")
cfg.freeze()


def extract_reid_features(reid_model, transform, track_history, device, cam_int):
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
            cam_label = torch.tensor([cam_int]).to(device)
            feat = reid_model(tensor, cam_label=cam_label)
            feat = torch.nn.functional.normalize(feat, p=2, dim=1)
            features.append(feat.cpu().numpy())

    if not features: return None
    avg_feature = np.mean(features, axis=0)
    return (avg_feature / np.linalg.norm(avg_feature))[0]


def extract_color_histogram(track_history):
    """Compute HSV histogram using center crop to reduce background noise."""
    if len(track_history) == 0:
        return None

    hists = []

    for entry in track_history:
        crop = entry["img"]
        if crop is None or crop.size == 0:
            continue

        h, w = crop.shape[:2]

        if CENTRAL_REGION:
            # Keep only central region (remove borders)
            x1 = int(w * 0.2)
            x2 = int(w * 0.8)
            y1 = int(h * 0.2)
            y2 = int(h * 0.8)

            center_crop = crop[y1:y2, x1:x2]

            hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        elif ROOF_REGION:
            # --- ROOF REGION ---
            # Use top part of the bounding box
            y1 = int(h * 0.05)   # small margin to avoid bbox border
            y2 = int(h * 0.35)   # roof area
            x1 = int(w * 0.1)
            x2 = int(w * 0.9)

            roof_crop = crop[y1:y2, x1:x2]

            if roof_crop.size == 0:
                continue

            hsv = cv2.cvtColor(roof_crop, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0], None, [32], [0,180])

        hist = cv2.normalize(hist, hist).flatten()
        hists.append(hist)

    if not hists:
        return None

    avg_hist = np.mean(hists, axis=0)
    return avg_hist / np.linalg.norm(avg_hist)

def histogram_similarity(h1, h2):
    if h1 is None or h2 is None:
        return 0.0
    return np.dot(h1, h2)

def extract_shape_signature(track_history):
    """
    Compute a simple vehicle shape signature for a track.
    Returns a normalized 3D vector:
    [avg_aspect_ratio, avg_area, size_change_rate]
    """
    if len(track_history) == 0:
        return None

    aspect_ratios = []
    areas = []

    for entry in track_history:
        bbox = entry["bbox"]

        w = bbox.right - bbox.left
        h = bbox.bottom - bbox.top

        if h <= 0:
            continue

        aspect_ratios.append(w / h)
        areas.append(w * h)

    if len(aspect_ratios) == 0:
        return None

    avg_aspect = np.mean(aspect_ratios)
    avg_area = np.mean(areas)

    # size change rate
    size_change = abs(areas[-1] - areas[0]) / (areas[0] + 1e-6)

    shape_vec = np.array([avg_aspect, avg_area, size_change], dtype=np.float32)

    return shape_vec / np.linalg.norm(shape_vec)

def main():
    if WANDB:
        run = wandb.init()
        config = run.config
    else:
        config = None

    camera_offsets = {}
    with open(CAM_TIMESTAMP_FILE) as f:
        for line in f:
            cam, ts = line.strip().split()
            camera_offsets[int(cam.replace("c", ""))] = float(ts)
        
    confidence = CONF_THRESHOLD if not WANDB else config.conf_threshold
    max_age = MAX_AGE if not WANDB else config.max_age
    iou_threshold = IOU_THRESHOLD if not WANDB else config.iou_threshold
    global_match_threshold = GLOBAL_MATCH_THRESH if not WANDB else config.global_match_threshold
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading YOLO Detector...")
    detector = YOLO(YOLO_WEIGHTS)

    print("Loading Re-ID Feature Extractor...")
    if MODEL == "RESNET":
        reid_model = ft_net(class_num=34071, ibn=True, linear_num=2048, circle=True)
        reid_model.load_state_dict(torch.load(REID_WEIGHTS, map_location=device), strict=False)
    elif MODEL == "VIT":
        reid_model = make_model(cfg, num_class=576, camera_num=0, view_num=0)
        #reid_model.load_param(VIT_WEIGHTS)
        reid_model.to(device)
        reid_model.eval()
        for name, param in reid_model.state_dict().items():
            if torch.isnan(param).any():
                print("NaNs in:", name)
    else:
        print("Unavailabe model specified!!")
        quit()

    reid_model = reid_model.to(device)

    reid_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((252, 252)),
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
            signature = extract_reid_features(reid_model, reid_transform, track["history"], device, cam_int=cam_int)
            if signature is None: continue

            color_signature = None
            if USE_COLOR_FUSION:
                color_signature = extract_color_histogram(track["history"])

            shape_signature = None
            if USE_SHAPE_SIGNATURE:
                shape_signature = extract_shape_signature(track["history"])

            fps = 10
            track_start_frame = track["history"][0]["frame"]
            track_end_frame = track["history"][-1]["frame"]
            offset = camera_offsets[cam_int]
            track_start = offset + track_start_frame / fps
            track_end = offset + track_end_frame / fps

            matched_global_id = None
            valid_gallery = []


            # Filter gallery entries using temporal constraint
            for g in global_gallery:

                if TWO_CAMERAS_SAME_TIME_CONSTRAINT:
                    # If from different cameras, check time overlap
                    if g["camera"] != cam_int:
                        overlap = not (track_end < g["start_time"] or track_start > g["end_time"])
                        if overlap:
                            continue  # impossible match
                elif TIME_BETWEEN_CAMERAS_CONSTRAINT:
                    # Time since the car disappeared from the previous camera
                    time_gap = track_start - g["end_time"]

                    # If the car appears too late in another camera, reject
                    if time_gap > MAX_TIME_BETWEEN_CAMERAS:
                        continue

                valid_gallery.append(g)
                

            if not valid_gallery:
                matched_global_id = next_global_id
                global_gallery.append({
                    "global_id": next_global_id,
                    "feature": signature,
                    "color": color_signature,
                    "shape": shape_signature,
                    "camera": cam_int,
                    "start_time": track_start,
                    "end_time": track_end
                })
                next_global_id += 1
            else:
                similarities = []

                for gallery_entry in valid_gallery:
                    reid_sim = np.dot(gallery_entry["feature"], signature)

                    sim = REID_WEIGHT * reid_sim

                    if USE_COLOR_FUSION:
                        color_sim = histogram_similarity(gallery_entry["color"], color_signature)
                        sim += COLOR_WEIGHT * color_sim

                    if USE_SHAPE_SIGNATURE and gallery_entry.get("shape") is not None and shape_signature is not None:
                        shape_sim = np.dot(gallery_entry["shape"], shape_signature)
                        sim += SHAPE_WEIGHT * shape_sim
                    similarities.append(sim)

                similarities = np.array(similarities)

                best_match_idx = np.argmax(similarities)
                best_sim = similarities[best_match_idx]

                if best_sim > global_match_threshold:
                    best_match = valid_gallery[best_match_idx]
                    matched_global_id = best_match["global_id"]

                    updated_feat = 0.8 * best_match["feature"] + 0.2 * signature
                    best_match["feature"] = updated_feat / np.linalg.norm(updated_feat)

                    if USE_COLOR_FUSION and color_signature is not None:
                        updated_color = 0.8 * best_match["color"] + 0.2 * color_signature
                        best_match["color"] = updated_color / np.linalg.norm(updated_color)
                    
                    if USE_SHAPE_SIGNATURE and shape_signature is not None:
                        updated_shape = 0.8 * best_match["shape"] + 0.2 * shape_signature
                        best_match["shape"] = updated_shape / np.linalg.norm(updated_shape)

                    # update time span
                    best_match["start_time"] = min(best_match["start_time"], track_start)
                    best_match["end_time"] = max(best_match["end_time"], track_end)

                else:
                    matched_global_id = next_global_id
                    global_gallery.append({
                        "global_id": next_global_id,
                        "feature": signature,
                        "camera": cam_int,
                        "color": color_signature,
                        "shape": shape_signature,
                        "start_time": track_start,
                        "end_time": track_end
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
        print(results)
        
        if WANDB:
            wandb.log(results)
            run.finish()
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

if __name__ == "__main__":
    main()