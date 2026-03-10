import os
import sys
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from ultralytics import YOLO
from tqdm import tqdm
import wandb
import argparse
import torch
import time



# Adjust paths based on the provided directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Import provided source modules
from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource
from src.trackeval_metrics import compute_trackeval_metrics, print_trackeval_metrics, _iou

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', 'models', 'RAFT', 'core'))
sys.path.append(parent_dir)

from raft import RAFT
from utils.utils import InputPadder

# Paths
AI_CITY_CHALLENGE_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/train"
YOLO_WEIGHTS = "best.pt"
GLOBAL_GT_PATH = "../data/AI_CITY_CHALLENGE_2022_TRAIN/AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_train.txt"
CAR_CLASS = 0

# ---------- Hyperparameters ----------
MAX_AGE=10
IoU_THRESHOLD=0.46402325045147447
CONF_THRESHOLD=0.4237529849452388

WANDB=True
SAVE_VIDEO = False
 
class TrackedDetection:
    """Helper class to store predictions in the format required by trackeval_metrics.py"""
    def __init__(self, frame_id: int, track_id: int, bbox: BoundingBox):
        self.frame_id = frame_id
        self.track_id = track_id
        self.bbox = bbox
        
        
def get_track_color(track_id: int):
    """Generates a consistent, distinct color for a given track ID."""
    np.random.seed(track_id)
    # Generate random RGB values, keeping them bright enough to be visible
    color = np.random.randint(50, 255, size=3)
    return tuple(int(c) for c in color)

def load_global_gt_with_tracks(path: str, target_camera_str: str):
    """
    Parses the global ground_truth_train.txt file which contains all cameras and local traffic.
    Format: camera_id track_id frame_id left top width height x_world y_world
    """
    # Convert string like 'c001' into integer 1 to match the master file's first column
    target_camera_id = int(target_camera_str.replace('c', ''))
    
    gt_with_tracks = dict()
    
    with open(path, 'r') as f:
        for line in f:
            # IMPORTANT: The global file uses spaces, not commas!
            parts = line.strip().split(' ') 
            if len(parts) < 7: 
                continue 
            
            # --- THE FILTER ---
            # Extract the camera ID from the first column
            camera_id = int(parts[0])
            
            # If this line belongs to a different camera, ignore it!
            if camera_id != target_camera_id:
                continue 
            # ------------------
                
            track_id = int(parts[1])
            frame_id = int(parts[2])
            left = float(parts[3])
            top = float(parts[4])
            width = float(parts[5])
            height = float(parts[6])
            
            # Calculate right and bottom coordinates
            right = left + width
            bottom = top + height
            
            bbox = BoundingBox(top=top, bottom=bottom, left=left, right=right, confidence=1.0)
            
            if frame_id not in gt_with_tracks:
                gt_with_tracks[frame_id] = []
            
            gt_with_tracks[frame_id].append((bbox, track_id))
            
    total_boxes = sum(len(v) for v in gt_with_tracks.values())
    print(f"Loaded {total_boxes} bounding boxes for camera {target_camera_str}.") 
    return gt_with_tracks


def get_predominant_flow(bbox: BoundingBox, u: np.ndarray, v: np.ndarray):
    """Extracts the median optical flow inside the bounding box."""
    x1, y1 = max(0, int(bbox.left)), max(0, int(bbox.top))
    x2, y2 = min(u.shape[1], int(bbox.right)), min(u.shape[0], int(bbox.bottom))

    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0

    # Median flow ignores outliers and background movement 
    u_med = np.median(u[y1:y2, x1:x2])
    v_med = np.median(v[y1:y2, x1:x2])
    return u_med, v_med

def interpolate_bboxes(bbox1: BoundingBox, bbox2: BoundingBox, steps: int):
    """Linearly interpolates between two bounding boxes for missed detections."""
    step_t = (bbox2.top - bbox1.top) / (steps + 1)
    step_b = (bbox2.bottom - bbox1.bottom) / (steps + 1)
    step_l = (bbox2.left - bbox1.left) / (steps + 1)
    step_r = (bbox2.right - bbox1.right) / (steps + 1)
    
    interpolated = []
    for i in range(1, steps + 1):
        interpolated.append(BoundingBox(
            top=bbox1.top + step_t * i,
            bottom=bbox1.bottom + step_b * i,
            left=bbox1.left + step_l * i,
            right=bbox1.right + step_r * i,
            confidence=1.0
        ))
    return interpolated

class OpticalFlowTracker:
    def __init__(self):
        self.next_id = 0
        self.active_tracks = {} # {id: {"bbox": BoundingBox, "frame": int}}
        self.lost_tracks = {}   # Tracks lost within N-5 frames
        self.all_predictions = [] # List of TrackedDetection

    def init_tracks(self, detections: list[BoundingBox], frame_idx: int):
        """Initialization: assign unique ID to each detected object."""
        for det in detections:
            self.active_tracks[self.next_id] = {"bbox": det, "frame": frame_idx}
            self.all_predictions.append(TrackedDetection(frame_idx, self.next_id, det))
            self.next_id += 1

    def update(self, detections: list[BoundingBox], u: np.ndarray, v: np.ndarray, frame_idx: int, config):
        current_iou_thresh = IoU_THRESHOLD
        current_max_age = MAX_AGE
        new_active_tracks = {}
        unmatched_dets = list(detections)
        
        # 1. Predict BBox position using Optical Flow 
        projected_tracks = {}
        for track_id, info in self.active_tracks.items():
            bbox = info["bbox"]
            med_u, med_v = get_predominant_flow(bbox, u, v)
            
            proj_bbox = BoundingBox(
                top=bbox.top + med_v, bottom=bbox.bottom + med_v,
                left=bbox.left + med_u, right=bbox.right + med_u,
                confidence=bbox.confidence
            )
            projected_tracks[track_id] = proj_bbox

        # 2. Greedy Matching with IoU > threshold
        used_detection_indices = set()
        for track_id, pred_bbox in projected_tracks.items():
            best_iou = -1.0
            best_idx = -1
            
            for i, det in enumerate(unmatched_dets):
                if i in used_detection_indices:
                    continue
                    
                iou = _iou(pred_bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Validation: 
            # 1. Did we find a match? 
            # 2. Is it better than the sweep's threshold?
            # 3. Is the index actually in the list? (Final safety check)
            if best_idx != -1 and best_iou > current_iou_thresh:
                if best_idx < len(unmatched_dets):
                    matched_det = unmatched_dets[best_idx]
                    used_detection_indices.add(best_idx)
                    
                    new_active_tracks[track_id] = {"bbox": matched_det, "frame": frame_idx}
                    self.all_predictions.append(TrackedDetection(frame_idx, track_id, matched_det))
                else:
                    self.lost_tracks[track_id] = self.active_tracks[track_id]
            else:
                self.lost_tracks[track_id] = self.active_tracks[track_id]

        # Final Cleanup: Remove the detections that were actually used
        unmatched_dets = [d for i, d in enumerate(unmatched_dets) if i not in used_detection_indices]

        # 3. Handle Lost Tracks (Look backwards up to N-5)
        still_unmatched_dets = []
        for det in unmatched_dets:
            best_iou, best_lost_id = 0.0, None
            
            for lost_id, lost_info in list(self.lost_tracks.items()):
                frames_missed = frame_idx - lost_info["frame"]
                if frames_missed <= current_max_age:
                    iou = _iou(lost_info["bbox"], det)
                    if iou > best_iou:
                        best_iou = iou
                        best_lost_id = lost_id
                else:
                    del self.lost_tracks[lost_id] # Forget if lost for > N frames

            if best_iou > current_iou_thresh:
                # Match found: Assign ID and interpolate missing frames
                frames_missed = frame_idx - self.lost_tracks[best_lost_id]["frame"]
                interp_boxes = interpolate_bboxes(self.lost_tracks[best_lost_id]["bbox"], det, frames_missed - 1)
                
                # Add interpolated boxes to history
                for step, ibox in enumerate(interp_boxes):
                    missed_frame_id = frame_idx - frames_missed + 1 + step
                    self.all_predictions.append(TrackedDetection(missed_frame_id, best_lost_id, ibox))
                
                # Add current detection
                new_active_tracks[best_lost_id] = {"bbox": det, "frame": frame_idx}
                self.all_predictions.append(TrackedDetection(frame_idx, best_lost_id, det))
                del self.lost_tracks[best_lost_id]
            else:
                # Assign new ID
                new_active_tracks[self.next_id] = {"bbox": det, "frame": frame_idx}
                self.all_predictions.append(TrackedDetection(frame_idx, self.next_id, det))
                self.next_id += 1

        self.active_tracks = new_active_tracks
        return self.all_predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='raft-kitti.pth')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    args = parser.parse_args(args=[]) # Empty list parses default values
    
    if WANDB:
        run = wandb.init()
        config = run.config
    else:
        config = None
    
    # Load roi mask
    roi_mask = cv2.imread(f"{AI_CITY_CHALLENGE_PATH}/{config.sequence}/{config.camera}/roi.jpg", cv2.IMREAD_GRAYSCALE)
    _, roi_mask = cv2.threshold(roi_mask, 127, 255, cv2.THRESH_BINARY)
    
    print("Loading RAFT Model...")
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.model))
    raft_model.cuda()
    raft_model.eval()
    
    print("Loading Ground Truth...")
    gt_with_tracks = load_global_gt_with_tracks(GLOBAL_GT_PATH, config.camera)

    print("Loading YOLO Model...")
    model = YOLO(YOLO_WEIGHTS)
    
    video = VideoPartSource(f"{AI_CITY_CHALLENGE_PATH}/{config.sequence}/{config.camera}/vdo.avi", start_frac=0, end_frac=1) 
    
    tracker = OpticalFlowTracker()
    all_predictions = []

    print("--- PASS 1: Tracking ---")
    for idx, frame in enumerate(tqdm(video)):
        current_frame_id = idx+1
        
        # YOLO Detection
        current_conf = CONF_THRESHOLD
        results = model(frame, verbose=False, conf=current_conf)[0]
        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == CAR_CLASS:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to integers for mask indexing
                ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
                height, width = roi_mask.shape
                
                is_outlier = False
                
                # Check all 4 corners against the ROI mask (mirroring eval.py logic)
                if 0 <= ix1 < width:
                    if 0 <= iy1 < height and roi_mask[iy1, ix1] < 255: is_outlier = True
                    if 0 <= iy2 < height and roi_mask[iy2, ix1] < 255: is_outlier = True
                if 0 <= ix2 < width:
                    if 0 <= iy1 < height and roi_mask[iy1, ix2] < 255: is_outlier = True
                    if 0 <= iy2 < height and roi_mask[iy2, ix2] < 255: is_outlier = True
                
                # Only keep the detection if NONE of the corners hit the black mask
                if not is_outlier:
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append(BoundingBox(top=y1, bottom=y2, left=x1, right=x2, confidence=conf))
                
        if idx == 0:
            tracker.init_tracks(detections, current_frame_id)
        else:            
            # Image scaling
            scale = 0.5
            orig_h, orig_w = frame.shape[:2]
            
            # 1. Resize frames down (CPU/OpenCV)
            prev_small = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
            curr_small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            
            # Convert BGR to RGB
            prev_rgb = cv2.cvtColor(prev_small, cv2.COLOR_BGR2RGB)
            curr_rgb = cv2.cvtColor(curr_small, cv2.COLOR_BGR2RGB)
            
            # 2. Prepare PyTorch tensors [Batch, Channel, H_small, W_small]
            image1 = torch.from_numpy(prev_rgb).permute(2, 0, 1).float().unsqueeze(0).cuda()
            image2 = torch.from_numpy(curr_rgb).permute(2, 0, 1).float().unsqueeze(0).cuda()
            
            # 3. RAFT Inference
            with torch.no_grad():
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                
                _, flow_pr = raft_model.module(image1, image2, iters=12, test_mode=True)
                
                flow_small = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
            
            # 4. Upsample flow 
            u = cv2.resize(flow_small[:, :, 0], (orig_w, orig_h)) / scale
            v = cv2.resize(flow_small[:, :, 1], (orig_w, orig_h)) / scale
                
            # 5. Update tracks 
            all_predictions = tracker.update(detections, u, v, current_frame_id, config)
            
        prev_frame = frame.copy()
        
    # Evaluate
    print("\nEvaluating Tracking Performance...")
    metrics = compute_trackeval_metrics(gt_with_tracks, all_predictions)
    print_trackeval_metrics(metrics)
    hota_idf1 = 0.5 * metrics["HOTA"] + 0.5 * metrics["IDF1"]

    if WANDB:
        metrics["hota_idf1"] = hota_idf1
        wandb.log(metrics)
        run.finish()
        
    OUTPUT_VIDEO_PATH = f"task_2_{config.camera}.avi"

    # --- PASS 2: Video Generation (Offline) ---
    if SAVE_VIDEO:
        print("\n--- PASS 2: Generating Interpolated Video ---")
        # Re-initialize the video reader so we start from frame 0 again
        video_vis = VideoPartSource(f"{AI_CITY_CHALLENGE_PATH}/{config.sequence}/{config.camera}/vdo.avi", start_frac=0, end_frac=1) 
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, video_vis.fps, (video_vis.width, video_vis.height))
        
        # Group predictions by frame ID for super fast lookup
        preds_by_frame = {}
        for pred in all_predictions:
            if pred.frame_id not in preds_by_frame:
                preds_by_frame[pred.frame_id] = []
            preds_by_frame[pred.frame_id].append(pred)

        for idx, frame in enumerate(tqdm(video_vis)):
            vis_frame = frame.copy()
            current_frame_id = idx + 1
            
            if current_frame_id in preds_by_frame:
                for pred in preds_by_frame[current_frame_id]:
                    bbox = pred.bbox
                    track_id = pred.track_id
                    color = get_track_color(track_id)
                    
                    # Draw Bounding Box
                    cv2.rectangle(vis_frame, 
                                  (int(bbox.left), int(bbox.top)), 
                                  (int(bbox.right), int(bbox.bottom)), 
                                  color, 2)
                    
                    # Draw Track ID Text
                    text = f"ID: {track_id}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, 
                                  (int(bbox.left), int(bbox.top) - text_h - 5), 
                                  (int(bbox.left) + text_w, int(bbox.top)), 
                                  color, -1)
                    cv2.putText(vis_frame, text, 
                                (int(bbox.left), int(bbox.top) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out_writer.write(vis_frame)
            
        out_writer.release()

if __name__ == "__main__":
    main() 