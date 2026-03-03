import os
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_parallel_coordinate,
    plot_slice,
)
import plotly
from tqdm import tqdm
from ultralytics import YOLO

from src.bounding_box import BoundingBox
from src.video_source import VideoPartSource


@dataclass
class Detection:
    frame_id: int
    bbox: BoundingBox
    track_id: Optional[int] = None


@dataclass
class Track:
    track_id: int
    detections: List[Detection] = field(default_factory=list)

    def add_detection(self, detection: Detection):
        detection.track_id = self.track_id
        self.detections.append(detection)

    @property
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None


def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    x_left = max(box1.left, box2.left)
    y_top = max(box1.top, box2.top)
    x_right = min(box1.right, box2.right)
    y_bottom = min(box1.bottom, box2.bottom)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (box1.right - box1.left) * (box1.bottom - box1.top)
    area2 = (box2.right - box2.left) * (box2.bottom - box2.top)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def compute_iou_matrix(boxes1: List[BoundingBox], boxes2: List[BoundingBox]) -> np.ndarray:
    n1, n2 = len(boxes1), len(boxes2)
    iou_matrix = np.zeros((n1, n2))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou_matrix[i, j] = compute_iou(box1, box2)

    return iou_matrix


class MaxOverlapTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.track_last_seen: Dict[int, int] = {}

    def update(self, frame_id: int, detections: List[BoundingBox]) -> List[Detection]:
        if not detections:
            return []

        active_track_ids = [
            tid for tid, last_seen in self.track_last_seen.items()
            if frame_id - last_seen <= self.max_age
        ]

        if not active_track_ids:
            result = []
            for bbox in detections:
                det = Detection(frame_id=frame_id, bbox=bbox)
                track = Track(track_id=self.next_track_id)
                track.add_detection(det)
                self.tracks[self.next_track_id] = track
                self.track_last_seen[self.next_track_id] = frame_id
                self.next_track_id += 1
                result.append(det)
            return result

        active_tracks = [self.tracks[tid] for tid in active_track_ids]
        last_boxes = [t.last_detection.bbox for t in active_tracks]

        iou_matrix = compute_iou_matrix(last_boxes, detections)

        result = []
        matched_tracks = set()
        matched_detections = set()

        iou_flat = []
        for i in range(len(active_tracks)):
            for j in range(len(detections)):
                iou_flat.append((iou_matrix[i, j], i, j))
        iou_flat.sort(reverse=True)

        for iou_val, track_idx, det_idx in iou_flat:
            if iou_val < self.iou_threshold:
                break
            if track_idx in matched_tracks or det_idx in matched_detections:
                continue

            track = active_tracks[track_idx]
            det = Detection(frame_id=frame_id, bbox=detections[det_idx])
            track.add_detection(det)
            self.track_last_seen[track.track_id] = frame_id

            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)
            result.append(det)

        for det_idx, bbox in enumerate(detections):
            if det_idx not in matched_detections:
                det = Detection(frame_id=frame_id, bbox=bbox)
                track = Track(track_id=self.next_track_id)
                track.add_detection(det)
                self.tracks[self.next_track_id] = track
                self.track_last_seen[self.next_track_id] = frame_id
                self.next_track_id += 1
                result.append(det)

        return result

    def get_all_detections(self) -> List[Detection]:
        all_dets = []
        for track in self.tracks.values():
            all_dets.extend(track.detections)
        return all_dets


def load_annotations_with_tracks(path: str) -> Tuple[Dict[int, List[BoundingBox]], Dict[int, List[Tuple[BoundingBox, int]]]]:
    tree = ET.parse(path)
    root = tree.getroot()

    boxes_per_frame = defaultdict(list)
    boxes_with_tracks = defaultdict(list)

    for track_elem in root.findall("track"):
        track_id = int(track_elem.get("id"))
        label = track_elem.get("label")

        if label != "car":
            continue

        for box_elem in track_elem.findall("box"):
            frame = int(box_elem.get("frame"))
            outside = box_elem.get("outside") == "1"

            if outside:
                continue

            xtl = float(box_elem.get("xtl"))
            ytl = float(box_elem.get("ytl"))
            xbr = float(box_elem.get("xbr"))
            ybr = float(box_elem.get("ybr"))

            bbox = BoundingBox(
                top=ytl,
                bottom=ybr,
                left=xtl,
                right=xbr,
                confidence=1.0
            )

            boxes_per_frame[frame].append(bbox)
            boxes_with_tracks[frame].append((bbox, track_id))

    return dict(boxes_per_frame), dict(boxes_with_tracks)


def save_tracking_results_mot(detections: List[Detection], output_path: str):
    sorted_dets = sorted(detections, key=lambda d: (d.frame_id, d.track_id))

    with open(output_path, 'w') as f:
        for det in sorted_dets:
            frame = det.frame_id + 1
            track_id = det.track_id
            left = det.bbox.left
            top = det.bbox.top
            width = det.bbox.right - det.bbox.left
            height = det.bbox.bottom - det.bbox.top
            conf = det.bbox.confidence if det.bbox.confidence else 1.0

            f.write(f"{frame},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},{conf:.2f},-1,-1,-1\n")


def save_gt_mot_format(boxes_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]], output_path: str):
    with open(output_path, 'w') as f:
        for frame_id in sorted(boxes_with_tracks.keys()):
            for bbox, track_id in boxes_with_tracks[frame_id]:
                frame = frame_id + 1
                left = bbox.left
                top = bbox.top
                width = bbox.right - bbox.left
                height = bbox.bottom - bbox.top

                f.write(f"{frame},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")


# ==================== TRACKING METRICS (Task 2.3) ====================

def compute_tracking_metrics(
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    pred_detections: List[Detection],
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """Compute IDF1, HOTA, MOTA and other tracking metrics."""

    pred_by_frame = defaultdict(list)
    for det in pred_detections:
        pred_by_frame[det.frame_id].append(det)

    all_frames = sorted(set(gt_with_tracks.keys()) | set(pred_by_frame.keys()))

    # For MOTA
    total_gt = 0
    total_fp = 0
    total_fn = 0
    total_id_switches = 0

    # For IDF1
    total_idtp = 0
    total_idfp = 0
    total_idfn = 0

    # Track ID mappings (pred_id -> gt_id)
    last_match = {}  # pred_track_id -> gt_track_id from previous frame

    # For HOTA components
    detection_matches = []  # list of (iou, pred_track, gt_track)

    gt_track_ids = set()
    pred_track_ids = set()

    for frame_id in all_frames:
        gt_boxes_tracks = gt_with_tracks.get(frame_id, [])
        pred_dets = pred_by_frame.get(frame_id, [])

        gt_boxes = [bt[0] for bt in gt_boxes_tracks]
        gt_ids = [bt[1] for bt in gt_boxes_tracks]
        pred_boxes = [d.bbox for d in pred_dets]
        pred_ids = [d.track_id for d in pred_dets]

        for gid in gt_ids:
            gt_track_ids.add(gid)
        for pid in pred_ids:
            pred_track_ids.add(pid)

        total_gt += len(gt_boxes)

        if len(gt_boxes) == 0:
            total_fp += len(pred_boxes)
            total_idfp += len(pred_boxes)
            continue

        if len(pred_boxes) == 0:
            total_fn += len(gt_boxes)
            total_idfn += len(gt_boxes)
            continue

        iou_matrix = compute_iou_matrix(gt_boxes, pred_boxes)

        cost_matrix = 1 - iou_matrix
        cost_matrix[iou_matrix < iou_threshold] = 1e6

        gt_indices, pred_indices = linear_sum_assignment(cost_matrix)

        matched_gt = set()
        matched_pred = set()
        current_match = {}

        for gi, pi in zip(gt_indices, pred_indices):
            if iou_matrix[gi, pi] >= iou_threshold:
                matched_gt.add(gi)
                matched_pred.add(pi)

                gt_id = gt_ids[gi]
                pred_id = pred_ids[pi]
                current_match[pred_id] = gt_id

                detection_matches.append((iou_matrix[gi, pi], pred_id, gt_id))

                total_idtp += 1

                if pred_id in last_match and last_match[pred_id] != gt_id:
                    total_id_switches += 1

        total_fn += len(gt_boxes) - len(matched_gt)
        total_fp += len(pred_boxes) - len(matched_pred)
        total_idfn += len(gt_boxes) - len(matched_gt)
        total_idfp += len(pred_boxes) - len(matched_pred)

        last_match = current_match

    # MOTA
    mota = 1 - (total_fn + total_fp + total_id_switches) / max(total_gt, 1)
    mota = max(mota, -1.0)  # MOTA can be negative

    # IDF1
    idf1 = 2 * total_idtp / max(2 * total_idtp + total_idfp + total_idfn, 1)

    # HOTA computation
    # DetA = detection accuracy
    det_a = total_idtp / max(total_idtp + total_fn + total_fp, 1)

    # AssA = association accuracy (average over matched detections)
    if detection_matches:
        # Group matches by (pred_track, gt_track) pairs
        track_pairs = defaultdict(list)
        for iou_val, pred_id, gt_id in detection_matches:
            track_pairs[(pred_id, gt_id)].append(iou_val)

        # Count TPA, FPA, FNA for each ground truth track
        gt_track_counts = defaultdict(int)
        pred_track_counts = defaultdict(int)
        pair_counts = defaultdict(int)

        for det in pred_detections:
            pred_track_counts[det.track_id] += 1

        for frame_id, boxes_tracks in gt_with_tracks.items():
            for _, gt_id in boxes_tracks:
                gt_track_counts[gt_id] += 1

        for (pred_id, gt_id), ious in track_pairs.items():
            pair_counts[(pred_id, gt_id)] = len(ious)

        ass_scores = []
        for (pred_id, gt_id), tpa in pair_counts.items():
            fpa = pred_track_counts[pred_id] - tpa
            fna = gt_track_counts[gt_id] - tpa
            ass_score = tpa / max(tpa + fpa + fna, 1)
            for _ in range(tpa):
                ass_scores.append(ass_score)

        ass_a = np.mean(ass_scores) if ass_scores else 0.0
    else:
        ass_a = 0.0

    # HOTA = sqrt(DetA * AssA)
    hota = np.sqrt(det_a * ass_a)

    # Additional metrics
    precision = total_idtp / max(total_idtp + total_fp, 1)
    recall = total_idtp / max(total_idtp + total_fn, 1)

    return {
        "HOTA": hota,
        "IDF1": idf1,
        "MOTA": mota,
        "DetA": det_a,
        "AssA": ass_a,
        "Precision": precision,
        "Recall": recall,
        "ID_Switches": total_id_switches,
        "FP": total_fp,
        "FN": total_fn,
        "GT_Tracks": len(gt_track_ids),
        "Pred_Tracks": len(pred_track_ids),
    }


def print_metrics(metrics: Dict[str, float]):
    print("\n" + "=" * 50)
    print("TRACKING METRICS")
    print("=" * 50)
    print(f"  HOTA:        {metrics['HOTA']:.4f}")
    print(f"  IDF1:        {metrics['IDF1']:.4f}")
    print(f"  MOTA:        {metrics['MOTA']:.4f}")
    print("-" * 50)
    print(f"  DetA:        {metrics['DetA']:.4f}")
    print(f"  AssA:        {metrics['AssA']:.4f}")
    print(f"  Precision:   {metrics['Precision']:.4f}")
    print(f"  Recall:      {metrics['Recall']:.4f}")
    print("-" * 50)
    print(f"  ID Switches: {metrics['ID_Switches']}")
    print(f"  FP:          {metrics['FP']}")
    print(f"  FN:          {metrics['FN']}")
    print(f"  GT Tracks:   {metrics['GT_Tracks']}")
    print(f"  Pred Tracks: {metrics['Pred_Tracks']}")
    print("=" * 50)


def run_detector(
    video_path: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.5,
    car_class_id: int = 2,
    start_frac: float = 0.0,
    end_frac: float = 1.0,
    device: str = "cuda"
) -> Dict[int, List[BoundingBox]]:
    print(f"Loading model: {model_name} on {device}...")
    model = YOLO(model_name)

    print(f"Loading video: {video_path}...")
    video_source = VideoPartSource(video_path, start_frac=start_frac, end_frac=end_frac)

    print("Running inference...")
    pred_per_frame = {}
    current_frame_id = video_source.start_frame

    for frame in tqdm(video_source, total=len(video_source)):
        results = model.predict(frame, verbose=False, conf=conf_threshold, device=device)

        frame_detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])

                if cls_id == car_class_id:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])

                    detection = BoundingBox(
                        left=x1,
                        top=y1,
                        right=x2,
                        bottom=y2,
                        confidence=conf
                    )
                    frame_detections.append(detection)

        if frame_detections:
            pred_per_frame[current_frame_id] = frame_detections

        current_frame_id += 1

    return pred_per_frame


def visualize_tracking(
    video_path: str,
    detections: List[Detection],
    output_path: str,
    start_frac: float = 0.0,
    end_frac: float = 1.0
):
    dets_by_frame = defaultdict(list)
    for det in detections:
        dets_by_frame[det.frame_id].append(det)

    np.random.seed(42)
    colors = {}

    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors:
            colors[track_id] = tuple(int(c) for c in np.random.randint(0, 255, 3))
        return colors[track_id]

    video_source = VideoPartSource(video_path, start_frac=start_frac, end_frac=end_frac)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        video_source.fps,
        (video_source.width, video_source.height)
    )

    current_frame_id = video_source.start_frame
    for frame in tqdm(video_source, desc="Creating visualization", total=len(video_source)):
        if current_frame_id in dets_by_frame:
            for det in dets_by_frame[current_frame_id]:
                color = get_color(det.track_id)

                pt1 = (int(det.bbox.left), int(det.bbox.top))
                pt2 = (int(det.bbox.right), int(det.bbox.bottom))
                cv2.rectangle(frame, pt1, pt2, color, 2)

                cv2.putText(
                    frame,
                    f"ID: {det.track_id}",
                    (pt1[0], pt1[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        out.write(frame)
        current_frame_id += 1

    out.release()
    print(f"Visualization saved to: {output_path}")


def run_tracking_experiment(
    pred_per_frame: Dict[int, List[BoundingBox]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    iou_threshold: float = 0.3,
    max_age: int = 5
) -> Tuple[List[Detection], Dict[str, float]]:
    """Run tracking with given parameters and return detections + metrics."""
    tracker = MaxOverlapTracker(iou_threshold=iou_threshold, max_age=max_age)

    all_frame_ids = sorted(pred_per_frame.keys())
    for frame_id in all_frame_ids:
        detections = pred_per_frame[frame_id]
        tracker.update(frame_id, detections)

    tracked_detections = tracker.get_all_detections()
    metrics = compute_tracking_metrics(gt_with_tracks, tracked_detections)

    return tracked_detections, metrics


def optuna_parameter_search(
    pred_per_frame: Dict[int, List[BoundingBox]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    output_dir: str,
    n_trials: int = 200,
    metric: str = "HOTA"
) -> Dict:

    os.makedirs(output_dir, exist_ok=True)
    db_path = os.path.join(output_dir, "optuna_tracking.db")
    study_name = "tracking_overlap_study"

    def objective(trial):
        iou_threshold = trial.suggest_float("iou_threshold", 0.01, 0.8)
        max_age = trial.suggest_int("max_age", 1, 50)

        _, metrics = run_tracking_experiment(
            pred_per_frame, gt_with_tracks,
            iou_threshold=iou_threshold, max_age=max_age
        )

        trial.set_user_attr("HOTA", metrics["HOTA"])
        trial.set_user_attr("IDF1", metrics["IDF1"])
        trial.set_user_attr("MOTA", metrics["MOTA"])
        trial.set_user_attr("DetA", metrics["DetA"])
        trial.set_user_attr("AssA", metrics["AssA"])
        trial.set_user_attr("ID_Switches", metrics["ID_Switches"])
        trial.set_user_attr("Precision", metrics["Precision"])
        trial.set_user_attr("Recall", metrics["Recall"])

        return metrics[metric]

    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )

    print(f"\n{'='*60}")
    print(f"OPTUNA BAYESIAN HYPERPARAMETER SEARCH")
    print(f"{'='*60}")
    print(f"  Trials: {n_trials}")
    print(f"  Metric: {metric}")
    print(f"  Database: {db_path}")
    print(f"  Study: {study_name}")
    print(f"{'='*60}\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Best {metric}: {best_value:.4f}")
    print(f"  Best IoU Threshold: {best_params['iou_threshold']:.4f}")
    print(f"  Best Max Age: {best_params['max_age']}")
    print(f"{'='*60}\n")

    print("Generating visualization plots...")
    plots_dir = os.path.join(output_dir, "optuna_plots")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(plots_dir, "optimization_history.html"))
        fig.write_image(os.path.join(plots_dir, "optimization_history.png"), scale=2)
        print(f"  - Optimization history saved")
    except Exception as e:
        print(f"  - Could not generate optimization history: {e}")

    try:
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(plots_dir, "param_importances.html"))
        fig.write_image(os.path.join(plots_dir, "param_importances.png"), scale=2)
        print(f"  - Parameter importances saved")
    except Exception as e:
        print(f"  - Could not generate param importances: {e}")

    try:
        fig = plot_contour(study, params=["iou_threshold", "max_age"])
        fig.write_html(os.path.join(plots_dir, "contour.html"))
        fig.write_image(os.path.join(plots_dir, "contour.png"), scale=2)
        print(f"  - Contour plot saved")
    except Exception as e:
        print(f"  - Could not generate contour plot: {e}")

    try:
        fig = plot_parallel_coordinate(study)
        fig.write_html(os.path.join(plots_dir, "parallel_coordinate.html"))
        fig.write_image(os.path.join(plots_dir, "parallel_coordinate.png"), scale=2)
        print(f"  - Parallel coordinate plot saved")
    except Exception as e:
        print(f"  - Could not generate parallel coordinate: {e}")

    try:
        fig = plot_slice(study)
        fig.write_html(os.path.join(plots_dir, "slice.html"))
        fig.write_image(os.path.join(plots_dir, "slice.png"), scale=2)
        print(f"  - Slice plot saved")
    except Exception as e:
        print(f"  - Could not generate slice plot: {e}")

    results_summary = {
        "best_params": best_params,
        "best_value": best_value,
        "metric": metric,
        "n_trials": len(study.trials),
        "all_trials": []
    }

    for trial in study.trials:
        results_summary["all_trials"].append({
            "number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "user_attrs": trial.user_attrs
        })

    with open(os.path.join(output_dir, "optuna_results.json"), "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"  - Results JSON saved")

    _, best_metrics = run_tracking_experiment(
        pred_per_frame, gt_with_tracks,
        iou_threshold=best_params["iou_threshold"],
        max_age=best_params["max_age"]
    )

    print(f"\nAll outputs saved to: {output_dir}")

    return {
        "best_params": best_params,
        "best_value": best_value,
        "best_metrics": best_metrics,
        "study": study,
        "db_path": db_path
    }


def main():
    VIDEO_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"

    MODEL_NAME = "yolov8n.pt"
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3
    MAX_AGE = 5

    DO_PARAM_SEARCH = True
    CREATE_VIS = True
    N_TRIALS = 500

    OUTPUT_DIR = "/home/priubrogent/MCV/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(VIDEO_PATH):
        print(f"Video file NOT found at: {VIDEO_PATH}")
        return

    if not os.path.exists(XML_PATH):
        print(f"Annotation file NOT found at: {XML_PATH}")
        return

    print(f"Video: {VIDEO_PATH}")
    print(f"Annotations: {XML_PATH}")

    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)

    print("\n=== Step 1: Running Object Detection ===")
    pred_per_frame = run_detector(VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD)
    print(f"Detected objects in {len(pred_per_frame)} frames")

    if DO_PARAM_SEARCH:
        search_results = optuna_parameter_search(
            pred_per_frame, gt_with_tracks,
            output_dir=OUTPUT_DIR,
            n_trials=N_TRIALS,
            metric="HOTA"
        )

        best = search_results["best_params"]
        IOU_THRESHOLD = best["iou_threshold"]
        MAX_AGE = best["max_age"]
        print(f"\nUsing best parameters: IoU={IOU_THRESHOLD:.4f}, MaxAge={MAX_AGE}")

    print("\n=== Step 2: Running Maximum Overlap Tracker ===")
    tracked_detections, metrics = run_tracking_experiment(
        pred_per_frame, gt_with_tracks,
        iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE
    )

    print(f"Created {len(set(d.track_id for d in tracked_detections))} unique tracks")
    print(f"Total tracked detections: {len(tracked_detections)}")

    print_metrics(metrics)

    print("\n=== Step 3: Saving Results ===")
    pred_mot_path = os.path.join(OUTPUT_DIR, "pred_tracking.txt")
    save_tracking_results_mot(tracked_detections, pred_mot_path)
    print(f"Predictions saved to: {pred_mot_path}")

    gt_mot_path = os.path.join(OUTPUT_DIR, "gt_tracking.txt")
    save_gt_mot_format(gt_with_tracks, gt_mot_path)
    print(f"Ground truth saved to: {gt_mot_path}")

    if CREATE_VIS:
        print("\n=== Step 4: Creating Visualization ===")
        vis_path = os.path.join(OUTPUT_DIR, "tracking_visualization.mp4")
        visualize_tracking(VIDEO_PATH, tracked_detections, vis_path)

    print("\n=== Task 2.1 Complete ===")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
