import os
import pickle
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

from task_2_1 import (
    Detection, BoundingBox,
    load_annotations_with_tracks
)
from task_2_2 import KalmanFilterTracker


def load_mot_predictions(pred_file: str) -> Dict[int, List[Tuple[int, BoundingBox]]]:
    """Load MOT format predictions: frame,track_id,left,top,width,height,conf,..."""
    preds_by_frame = defaultdict(list)
    with open(pred_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0]) - 1  # MOT is 1-indexed
            track_id = int(parts[1])
            left = float(parts[2])
            top = float(parts[3])
            width = float(parts[4])
            height = float(parts[5])
            bbox = BoundingBox(left=left, top=top, right=left+width, bottom=top+height, confidence=1.0)
            preds_by_frame[frame].append((track_id, bbox))
    return dict(preds_by_frame)


def find_tracking_differences(
    preds_2_1: Dict[int, List[Tuple[int, BoundingBox]]],
    preds_2_2: Dict[int, List[Tuple[int, BoundingBox]]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
) -> List[Tuple[int, str, dict]]:
    """Find frames where Task 2.1 and 2.2 differ significantly."""
    differences = []

    all_frames = sorted(set(preds_2_1.keys()) | set(preds_2_2.keys()) | set(gt_with_tracks.keys()))

    for frame in all_frames:
        p1 = preds_2_1.get(frame, [])
        p2 = preds_2_2.get(frame, [])
        gt = gt_with_tracks.get(frame, [])

        n_p1 = len(p1)
        n_p2 = len(p2)
        n_gt = len(gt)

        # Track IDs in each
        ids_p1 = set(t[0] for t in p1)
        ids_p2 = set(t[0] for t in p2)

        # Check for differences
        diff_info = {
            'n_p1': n_p1, 'n_p2': n_p2, 'n_gt': n_gt,
            'ids_p1': ids_p1, 'ids_p2': ids_p2
        }

        # Different number of detections
        if abs(n_p1 - n_p2) >= 2:
            differences.append((frame, 'detection_count_diff', diff_info))

        # ID switch detection (track ID changes between consecutive frames)
        # This is harder to detect from single frames

    return differences


def generate_comparison_video(
    video_path: str,
    preds_2_1: Dict[int, List[Tuple[int, BoundingBox]]],
    preds_2_2: Dict[int, List[Tuple[int, BoundingBox]]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    output_dir: str,
    trail_length: int = 100
):
    """Create side-by-side comparison video of Task 2.1 vs 2.2."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video (side by side)
    output_path = os.path.join(output_dir, "comparison_2_1_vs_2_2.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

    np.random.seed(42)
    colors_p1 = {}
    colors_p2 = {}

    def get_color_p1(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors_p1:
            colors_p1[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return colors_p1[track_id]

    np.random.seed(42)  # Same seed for consistent colors
    def get_color_p2(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors_p2:
            colors_p2[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return colors_p2[track_id]

    track_history_p1 = defaultdict(list)
    track_history_p2 = defaultdict(list)

    print(f"Creating comparison video ({total_frames} frames)...")

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_p1 = frame.copy()
        frame_p2 = frame.copy()

        # Process Task 2.1
        if frame_idx in preds_2_1:
            for track_id, bbox in preds_2_1[frame_idx]:
                cx = int((bbox.left + bbox.right) / 2)
                cy = int((bbox.top + bbox.bottom) / 2)
                track_history_p1[track_id].append((cx, cy, frame_idx))

        # Process Task 2.2
        if frame_idx in preds_2_2:
            for track_id, bbox in preds_2_2[frame_idx]:
                cx = int((bbox.left + bbox.right) / 2)
                cy = int((bbox.top + bbox.bottom) / 2)
                track_history_p2[track_id].append((cx, cy, frame_idx))

        # Draw trails for 2.1
        for track_id, history in track_history_p1.items():
            recent = [(x, y) for x, y, f in history if frame_idx - f <= trail_length]
            if len(recent) > 1:
                color = get_color_p1(track_id)
                for i in range(1, len(recent)):
                    alpha = i / len(recent)
                    thickness = max(2, int(4 * alpha))
                    cv2.line(frame_p1, recent[i-1], recent[i], color, thickness)

        # Draw trails for 2.2
        for track_id, history in track_history_p2.items():
            recent = [(x, y) for x, y, f in history if frame_idx - f <= trail_length]
            if len(recent) > 1:
                color = get_color_p2(track_id)
                for i in range(1, len(recent)):
                    alpha = i / len(recent)
                    thickness = max(2, int(4 * alpha))
                    cv2.line(frame_p2, recent[i-1], recent[i], color, thickness)

        # Draw boxes for 2.1
        if frame_idx in preds_2_1:
            for track_id, bbox in preds_2_1[frame_idx]:
                color = get_color_p1(track_id)
                pt1 = (int(bbox.left), int(bbox.top))
                pt2 = (int(bbox.right), int(bbox.bottom))
                cv2.rectangle(frame_p1, pt1, pt2, color, 3)
                cv2.putText(frame_p1, f"ID:{track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw boxes for 2.2
        if frame_idx in preds_2_2:
            for track_id, bbox in preds_2_2[frame_idx]:
                color = get_color_p2(track_id)
                pt1 = (int(bbox.left), int(bbox.top))
                pt2 = (int(bbox.right), int(bbox.bottom))
                cv2.rectangle(frame_p2, pt1, pt2, color, 3)
                cv2.putText(frame_p2, f"ID:{track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Add labels
        cv2.putText(frame_p1, f"Task 2.1: Max Overlap | Frame {frame_idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_p2, f"Task 2.2: Kalman Filter | Frame {frame_idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Combine side by side
        combined = np.hstack([frame_p1, frame_p2])
        out.write(combined)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def create_task22_video_with_trails(
    video_path: str,
    preds: Dict[int, List[Tuple[int, BoundingBox]]],
    output_dir: str,
    trail_length: int = 100
):
    """Create video with trails for Task 2.2 only."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = os.path.join(output_dir, "tracking_full_with_trails.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    np.random.seed(42)
    colors = {}

    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors:
            colors[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return colors[track_id]

    track_history = defaultdict(list)

    print(f"Creating Task 2.2 video with trails ({total_frames} frames)...")

    for frame_idx in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in preds:
            for track_id, bbox in preds[frame_idx]:
                cx = int((bbox.left + bbox.right) / 2)
                cy = int((bbox.top + bbox.bottom) / 2)
                track_history[track_id].append((cx, cy, frame_idx))

        # Draw trails
        for track_id, history in track_history.items():
            recent = [(x, y) for x, y, f in history if frame_idx - f <= trail_length]
            if len(recent) > 1:
                color = get_color(track_id)
                for i in range(1, len(recent)):
                    alpha = i / len(recent)
                    thickness = max(2, int(4 * alpha))
                    cv2.line(frame, recent[i-1], recent[i], color, thickness)

        # Draw boxes
        if frame_idx in preds:
            for track_id, bbox in preds[frame_idx]:
                color = get_color(track_id)
                pt1 = (int(bbox.left), int(bbox.top))
                pt2 = (int(bbox.right), int(bbox.bottom))
                cv2.rectangle(frame, pt1, pt2, color, 3)
                cv2.putText(frame, f"ID:{track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Task 2.2: Kalman Filter | Frame {frame_idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")


def analyze_id_switches(
    preds: Dict[int, List[Tuple[int, BoundingBox]]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    iou_threshold: float = 0.5
) -> List[Tuple[int, int, int]]:
    """Find frames where ID switches occur by comparing with GT."""
    from task_2_1 import compute_iou

    switches = []
    last_match = {}  # pred_id -> gt_id

    for frame in sorted(set(preds.keys()) & set(gt_with_tracks.keys())):
        pred_boxes = preds[frame]
        gt_boxes = gt_with_tracks[frame]

        current_match = {}

        for pred_id, pred_bbox in pred_boxes:
            best_iou = 0
            best_gt_id = None
            for gt_bbox, gt_id in gt_boxes:
                iou = compute_iou(pred_bbox, gt_bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_gt_id = gt_id

            if best_gt_id is not None:
                current_match[pred_id] = best_gt_id

                if pred_id in last_match and last_match[pred_id] != best_gt_id:
                    switches.append((frame, pred_id, best_gt_id))

        last_match.update(current_match)

    return switches


if __name__ == "__main__":
    VIDEO_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"

    PRED_2_1 = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_1/pred_tracking.txt"
    PRED_2_2 = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_2/pred_tracking.txt"

    OUTPUT_DIR = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_2/qualitative"

    print("Loading predictions...")
    preds_2_1 = load_mot_predictions(PRED_2_1)
    preds_2_2 = load_mot_predictions(PRED_2_2)

    print("Loading ground truth...")
    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)

    print("\n=== Analyzing ID switches ===")
    switches_2_1 = analyze_id_switches(preds_2_1, gt_with_tracks)
    switches_2_2 = analyze_id_switches(preds_2_2, gt_with_tracks)
    print(f"Task 2.1 ID switches: {len(switches_2_1)}")
    print(f"Task 2.2 ID switches: {len(switches_2_2)}")

    if switches_2_1:
        print(f"  2.1 switch frames: {[s[0] for s in switches_2_1[:10]]}...")
    if switches_2_2:
        print(f"  2.2 switch frames: {[s[0] for s in switches_2_2[:10]]}...")

    print("\n=== Creating Task 2.2 video with trails ===")
    create_task22_video_with_trails(VIDEO_PATH, preds_2_2, OUTPUT_DIR, trail_length=100)

    print("\n=== Creating comparison video ===")
    generate_comparison_video(VIDEO_PATH, preds_2_1, preds_2_2, gt_with_tracks, OUTPUT_DIR, trail_length=100)

    print(f"\nAll results saved to: {OUTPUT_DIR}")
