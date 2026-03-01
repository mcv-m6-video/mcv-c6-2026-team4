import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import cv2
import numpy as np
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


def run_detector(
    video_path: str,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.5,
    car_class_id: int = 2,
    start_frac: float = 0.0,
    end_frac: float = 1.0
) -> Dict[int, List[BoundingBox]]:
    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)

    print(f"Loading video: {video_path}...")
    video_source = VideoPartSource(video_path, start_frac=start_frac, end_frac=end_frac)

    print("Running inference...")
    pred_per_frame = {}
    current_frame_id = video_source.start_frame

    for frame in tqdm(video_source, total=len(video_source)):
        results = model.predict(frame, verbose=False, conf=conf_threshold)

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


def main():
    VIDEO_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"

    MODEL_NAME = "yolov8n.pt"
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.3
    MAX_AGE = 5

    OUTPUT_DIR = "results/task_2_1"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(VIDEO_PATH):
        print(f"Video file NOT found at: {VIDEO_PATH}")
        return

    if not os.path.exists(XML_PATH):
        print(f"Annotation file NOT found at: {XML_PATH}")
        return

    print(f"Video: {VIDEO_PATH}")
    print(f"Annotations: {XML_PATH}")

    print("\n=== Step 1: Running Object Detection ===")
    pred_per_frame = run_detector(VIDEO_PATH, MODEL_NAME, CONF_THRESHOLD)
    print(f"Detected objects in {len(pred_per_frame)} frames")

    print("\n=== Step 2: Running Maximum Overlap Tracker ===")
    tracker = MaxOverlapTracker(iou_threshold=IOU_THRESHOLD, max_age=MAX_AGE)

    all_frame_ids = sorted(pred_per_frame.keys())
    for frame_id in tqdm(all_frame_ids, desc="Tracking"):
        detections = pred_per_frame[frame_id]
        tracker.update(frame_id, detections)

    tracked_detections = tracker.get_all_detections()
    print(f"Created {len(tracker.tracks)} unique tracks")
    print(f"Total tracked detections: {len(tracked_detections)}")

    print("\n=== Step 3: Saving Results ===")
    pred_mot_path = os.path.join(OUTPUT_DIR, "pred_tracking.txt")
    save_tracking_results_mot(tracked_detections, pred_mot_path)
    print(f"Predictions saved to: {pred_mot_path}")

    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)
    gt_mot_path = os.path.join(OUTPUT_DIR, "gt_tracking.txt")
    save_gt_mot_format(gt_with_tracks, gt_mot_path)
    print(f"Ground truth saved to: {gt_mot_path}")

    print("\n=== Step 4: Creating Visualization ===")
    vis_path = os.path.join(OUTPUT_DIR, "tracking_visualization.mp4")
    visualize_tracking(VIDEO_PATH, tracked_detections, vis_path)

    print("\n=== Task 2.1 Complete ===")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
