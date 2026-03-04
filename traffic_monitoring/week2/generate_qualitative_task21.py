import os
import pickle
import cv2
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from PIL import Image

from task_2_1 import (
    MaxOverlapTracker, Detection, BoundingBox,
    load_annotations_with_tracks
)

def generate_qualitative_results(
    video_path: str,
    pred_per_frame: Dict[int, List[BoundingBox]],
    gt_with_tracks: Dict[int, List[Tuple[BoundingBox, int]]],
    output_dir: str,
    iou_threshold: float = 0.1094,
    max_age: int = 50,
    sample_frames: List[int] = None,
    create_gif: bool = True
):
    os.makedirs(output_dir, exist_ok=True)

    tracker = MaxOverlapTracker(iou_threshold=iou_threshold, max_age=max_age)

    all_frame_ids = sorted(pred_per_frame.keys())
    for frame_id in all_frame_ids:
        detections = pred_per_frame[frame_id]
        tracker.update(frame_id, detections)

    tracked_detections = tracker.get_all_detections()

    dets_by_frame = defaultdict(list)
    for det in tracked_detections:
        dets_by_frame[det.frame_id].append(det)

    np.random.seed(42)
    pred_colors = {}
    gt_colors = {}

    def get_pred_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in pred_colors:
            pred_colors[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return pred_colors[track_id]

    def get_gt_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in gt_colors:
            gt_colors[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return gt_colors[track_id]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if sample_frames is None:
        sample_frames = [100, 300, 500, 800, 1000, 1200]

    sample_frames = [f for f in sample_frames if f < total_frames]

    print(f"Generating qualitative frames: {sample_frames}")

    saved_frames_pred = []
    saved_frames_compare = []

    for target_frame in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_pred = frame.copy()
        frame_gt = frame.copy()
        frame_compare = frame.copy()

        if target_frame in dets_by_frame:
            for det in dets_by_frame[target_frame]:
                color = get_pred_color(det.track_id)
                pt1 = (int(det.bbox.left), int(det.bbox.top))
                pt2 = (int(det.bbox.right), int(det.bbox.bottom))
                cv2.rectangle(frame_pred, pt1, pt2, color, 3)
                cv2.putText(frame_pred, f"ID:{det.track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(frame_compare, pt1, pt2, color, 3)
                cv2.putText(frame_compare, f"P:{det.track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if target_frame in gt_with_tracks:
            for bbox, gt_id in gt_with_tracks[target_frame]:
                color = get_gt_color(gt_id)
                pt1 = (int(bbox.left), int(bbox.top))
                pt2 = (int(bbox.right), int(bbox.bottom))
                cv2.rectangle(frame_gt, pt1, pt2, color, 3)
                cv2.putText(frame_gt, f"GT:{gt_id}", (pt1[0], pt2[1] + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                cv2.rectangle(frame_compare, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_compare, f"G:{gt_id}", (pt1[0], pt2[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame_pred, f"Frame {target_frame} - Predictions", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_gt, f"Frame {target_frame} - Ground Truth", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame_compare, f"Frame {target_frame} - Pred(color) vs GT(green)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(output_dir, f"pred_frame_{target_frame:04d}.png"), frame_pred)
        cv2.imwrite(os.path.join(output_dir, f"gt_frame_{target_frame:04d}.png"), frame_gt)
        cv2.imwrite(os.path.join(output_dir, f"compare_frame_{target_frame:04d}.png"), frame_compare)

        saved_frames_pred.append(frame_pred)
        saved_frames_compare.append(frame_compare)

        print(f"  Saved frame {target_frame}")

    if create_gif and saved_frames_pred:
        print("Creating GIF animations...")

        pil_frames_pred = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in saved_frames_pred]
        pil_frames_pred[0].save(
            os.path.join(output_dir, "tracking_samples.gif"),
            save_all=True,
            append_images=pil_frames_pred[1:],
            duration=1000,
            loop=0
        )
        print(f"  Saved tracking_samples.gif")

        pil_frames_compare = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in saved_frames_compare]
        pil_frames_compare[0].save(
            os.path.join(output_dir, "tracking_comparison.gif"),
            save_all=True,
            append_images=pil_frames_compare[1:],
            duration=1000,
            loop=0
        )
        print(f"  Saved tracking_comparison.gif")

    cap.release()

    print("\nGenerating full tracking video with trails...")
    create_full_video_with_trails(video_path, dets_by_frame, output_dir, trail_length=150)

    print(f"\nAll qualitative results saved to: {output_dir}")


def create_continuous_gif(
    video_path: str,
    dets_by_frame: Dict[int, List[Detection]],
    output_dir: str,
    start_frame: int = 100,
    num_frames: int = 60,
    fps: int = 10,
    trail_length: int = 30
):
    cap = cv2.VideoCapture(video_path)

    np.random.seed(42)
    colors = {}
    def get_color(track_id: int) -> Tuple[int, int, int]:
        if track_id not in colors:
            colors[track_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
        return colors[track_id]

    track_history = defaultdict(list)

    frames_pil = []

    for frame_idx in range(start_frame, start_frame + num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in dets_by_frame:
            for det in dets_by_frame[frame_idx]:
                cx = int((det.bbox.left + det.bbox.right) / 2)
                cy = int((det.bbox.top + det.bbox.bottom) / 2)
                track_history[det.track_id].append((cx, cy, frame_idx))

        for track_id, history in track_history.items():
            recent = [(x, y) for x, y, f in history if frame_idx - f <= trail_length]
            if len(recent) > 1:
                color = get_color(track_id)
                for i in range(1, len(recent)):
                    alpha = i / len(recent)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(frame, recent[i-1], recent[i], color, thickness)

        if frame_idx in dets_by_frame:
            for det in dets_by_frame[frame_idx]:
                color = get_color(det.track_id)
                pt1 = (int(det.bbox.left), int(det.bbox.top))
                pt2 = (int(det.bbox.right), int(det.bbox.bottom))
                cv2.rectangle(frame, pt1, pt2, color, 3)
                cv2.putText(frame, f"ID:{det.track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Frame {frame_idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        frame_resized = cv2.resize(frame, (960, 540))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frames_pil.append(Image.fromarray(frame_rgb))

    cap.release()

    if frames_pil:
        duration = int(1000 / fps)
        frames_pil[0].save(
            os.path.join(output_dir, "tracking_with_trails.gif"),
            save_all=True,
            append_images=frames_pil[1:],
            duration=duration,
            loop=0
        )
        print(f"  Saved tracking_with_trails.gif ({len(frames_pil)} frames)")


def create_full_video_with_trails(
    video_path: str,
    dets_by_frame: Dict[int, List[Detection]],
    output_dir: str,
    trail_length: int = 30
):
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

    print(f"  Processing {total_frames} frames...")
    from tqdm import tqdm

    for frame_idx in tqdm(range(total_frames), desc="Creating video"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in dets_by_frame:
            for det in dets_by_frame[frame_idx]:
                cx = int((det.bbox.left + det.bbox.right) / 2)
                cy = int((det.bbox.top + det.bbox.bottom) / 2)
                track_history[det.track_id].append((cx, cy, frame_idx))

        for track_id, history in track_history.items():
            recent = [(x, y) for x, y, f in history if frame_idx - f <= trail_length]
            if len(recent) > 1:
                color = get_color(track_id)
                for i in range(1, len(recent)):
                    alpha = i / len(recent)
                    thickness = max(2, int(4 * alpha))
                    cv2.line(frame, recent[i-1], recent[i], color, thickness)

        if frame_idx in dets_by_frame:
            for det in dets_by_frame[frame_idx]:
                color = get_color(det.track_id)
                pt1 = (int(det.bbox.left), int(det.bbox.top))
                pt2 = (int(det.bbox.right), int(det.bbox.bottom))
                cv2.rectangle(frame, pt1, pt2, color, 3)
                cv2.putText(frame, f"ID:{det.track_id}", (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Frame {frame_idx}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    VIDEO_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/AICity_data/AICity_data/train/S03/c010/vdo.avi"
    XML_PATH = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/ai_challenge_s03_c010-full_annotation.xml"
    RESULTS_DIR = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_1"
    OUTPUT_DIR = "/home/riubro/mcv-c6-2026-team4/traffic_monitoring/week2/results/task_2_1/qualitative"

    PRED_PKL = os.path.join(RESULTS_DIR, "pred_per_frame.pkl")

    print("Loading predictions...")
    with open(PRED_PKL, "rb") as f:
        pred_per_frame = pickle.load(f)

    print("Loading ground truth...")
    _, gt_with_tracks = load_annotations_with_tracks(XML_PATH)

    generate_qualitative_results(
        video_path=VIDEO_PATH,
        pred_per_frame=pred_per_frame,
        gt_with_tracks=gt_with_tracks,
        output_dir=OUTPUT_DIR,
        iou_threshold=0.1094,
        max_age=50,
        sample_frames=[100, 300, 500, 800, 1000, 1200, 1500],
        create_gif=True
    )
