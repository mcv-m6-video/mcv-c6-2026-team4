import cv2
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

from src.object_detection import BoundingBox, CarDetector
from src.evaluation import evaluate_detections, load_annotations, show_metrics
from src.video_source import VideoPartSource
from try_still_model import draw_bboxes


ANNOTATIONS_PATH = "../ai_challenge_s03_c010-full_annotation.xml"
VIDEO_PATH = "../AICity_data/AICity_data/train/S03/c010/vdo.avi"
MASKS_VIDEO_PATH = "/home/arnau-marcos-almansa/workspace/ZBS/zbs_result/vdo_alternative.mp4"

OUTPUT_MASKS = "zbs_predicted_masks.avi"
OUTPUT_DETECTIONS = "zbs_predicted_detections.avi"
OUTPUT_SIDE_BY_SIDE = "zbs_side_by_side.avi"


def load_annotations(path: str):
    tree = ET.parse(path)
    boxes = tree.findall("track/box")
    boxes_per_frame = dict()
    for box in boxes:
        frame    = int(box.get("frame"))
        xtl      = float(box.get("xtl"))
        ytl      = float(box.get("ytl"))
        xbr      = float(box.get("xbr"))
        ybr      = float(box.get("ybr"))

        parked = None
        for attr in box.findall("attribute"):
            if attr.get("name") == "parked":
                parked = attr.text == "true"

        bbox = BoundingBox(top=ytl, bottom=ybr, left=xtl, right=xbr, confidence=1.0)

        if frame not in boxes_per_frame:
            boxes_per_frame[frame] = [bbox]
        else:
            boxes_per_frame[frame].append(bbox)

    return boxes_per_frame


def viridis_to_binary(frame_bgr: np.ndarray) -> np.ndarray:
    r = frame_bgr[:, :, 2].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    b = frame_bgr[:, :, 0].astype(np.float32)
    return ((r + g) - 2 * b > 0).astype(np.uint8)


def iter_masks_video(path: str, start_frame: int, n_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open masks video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


if __name__ == "__main__":
    video_source = VideoPartSource(VIDEO_PATH, 0.25, 1.0)

    _cap = cv2.VideoCapture(MASKS_VIDEO_PATH)
    _total_masks = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()
    _total_orig = video_source.start_frame + video_source.n_frames
    masks_frame_offset = _total_orig - _total_masks
    masks_start = video_source.start_frame - masks_frame_offset
    print(f"Original: {_total_orig} frames, Masks: {_total_masks} frames, offset: {masks_frame_offset}")
    print(f"Seeking masks video to frame {masks_start} (original test start: {video_source.start_frame})")

    masks_source = iter_masks_video(MASKS_VIDEO_PATH, masks_start, video_source.n_frames)

    annotations = load_annotations(ANNOTATIONS_PATH)

    detector = CarDetector(
        area=[1550, 10_000_000_000],
        aspect_ratio=[0.22118807684047892, 4.474026431574192],
        fill_ratio=[0.20821692159573382, 1.0],
    )

    mask_writer = cv2.VideoWriter(
        OUTPUT_MASKS,
        cv2.VideoWriter_fourcc(*"XVID"),
        video_source.fps,
        (video_source.width, video_source.height),
        isColor=False,
    )
    detection_writer = cv2.VideoWriter(
        OUTPUT_DETECTIONS,
        cv2.VideoWriter_fourcc(*"XVID"),
        video_source.fps,
        (video_source.width, video_source.height),
    )
    side_by_side_writer = cv2.VideoWriter(
        OUTPUT_SIDE_BY_SIDE,
        cv2.VideoWriter_fourcc(*"XVID"),
        video_source.fps,
        (video_source.width * 2, video_source.height),
    )

    predictions = {}

    print("Processing test frames...")
    for orig_frame, mask_frame, frame_id in tqdm(
        zip(video_source, masks_source, range(video_source.start_frame, video_source.end_frame)),
        total=video_source.n_frames,
    ):
        binary_mask = viridis_to_binary(mask_frame)
        detections = detector.detect(binary_mask)
        predictions[frame_id] = detections

        mask_u8 = binary_mask * 255
        det_frame = draw_bboxes(orig_frame, detections, (0, 255, 0))
        if frame_id in annotations:
            det_frame = draw_bboxes(det_frame, annotations[frame_id], (255, 0, 0))

        mask_writer.write(mask_u8)
        detection_writer.write(det_frame)
        side_by_side_writer.write(np.concatenate([cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR), det_frame], axis=1))

    mask_writer.release()
    detection_writer.release()
    side_by_side_writer.release()

    print(f"\nSaved: {OUTPUT_MASKS}, {OUTPUT_DETECTIONS}, {OUTPUT_SIDE_BY_SIDE}")

    gt_for_eval = {fid: boxes for fid, boxes in annotations.items() if fid in predictions}
    metrics = evaluate_detections(gt_for_eval, predictions)
    show_metrics(metrics)
