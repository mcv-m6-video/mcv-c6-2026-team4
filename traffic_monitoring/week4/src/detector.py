"""
Detector wrappers that normalize different model APIs to a common interface.

All wrappers are callable as:

    results = detector(frame_bgr, verbose=False, conf=threshold)
    result  = results[0]
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])

This is the Ultralytics convention that the rest of the pipeline expects.
Ultralytics models (YOLO, RT-DETR) satisfy it natively.
FasterRCNN is wrapped to produce the same interface.

Car class indices by detector / weight set
------------------------------------------
yolo (yolov10s_coco.pt — custom single-class):  0
yolo_coco (standard Ultralytics COCO weights):   2
rtdetr (Ultralytics COCO):                        2
fasterrcnn (torchvision COCO-91, 1-indexed):      3
"""
from __future__ import annotations

import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# Known car-class indices per detector type
# (override with --car-class if your weights differ)
# ---------------------------------------------------------------------------
CAR_CLASS_BY_DETECTOR: dict[str, int] = {
    "yolo":        0,   # custom yolov10s_coco.pt used in this project
    "yolo_coco":   2,   # standard Ultralytics COCO weights (yolov8, v10, v12…)
    "rtdetr":      2,   # Ultralytics RT-DETR COCO
    "fasterrcnn":  3,   # torchvision COCO-91 (background=0, car=3)
}


# ---------------------------------------------------------------------------
# Minimal box / result adapters — make torchvision output look like Ultralytics
# ---------------------------------------------------------------------------

class _Box:
    """Wraps a single detection so that box.xyxy[0], box.conf[0], box.cls[0] work."""

    def __init__(
        self,
        xyxy: list[float],
        conf: float,
        cls: int,
    ) -> None:
        self.xyxy = [torch.tensor(xyxy, dtype=torch.float32)]
        self.conf = [torch.tensor([conf], dtype=torch.float32)]
        self.cls  = [torch.tensor([cls],  dtype=torch.float32)]


class _Boxes:
    """Iterable collection of _Box objects, mimicking result.boxes."""

    def __init__(self, boxes: list[_Box]) -> None:
        self._boxes = boxes

    def __iter__(self):
        return iter(self._boxes)

    def __len__(self) -> int:
        return len(self._boxes)


class _Result:
    """Minimal mimic of a single Ultralytics result object."""

    def __init__(self, boxes: list[_Box]) -> None:
        self.boxes = _Boxes(boxes)


# ---------------------------------------------------------------------------
# Faster R-CNN wrapper
# ---------------------------------------------------------------------------

class FasterRCNNDetector:
    """
    Wraps torchvision Faster R-CNN to look like an Ultralytics YOLO model.

    Parameters
    ----------
    backbone:
        "resnet50"  → fasterrcnn_resnet50_fpn_v2  (best accuracy / speed balance)
        "resnet101" → (not available in stock torchvision — falls back to resnet50)
        "mobilenet" → fasterrcnn_mobilenet_v3_large_fpn  (fastest, least accurate)
    device:
        torch.device to run on.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")

        if backbone == "mobilenet":
            weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
            self._model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
                weights=weights
            )
        else:
            # resnet50_v2 is strictly better than v1 with similar speed
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self._model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
                weights=weights
            )

        self._model = self._model.to(self.device).eval()

        # Pre-build the transform from the model's metadata
        self._transform = weights.transforms()

        print(f"  [FasterRCNN] backbone={backbone}  device={self.device}")

    @torch.no_grad()
    def __call__(
        self,
        frame_bgr: np.ndarray,
        verbose: bool = False,
        conf: float = 0.5,
    ) -> list[_Result]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame_bgr:
            HxWx3 uint8 numpy array (OpenCV format).
        conf:
            Minimum score threshold — detections below this are discarded.

        Returns
        -------
        [_Result] — single-element list so callers can index [0] like Ultralytics.
        """
        import cv2

        # BGR → RGB tensor
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        # torchvision inference
        output = self._model([tensor])[0]

        boxes_out: list[_Box] = []
        for xyxy, score, label in zip(
            output["boxes"].cpu(),
            output["scores"].cpu(),
            output["labels"].cpu(),
        ):
            if float(score) < conf:
                continue
            boxes_out.append(
                _Box(
                    xyxy=xyxy.tolist(),
                    conf=float(score),
                    cls=int(label),
                )
            )

        return [_Result(boxes_out)]
