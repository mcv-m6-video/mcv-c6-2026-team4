
from typing import Protocol
import cv2
import numpy as np


class VideoSource(Protocol):
    def __iter__(self):
        pass


class VideoPartSource(VideoSource):
    def __init__(self, path: str, start_frac: float = 0.0, end_frac: float = 1.0):
        assert (0.0 <= start_frac < end_frac <= 1.0)

        self.path = path
        self.start_frac = start_frac
        self.end_frac = end_frac

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.start_frame: int = int(total * start_frac)
        self.end_frame: int   = int(total * end_frac)
        self.n_frames = self.end_frame - self.start_frame
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()


    def __iter__(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")

        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

            for _ in range(self.end_frame - self.start_frame):
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # BGR to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # HWC to CHW
                yield np.ascontiguousarray(frame_rgb.transpose(2, 0, 1))
        finally:
            cap.release()

    def __len__(self) -> int:
        return self.end_frame - self.start_frame

    def __repr__(self) -> str:
        return (
            f"VideoPartialSource(path={self.path!r}, "
            f"frames=from {self.start_frame} to {self.end_frame}, "
            f"frac=[{self.start_frac}, {self.end_frac}])"
        )
