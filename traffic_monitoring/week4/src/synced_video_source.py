from pathlib import Path
from typing import Iterator, Optional
import cv2
import numpy as np

from dataset import Camera, Sequence


class SyncedCameraSource:
    """
    Iterates a camera's video and yields (frame_bgr, global_timestamp) pairs.

    Global timestamp is defined as:
        t = camera.start_timestamp + frame_index / fps

    where `start_timestamp` is the camera's offset (in seconds) within the
    sequence, loaded from cam_timestamp/SXX.txt.  This places every frame on
    a shared time axis across all cameras in the same sequence.

    Usage:
        src = SyncedCameraSource(camera)
        for frame, t in src:
            ...  # t is seconds on the sequence clock
    """

    def __init__(self, camera: Camera, fps: Optional[float] = None):
        """
        Parameters
        ----------
        camera:
            A `Camera` dataclass loaded by `AICityDataset`.
        fps:
            Override the frame rate.  If None, reads it from the video header
            via OpenCV.  Useful to correct metadata errors or to pass 8.0 for
            the one camera (S03/c015) that records at 8 fps.
        """
        self.camera = camera

        if fps is not None:
            self.fps = fps
        else:
            cap = cv2.VideoCapture(str(camera.video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {camera.video_path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

        self.start_offset: float = camera.start_timestamp
        self.frame_count: int = camera.frame_count

        # Absolute time bounds on the sequence clock
        self.t_start: float = self.start_offset
        self.t_end: float = self.start_offset + self.frame_count / self.fps

    # ------------------------------------------------------------------

    def global_time(self, frame_idx: int) -> float:
        """Returns the global timestamp for a given 0-based frame index."""
        return self.start_offset + frame_idx / self.fps

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """Yields (frame_bgr, global_timestamp_seconds) for every frame."""
        cap = cv2.VideoCapture(str(self.camera.video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.camera.video_path}")
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame_idx, frame, self.global_time(frame_idx)
                frame_idx += 1
        finally:
            cap.release()

    def __len__(self) -> int:
        return self.frame_count

    def __repr__(self) -> str:
        return (
            f"SyncedCameraSource(camera={self.camera.id!r}, "
            f"fps={self.fps}, offset={self.start_offset:.3f}s, "
            f"frames={self.frame_count})"
        )


# ---------------------------------------------------------------------------


class MultiCameraSync:
    """
    Iterates multiple cameras in lock-step on a shared time grid.

    At each tick the iterator yields the nearest available frame from every
    camera, producing a consistent snapshot across cameras.

    Only the *intersection* of all cameras' time windows is iterated, so
    every camera always has a valid frame in the output.

    Usage:
        sync = MultiCameraSync.from_sequence(sequence, tick_fps=10.0)
        for t, frames in sync:
            c001_frame, c001_t = frames["c001"]
            ...

    Parameters
    ----------
    sources:
        Mapping from camera id to `SyncedCameraSource`.
    tick_fps:
        Rate of the output time grid in Hz.  Defaults to 10.0 (the majority
        camera frame rate in the dataset).
    """

    def __init__(
        self,
        sources: dict[str, SyncedCameraSource],
        tick_fps: float = 10.0,
    ):
        self.sources = sources
        self.tick_fps = tick_fps

        # Intersection of all time windows
        self.t_start: float = max(src.t_start for src in sources.values())
        self.t_end: float = min(src.t_end for src in sources.values())

        if self.t_start >= self.t_end:
            raise ValueError(
                "Cameras have no overlapping time window. "
                f"Computed window: [{self.t_start:.3f}, {self.t_end:.3f}]"
            )

    # ------------------------------------------------------------------

    @classmethod
    def from_sequence(
        cls,
        sequence: Sequence,
        tick_fps: float = 10.0,
        fps_overrides: Optional[dict[str, float]] = None,
    ) -> "MultiCameraSync":
        """
        Build a `MultiCameraSync` from a `Sequence` object directly.

        Parameters
        ----------
        sequence:
            A `Sequence` loaded by `AICityDataset`.
        tick_fps:
            Output tick rate in Hz.
        fps_overrides:
            Per-camera fps override, e.g. ``{"c015": 8.0}`` for S03.
            Any camera not listed uses the fps from its video header.
        """
        overrides = fps_overrides or {}
        sources = {
            cam_id: SyncedCameraSource(cam, fps=overrides.get(cam_id))
            for cam_id, cam in sequence.cameras.items()
        }
        return cls(sources, tick_fps=tick_fps)

    # ------------------------------------------------------------------

    def __iter__(
        self,
    ) -> Iterator[tuple[float, dict[str, tuple[np.ndarray, float]]]]:
        """
        Yields ``(tick_time, frames)`` at each output tick.

        ``frames`` is a dict mapping camera_id to ``(frame_bgr, frame_time)``
        where ``frame_time`` is the actual timestamp of the chosen frame
        (may differ slightly from ``tick_time`` due to frame-rate differences).
        """
        dt = 1.0 / self.tick_fps

        # Open all video captures and seek to the start of the common window
        caps: dict[str, cv2.VideoCapture] = {}
        # frame_next[cam_id] = index of the frame that cap.read() will return next
        frame_next: dict[str, int] = {}
        # buffers hold the last-read (frame, timestamp) per camera
        buffers: dict[str, tuple[np.ndarray, float]] = {}

        try:
            for cam_id, src in self.sources.items():
                cap = cv2.VideoCapture(str(src.camera.video_path))
                if not cap.isOpened():
                    raise FileNotFoundError(
                        f"Cannot open video: {src.camera.video_path}"
                    )
                caps[cam_id] = cap

                # Skip frames that precede the common start time
                skip = max(0, int((self.t_start - src.start_offset) * src.fps))
                if skip > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)
                frame_next[cam_id] = skip

                # Prime the buffer with the first valid frame
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        f"Could not read first frame from {src.camera.video_path}"
                    )
                t_frame = src.global_time(frame_next[cam_id])
                frame_next[cam_id] += 1
                buffers[cam_id] = (frame, t_frame)

            # Main tick loop — integer counter avoids float accumulation drift
            n_ticks = int((self.t_end - self.t_start) * self.tick_fps)
            for step in range(n_ticks + 1):
                t_tick = self.t_start + step * dt

                for cam_id, src in self.sources.items():
                    cap = caps[cam_id]
                    # Advance the buffer while the *next* frame is closer to t_tick
                    while True:
                        _, t_current = buffers[cam_id]
                        t_next_frame = src.global_time(frame_next[cam_id])

                        if t_next_frame > src.t_end:
                            break  # no more frames in this camera
                        if abs(t_next_frame - t_tick) >= abs(t_current - t_tick):
                            break  # current frame is already the closest

                        ret, frame = cap.read()
                        if not ret:
                            break
                        t_frame = src.global_time(frame_next[cam_id])
                        frame_next[cam_id] += 1
                        buffers[cam_id] = (frame, t_frame)

                yield t_tick, dict(buffers)  # shallow copy of buffer dict

        finally:
            for cap in caps.values():
                cap.release()

    def __len__(self) -> int:
        """Number of ticks in the common time window."""
        return int((self.t_end - self.t_start) * self.tick_fps) + 1

    def __repr__(self) -> str:
        cams = list(self.sources.keys())
        return (
            f"MultiCameraSync(cameras={cams}, "
            f"tick_fps={self.tick_fps}, "
            f"window=[{self.t_start:.3f}s, {self.t_end:.3f}s])"
        )
