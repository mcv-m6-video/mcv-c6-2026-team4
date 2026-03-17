from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from src.world_and_camera_tracking import WorldTrack

# torch is optional — only needed for neural-network extractors
try:
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class FeatureExtractor(Protocol):
    """
    Extracts a single L2-normalised feature vector from a WorldTrack.

    The extractor receives the full WorldTrack so it can use anything:
    crops, bounding boxes, world positions, timestamps, etc.

    Returns None if no feature can be computed (e.g. empty track).
    """
    def __call__(self, world_track: WorldTrack) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def compute_features(
    world_tracks: list[WorldTrack],
    extractor: FeatureExtractor,
) -> None:
    """
    Runs `extractor` on every WorldTrack and writes the result to
    `wt.reid_feature`.  Mutates the list in place.

    Each extractor controls its own sampling strategy since it has full
    access to the WorldTrack (crops, bboxes, world positions, timestamps).
    """
    for wt in world_tracks:
        wt.reid_feature = extractor(wt)


def _sample_indices(n_total: int, n_samples: int) -> list[int]:
    """Evenly spaced indices for sampling `n_samples` from a sequence of length `n_total`."""
    return list(np.linspace(0, n_total - 1, min(n_total, n_samples), dtype=int))


# ---------------------------------------------------------------------------
# ReID extractor  (wraps the ft_net model used in mtmc.py)
# ---------------------------------------------------------------------------

class ReIDExtractor:
    """
    Extracts a ReID feature vector by averaging embeddings over sampled crops.

    Mirrors the logic from `extract_reid_features` in mtmc.py, but packaged
    as a reusable, stateful extractor that holds the model and preprocessing.

    Parameters
    ----------
    model:
        A ReID model whose forward pass returns ``(logits, feature)``.
        Should already be moved to `device`.
    transform:
        A torchvision transform pipeline (Resize, ToTensor, Normalize, ...).
    device:
        torch.device to run inference on.
    n_crops:
        Number of evenly spaced crops to sample per track.
    min_crop_area:
        Crops smaller than this (in pixels²) are skipped.
    """

    def __init__(
        self,
        model,
        transform,
        device,
        n_crops: int = 5,
        min_crop_area: int = 16 * 16,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch is required for ReIDExtractor")
        self.model = model
        self.transform = transform
        self.device = device
        self.n_crops = n_crops
        self.min_crop_area = min_crop_area

    def __call__(self, world_track: WorldTrack) -> np.ndarray | None:
        history = world_track.track.history
        if not history:
            return None

        indices = _sample_indices(len(history), self.n_crops)
        features = []

        self.model.eval()
        with torch.no_grad():
            for i in indices:
                crop = history[i].img
                if crop is None or crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                if h * w < self.min_crop_area:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                _, feat = self.model(tensor)
                feat = F.normalize(feat, p=2, dim=1)
                features.append(feat.cpu().numpy())

        if not features:
            return None

        avg = np.mean(features, axis=0)
        norm = np.linalg.norm(avg)
        return (avg / norm if norm > 0 else avg)[0]


# ---------------------------------------------------------------------------
# Color histogram extractor
# ---------------------------------------------------------------------------

class ColorHistogramExtractor:
    """
    Extracts a color histogram feature by averaging per-channel histograms
    over sampled crops.

    Simple and fast — no GPU required.  Useful as a baseline or complement
    to a deep ReID model.

    Parameters
    ----------
    n_bins:
        Number of histogram bins per channel.
    n_crops:
        Number of evenly spaced crops to sample per track.
    color_space:
        ``'hsv'`` (default, hue-saturaton-value — more perceptually uniform),
        ``'rgb'``, or ``'bgr'`` (raw OpenCV).
    channels:
        Which channels to include, e.g. ``(0, 1)`` to use only H and S in HSV.
        Defaults to all three channels.
    min_crop_area:
        Crops smaller than this (pixels²) are skipped.
    """

    def __init__(
        self,
        n_bins: int = 32,
        n_crops: int = 5,
        color_space: str = "hsv",
        channels: tuple[int, ...] | None = None,
        min_crop_area: int = 16 * 16,
    ) -> None:
        self.n_bins = n_bins
        self.n_crops = n_crops
        self.color_space = color_space.lower()
        self.channels = channels
        self.min_crop_area = min_crop_area

        _valid = {"hsv", "rgb", "bgr"}
        if self.color_space not in _valid:
            raise ValueError(f"color_space must be one of {_valid}, got {color_space!r}")

    def __call__(self, world_track: WorldTrack) -> np.ndarray | None:
        history = world_track.track.history
        if not history:
            return None

        indices = _sample_indices(len(history), self.n_crops)
        hists: list[np.ndarray] = []

        for i in indices:
            crop = history[i].img
            if crop is None or crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if h * w < self.min_crop_area:
                continue
            hist = self._histogram(crop)
            if hist is not None:
                hists.append(hist)

        if not hists:
            return None

        avg = np.mean(hists, axis=0)
        norm = np.linalg.norm(avg)
        return avg / norm if norm > 0 else avg

    def _histogram(self, bgr_img: np.ndarray) -> np.ndarray | None:
        if self.color_space == "hsv":
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        elif self.color_space == "rgb":
            img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        else:
            img = bgr_img

        channels = self.channels if self.channels is not None else tuple(range(img.shape[2]))
        parts = []
        for ch in channels:
            h = cv2.calcHist([img], [ch], None, [self.n_bins], [0, 256]).flatten()
            norm = np.linalg.norm(h)
            parts.append(h / norm if norm > 0 else h)

        return np.concatenate(parts) if parts else None


# ---------------------------------------------------------------------------
# ft_net extractor  (self-contained: loads model + transform from disk)
# ---------------------------------------------------------------------------

class FtNetExtractor:
    """
    Self-contained ReID extractor built on the ``ft_net`` model used in
    ``mtmc.py``.  Loads the model weights and sets up the standard ImageNet
    preprocessing internally — no external setup required.

    Parameters
    ----------
    weights_path:
        Path to the ``.pth`` checkpoint (e.g. ``"./src/net_19.pth"``).
    device:
        ``torch.device`` or a string such as ``"cuda"`` / ``"cpu"``.
        Defaults to CUDA if available, otherwise CPU.
    n_crops:
        Number of evenly spaced crops to sample per track.
    class_num, ibn, linear_num, circle:
        ft_net architecture arguments — must match the checkpoint.
    min_crop_area:
        Crops whose pixel area falls below this threshold are skipped.
    """

    # Standard ImageNet normalisation used when the ft_net was trained
    _MEAN = [0.485, 0.456, 0.406]
    _STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        weights_path: str | Path,
        device: str | torch.device | None = None,
        n_crops: int = 5,
        class_num: int = 34071,
        ibn: bool = True,
        linear_num: int = 2048,
        circle: bool = True,
        min_crop_area: int = 16 * 16,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and torchvision are required for FtNetExtractor")

        from src.model import ft_net  # local import keeps the module usable without torch

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.n_crops = n_crops
        self.min_crop_area = min_crop_area

        model = ft_net(
            class_num=class_num,
            ibn=ibn,
            linear_num=linear_num,
            circle=circle,
        )
        model.load_state_dict(
            torch.load(str(weights_path), map_location=self.device),
            strict=False,
        )
        self.model = model.to(self.device)

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(self._MEAN, self._STD),
        ])

    def __call__(self, world_track: WorldTrack) -> np.ndarray | None:
        history = world_track.track.history
        if not history:
            return None

        indices = _sample_indices(len(history), self.n_crops)
        features = []

        self.model.eval()
        with torch.no_grad():
            for i in indices:
                crop = history[i].img
                if crop is None or crop.size == 0:
                    continue
                h, w = crop.shape[:2]
                if h * w < self.min_crop_area:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
                _, feat = self.model(tensor)
                feat = F.normalize(feat, p=2, dim=1)
                features.append(feat.cpu().numpy())

        if not features:
            return None

        avg = np.mean(features, axis=0)
        norm = np.linalg.norm(avg)
        return (avg / norm if norm > 0 else avg)[0]
