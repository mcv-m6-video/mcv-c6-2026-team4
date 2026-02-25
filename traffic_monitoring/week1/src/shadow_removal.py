from typing import Protocol
import cv2
import numpy as np

from src.models.base import BackgroundModel


class ShadowRemover(Protocol):
    def remove_shadows(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        bg_model: BackgroundModel
    ) -> np.ndarray:
        pass


class NoShadowRemoval(ShadowRemover):
    def remove_shadows(self, mask, frame, bg_model):
        return mask


class HSVBackgroundComparison(ShadowRemover):
    def __init__(
        self,
        hue_threshold: float = 15.0,
        value_ratio_min: float = 0.5,
        value_ratio_max: float = 0.95,
        saturation_diff_threshold: float = 30.0
    ):
        self.hue_threshold = hue_threshold
        self.value_ratio_min = value_ratio_min
        self.value_ratio_max = value_ratio_max
        self.saturation_diff_threshold = saturation_diff_threshold

    def remove_shadows(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        bg_model: BackgroundModel
    ) -> np.ndarray:
        bg_frame = bg_model.background_image()
        bg_frame_uint8 = np.clip(bg_frame, 0, 255).astype(np.uint8)

        hsv_current = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bg = cv2.cvtColor(bg_frame_uint8, cv2.COLOR_BGR2HSV).astype(np.float32)

        h_curr, s_curr, v_curr = cv2.split(hsv_current)
        h_bg, s_bg, v_bg = cv2.split(hsv_bg)

        hue_diff = np.minimum(np.abs(h_curr - h_bg), 180 - np.abs(h_curr - h_bg))
        value_ratio = v_curr / (v_bg + 1e-6)
        sat_diff = np.abs(s_curr - s_bg)

        shadow_mask = (
            (hue_diff < self.hue_threshold) &
            (value_ratio > self.value_ratio_min) &
            (value_ratio < self.value_ratio_max) &
            (sat_diff < self.saturation_diff_threshold) &
            (mask > 0)
        )

        cleaned_mask = mask.copy()
        cleaned_mask[shadow_mask] = 0
        return cleaned_mask


class ChromaticityBackgroundComparison(ShadowRemover):
    def __init__(
        self,
        chromaticity_threshold: float = 0.1,
        intensity_ratio_min: float = 0.4,
        intensity_ratio_max: float = 0.9
    ):
        self.chromaticity_threshold = chromaticity_threshold
        self.intensity_ratio_min = intensity_ratio_min
        self.intensity_ratio_max = intensity_ratio_max

    def remove_shadows(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        bg_model: BackgroundModel
    ) -> np.ndarray:
        bg_frame = bg_model.background_image()

        frame_float = frame.astype(np.float32) + 1e-6
        bg_float = np.clip(bg_frame, 0, 255).astype(np.float32) + 1e-6

        intensity_curr = frame_float.sum(axis=2, keepdims=True)
        intensity_bg = bg_float.sum(axis=2, keepdims=True)

        chrom_curr = frame_float / intensity_curr
        chrom_bg = bg_float / intensity_bg

        chrom_dist = np.linalg.norm(chrom_curr - chrom_bg, axis=2)
        intensity_ratio = (intensity_curr / intensity_bg).squeeze()

        shadow_mask = (
            (chrom_dist < self.chromaticity_threshold) &
            (intensity_ratio < self.intensity_ratio_max) &
            (intensity_ratio > self.intensity_ratio_min) &
            (mask > 0)
        )

        cleaned_mask = mask.copy()
        cleaned_mask[shadow_mask] = 0
        return cleaned_mask
