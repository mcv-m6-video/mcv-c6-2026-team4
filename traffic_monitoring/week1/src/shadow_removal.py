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
    """
    Remove shadows by comparing HSV values to background model.

    Shadows darken the background but preserve hue and have similar saturation.
    This distinguishes them from actual foreground objects (like gray cars) which
    have different hue/saturation from the background.
    """

    def __init__(
        self,
        hue_threshold: float = 15.0,
        value_ratio_min: float = 0.5,
        value_ratio_max: float = 0.95,
        saturation_diff_threshold: float = 30.0
    ):
        """
        Initialize HSV-based shadow removal.

        Args:
            hue_threshold: Maximum hue difference to consider as shadow (0-180)
            value_ratio_min: Minimum brightness ratio (shadow must be darker but not too much)
            value_ratio_max: Maximum brightness ratio (shadow must be darker than background)
            saturation_diff_threshold: Maximum saturation difference (0-255)
        """
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
        """Remove shadows using HSV background comparison."""
        # Get background frame
        bg_frame = bg_model.background_image()

        # Convert both to HSV
        hsv_current = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv_bg = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2HSV).astype(np.float32)

        h_curr, s_curr, v_curr = cv2.split(hsv_current)
        h_bg, s_bg, v_bg = cv2.split(hsv_bg)

        # Shadow characteristics:
        # 1. Similar hue (accounting for circular hue space 0-180)
        hue_diff = np.minimum(
            np.abs(h_curr - h_bg),
            180 - np.abs(h_curr - h_bg)
        )

        # 2. Darker than background (but not too much darker - actual objects differ more)
        value_ratio = v_curr / (v_bg + 1e-6)

        # 3. Similar saturation
        sat_diff = np.abs(s_curr - s_bg)

        # Identify shadow pixels
        shadow_mask = (
            (hue_diff < self.hue_threshold) &
            (value_ratio > self.value_ratio_min) &
            (value_ratio < self.value_ratio_max) &
            (sat_diff < self.saturation_diff_threshold) &
            (mask > 0)
        )

        # Remove shadows from mask
        cleaned_mask = mask.copy()
        cleaned_mask[shadow_mask] = 0

        return cleaned_mask


class ChromaticityBackgroundComparison(ShadowRemover):
    """
    Remove shadows using chromaticity comparison.

    Chromaticity (normalized color) is invariant to illumination changes.
    Shadows have similar chromaticity to the background but different intensity,
    while actual objects have different chromaticity.
    """

    def __init__(
        self,
        chromaticity_threshold: float = 0.1,
        intensity_ratio_min: float = 0.4,
        intensity_ratio_max: float = 0.9
    ):
        """
        Initialize chromaticity-based shadow removal.

        Args:
            chromaticity_threshold: Maximum chromaticity distance to consider as shadow
            intensity_ratio_min: Minimum intensity ratio (shadow darker but not too much)
            intensity_ratio_max: Maximum intensity ratio (shadow must be darker)
        """
        self.chromaticity_threshold = chromaticity_threshold
        self.intensity_ratio_min = intensity_ratio_min
        self.intensity_ratio_max = intensity_ratio_max

    def remove_shadows(
        self,
        mask: np.ndarray,
        frame: np.ndarray,
        bg_model: BackgroundModel
    ) -> np.ndarray:
        """Remove shadows using chromaticity comparison."""
        bg_frame = bg_model.background_image()

        # Convert to float and avoid division by zero
        frame_float = frame.astype(np.float32) + 1e-6
        bg_float = bg_frame.astype(np.float32) + 1e-6

        # Compute chromaticity: normalize each channel by total intensity
        # c_i = I_i / (I_r + I_g + I_b)
        intensity_curr = frame_float.sum(axis=2, keepdims=True)
        intensity_bg = bg_float.sum(axis=2, keepdims=True)

        chrom_curr = frame_float / intensity_curr
        chrom_bg = bg_float / intensity_bg

        # Compute chromaticity distance (Euclidean distance in RGB chromaticity space)
        chrom_dist = np.linalg.norm(chrom_curr - chrom_bg, axis=2)

        # Compute intensity ratio
        intensity_ratio = (intensity_curr / intensity_bg).squeeze()

        # Shadow criteria: similar chromaticity, lower intensity
        shadow_mask = (
            (chrom_dist < self.chromaticity_threshold) &
            (intensity_ratio < self.intensity_ratio_max) &
            (intensity_ratio > self.intensity_ratio_min) &
            (mask > 0)
        )

        # Remove shadows from mask
        cleaned_mask = mask.copy()
        cleaned_mask[shadow_mask] = 0

        return cleaned_mask
