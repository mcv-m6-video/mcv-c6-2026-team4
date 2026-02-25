"""
OpenCV-based background subtraction models for comparison with custom implementations.

References:
- KaewTraKulPong et al. "An improved adaptive background mixture model for real-time
  tracking with shadow detection." Video-Based Surveillance Systems, 2002.
  Implementation: BackgroundSubtractorMOG

- Z. Zivkovic et al. "Efficient adaptive density estimation per image pixel for the
  task of background subtraction." Pattern Recognition Letters, 2005.
  Implementation: BackgroundSubtractorMOG2

- L. Guo, et al. "Background subtraction using local svd binary pattern." CVPRW, 2016.
  Implementation: BackgroundSubtractorLSBP
"""

import numpy as np
import cv2

from src.video_source import VideoSource


class OpenCVBackgroundSubtractor:
    """Base wrapper for OpenCV background subtraction algorithms."""

    def __init__(self, bg_subtractor):
        """
        Args:
            bg_subtractor: An OpenCV background subtractor instance (e.g., cv2.createBackgroundSubtractorMOG2())
        """
        self.bg_subtractor = bg_subtractor

    def fit_from_source(self, source: VideoSource):
        """
        Train the background model using frames from the source.

        Args:
            source: VideoSource containing training frames

        Returns:
            self
        """
        for frame in iter(source):
            # Apply with learning rate to train the model
            self.bg_subtractor.apply(frame, learningRate=-1)  # -1 uses default learning rate
        return self

    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        """
        Predict foreground mask for a single frame.

        Args:
            frame: BGR image

        Returns:
            Binary mask where True indicates foreground
        """
        mask = self.bg_subtractor.apply(frame, learningRate=0)  # 0 = no learning during inference
        return mask > 0  # Convert to boolean mask

    def predict_from_source(self, source: VideoSource):
        """
        Generate foreground masks for all frames in the source.

        Args:
            source: VideoSource containing frames to process

        Yields:
            Binary mask for each frame
        """
        for frame in iter(source):
            yield self._predict_single(frame)


class MOGBackgroundSubtractor(OpenCVBackgroundSubtractor):
    """
    Mixture of Gaussians background subtraction (KaewTraKulPong et al., 2002).

    Uses an adaptive mixture of Gaussians to model each background pixel.
    Supports shadow detection.

    Note: Requires opencv-contrib-python. May not be available in all OpenCV versions.
    """

    def __init__(self, history: int = 200, n_mixtures: int = 5, background_ratio: float = 0.7,
                 noise_sigma: float = 0):
        """
        Args:
            history: Length of the history
            n_mixtures: Number of Gaussian mixtures
            background_ratio: Background ratio
            noise_sigma: Noise strength (standard deviation). 0 means automatic value.
        """
        try:
            # Try the old API first (older opencv-contrib versions)
            bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
                history=history,
                nmixtures=n_mixtures,
                backgroundRatio=background_ratio,
                noiseSigma=noise_sigma
            )
        except AttributeError:
            try:
                # Try newer API
                bg_subtractor = cv2.createBackgroundSubtractorMOG(
                    history=history,
                    nmixtures=n_mixtures,
                    backgroundRatio=background_ratio,
                    noiseSigma=noise_sigma
                )
            except AttributeError:
                raise AttributeError(
                    "BackgroundSubtractorMOG not found. Please install opencv-contrib-python: "
                    "pip install opencv-contrib-python"
                )
        super().__init__(bg_subtractor)


class MOG2BackgroundSubtractor(OpenCVBackgroundSubtractor):
    """
    Improved Gaussian Mixture Model background subtraction (Zivkovic, 2005).

    Uses efficient adaptive density estimation per pixel. Automatically selects
    the appropriate number of Gaussian distributions for each pixel.
    Supports shadow detection.
    """

    def __init__(self, history: int = 500, var_threshold: float = 16, detect_shadows: bool = True):
        """
        Args:
            history: Length of the history
            var_threshold: Threshold on the squared Mahalanobis distance between the pixel
                          and the model to decide whether it is well described by the background
            detect_shadows: If True, the algorithm will detect shadows and mark them (value 127 in mask)
        """
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        super().__init__(bg_subtractor)

    def _predict_single(self, frame: np.ndarray) -> np.ndarray:
        """Override to handle shadow pixels (127) as background."""
        mask = self.bg_subtractor.apply(frame, learningRate=0)
        # Treat shadows (127) as background (0), only 255 is foreground
        return mask == 255


class LSBPBackgroundSubtractor(OpenCVBackgroundSubtractor):
    """
    Local SVD Binary Pattern background subtraction (Guo et al., 2016).

    Uses local SVD binary patterns for robust background modeling.
    """

    def __init__(self, motion_compensation: int = 0,
                 n_samples: int = 20, lsbp_radius: int = 16, tlsbp_threshold: float = 2.0,
                 t_inc: float = 1.0, t_dec: float = 0.05, r_scale: float = 10.0,
                 r_inc_dec: float = 0.005, noise_removal_threshold_fac_bg: float = 0.0004,
                 noise_removal_threshold_fac_fg: float = 0.0008, lsbp_threshold: int = 8,
                 min_count: int = 2):
        """
        Args:
            motion_compensation: Type of camera motion compensation (0 = NONE, 1 = LK)
            n_samples: Number of samples to maintain
            lsbp_radius: LSBP descriptor radius
            tlsbp_threshold: Threshold for T-LSBP
            t_inc: Increase rate for threshold
            t_dec: Decrease rate for threshold
            r_scale: Scale parameter for r
            r_inc_dec: Increase/decrease parameter for r
            noise_removal_threshold_fac_bg: Noise removal threshold factor for background
            noise_removal_threshold_fac_fg: Noise removal threshold factor for foreground
            lsbp_threshold: LSBP threshold
            min_count: Minimal number of matches
        """
        bg_subtractor = cv2.bgsegm.createBackgroundSubtractorLSBP(
            mc=motion_compensation,
            nSamples=n_samples,
            LSBPRadius=lsbp_radius,
            Tlower=tlsbp_threshold,
            Tupper=tlsbp_threshold * 2,
            Tinc=t_inc,
            Tdec=t_dec,
            Rscale=r_scale,
            Rincdec=r_inc_dec,
            noiseRemovalThresholdFacBG=noise_removal_threshold_fac_bg,
            noiseRemovalThresholdFacFG=noise_removal_threshold_fac_fg,
            LSBPthreshold=lsbp_threshold,
            minCount=min_count
        )
        super().__init__(bg_subtractor)
