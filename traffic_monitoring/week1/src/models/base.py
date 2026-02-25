from typing import Generator, Protocol, Tuple

import numpy as np

from src.video_source import VideoSource


class BackgroundModel(Protocol):
    def fit_from_source(self, source: VideoSource):
        pass
    
    def predict_from_source(self, source: VideoSource) -> Generator[np.ndarray, None, None]:
        pass
    
    def predict_from_source_with_frame(self, source: VideoSource) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        pass
    
    def background_image(self) -> np.ndarray:
        pass
