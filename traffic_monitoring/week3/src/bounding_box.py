from typing import NamedTuple, Optional

class BoundingBox(NamedTuple):
    top: float
    bottom: float
    left: float
    right: float
    confidence: Optional[float]