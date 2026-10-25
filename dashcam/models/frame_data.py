"""
Frame data container
"""
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class FrameData:
    """Container for frame data"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    gps_data: Dict[str, Any]
