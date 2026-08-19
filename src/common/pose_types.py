from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class TrackedPlayerPose:
    track_id: int
    bbox: np.ndarray
    keypoints: np.ndarray
    confidence: float


@dataclass
class TrackedFramePose:
    frame_id: int
    players: List[TrackedPlayerPose]