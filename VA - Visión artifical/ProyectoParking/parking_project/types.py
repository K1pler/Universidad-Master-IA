from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    image_path: Path
    xml_path: Path
    view: str
    weather: str
    capture_date: str
    split: str = ""


@dataclass(frozen=True)
class ParkingSpace:
    space_id: str
    polygon: np.ndarray
    occupied: bool


@dataclass
class SlotExample:
    record: FrameRecord
    space: ParkingSpace
    crop: np.ndarray


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

