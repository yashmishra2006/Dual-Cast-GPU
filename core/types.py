# core/types.py
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class FrameHeader:
    frame_id: int
    timestamp: str
    shape: List[int]

@dataclass
class DetectionResult:
    frame_id: int
    phase: str
    label: str
    confidence: float
    metadata: Dict