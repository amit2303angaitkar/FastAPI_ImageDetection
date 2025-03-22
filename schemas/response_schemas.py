from pydantic import BaseModel
from typing import List, Dict

class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]

class ImageResponse(BaseModel):
    message: str
    detections: List[DetectionResult]
