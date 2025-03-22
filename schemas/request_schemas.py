from pydantic import BaseModel

class ImageRequest(BaseModel):
    model_type: str = "yolo"  # Allow flexibility for different models
