from fastapi import HTTPException

class ModelNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(status_code=404, detail="Model not found")

class InvalidImageException(HTTPException):
    def __init__(self):
        super().__init__(status_code=400, detail="Invalid image format")
