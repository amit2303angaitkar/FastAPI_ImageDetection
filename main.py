from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from schemas.request_schemas import ImageRequest
from services.image_service import ObjectDetectionService
import cv2
import numpy as np
import shutil
import os
from ultralytics import YOLO  # Import YOLOv8 model

app = FastAPI(title="AI-Image API")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

detection_service = ObjectDetectionService()

# Load the YOLOv8 object detection model
model = YOLO("yolov8s.pt")  

@app.get("/")
def home():
    return {"message": "Image API "}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    try:
        #  Validate image format
        if not file.filename.lower().endswith(("png", "jpg", "jpeg")):
            raise HTTPException(status_code=400, detail="Invalid file format. Only PNG, JPG, and JPEG are allowed.")

        #  Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the saved image using OpenCV 
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image. Please check the file.")

        # Run object detection
        results = model(file_path)  # Detect objects using YOLO
        detections = []
        
        for result in results:
            for box in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = box
                detections.append({
                    "class": result.names[int(class_id)],  # Get object name
                    "confidence": round(score, 2),
                    "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
                })

        #  Return the detected objects
        return JSONResponse(content={
            "message": "Image analyzed successfully",
            "filename": file.filename,
            "detections": detections
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)