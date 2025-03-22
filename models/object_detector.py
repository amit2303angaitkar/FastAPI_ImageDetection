from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, image):
        results = self.model(image)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                detections.append({
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })
        return detections
