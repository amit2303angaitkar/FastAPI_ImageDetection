import cv2
from models.object_detector import ObjectDetector
from utils.image_utils import draw_bounding_boxes



class ObjectDetectionService:
    def __init__(self):
        self.detector =ObjectDetector()

    def detect_object(self,image):
        detections =self.detector.detect(image)
        processed_image= draw_bounding_boxes(image,detections)
        return processed_image,detections


# def process_image(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply edge detection
#     edges = cv2.Canny(gray, 100, 200)
    
#     return {"edges_detected": True}
