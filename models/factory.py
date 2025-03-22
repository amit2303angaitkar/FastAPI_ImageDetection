from models.object_detector import ObjectDetector

class ModelFactory:
    @staticmethod
    def get_model(model_type):
        if model_type == "yolo":
            return ObjectDetector()
        else:
            raise ValueError("Unkown Model type") 

from models.factory import ModelFactory

class ObjectDetectionService:
    def __init__(self,model_type="yolo"):
        self.detector =ModelFactory.get_model(model_type)
        
