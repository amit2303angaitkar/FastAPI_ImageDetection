import cv2

def draw_bounding_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} {confidence:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 2)
    return image
