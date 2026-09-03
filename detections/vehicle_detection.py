import cv2
import pickle
from ultralytics import YOLO

class VehicleDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # 'Fixed Obstacle' ko hata diya gaya hai taaki sirf vehicles track hon
        self.vehicle_classes = [
            "Car", 
            "Truck", 
            "Bus", 
            "Tractor", 
            "Modified", 
            "Three Wheeler", 
            "Two Wheeler", 
            "Vikram"
        ]

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        vehicle_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                vehicle_detections = pickle.load(f)
            return vehicle_detections
            
        for frame in frames:
            vehicle_list = self.detect_frame(frame)
            vehicle_detections.append(vehicle_list)
            
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(vehicle_detections, f)
        return vehicle_detections

    def detect_frame(self, frame):
        # BoT-SORT tracker for smooth vehicle tracking
        results = self.model.track(frame, iou=0.1, conf=0.3, persist=True, tracker="botsort.yaml", verbose=False)[0]
        id_name_dict = results.names
        vehicle_list = []
        
        if results.boxes is not None:
            for box in results.boxes:
                result = box.xyxy.tolist()[0]
                cls_id = int(box.cls.tolist()[0])
                cls_name = id_name_dict[cls_id]
                
                # Case-insensitive check lagaya hai
                if cls_name.lower() in [v.lower() for v in self.vehicle_classes]:
                    vehicle_list.append((result, cls_name))
                    
        return vehicle_list

    def draw_bboxes(self, video_frames, vehicle_detections):
        output_video_frames = []
        for frame, vehicle_list in zip(video_frames, vehicle_detections):
            for item in vehicle_list:
                bbox, cls_name = item
                x1, y1, x2, y2 = map(int, bbox)
                
                # Jo class detect hogi (jaise "Two Wheeler" ya "Vikram"), wahi print hogi
                cv2.putText(frame, text=cls_name, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0), thickness=2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                
            output_video_frames.append(frame)
        return output_video_frames