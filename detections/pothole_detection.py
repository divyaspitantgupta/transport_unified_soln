import cv2
import numpy as np
from ultralytics import YOLO

class PotholeDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Blue color for pothole polygons (BGR format)
        self.color = (255, 0, 0)

    def detect_frames(self, frames):
        pothole_detections = []
        for frame in frames:
            mask_list = self.detect_frame(frame)
            pothole_detections.append(mask_list)
        return pothole_detections

    def detect_frame(self, frame):
        # Tracking is not usually needed for static potholes, so predict() is faster
        results = self.model.predict(frame, conf=0.25, verbose=False)[0]
        mask_list = []
        
        # Check if any masks (polygons) are detected
        if results.masks is not None:
            for mask in results.masks.xy:
                # Convert float coordinates to integer points for OpenCV
                points = np.int32([mask])
                mask_list.append(points)
                
        return mask_list

    def draw_polygons(self, video_frames, pothole_detections):
        output_video_frames = []
        for frame, mask_list in zip(video_frames, pothole_detections):
            for points in mask_list:
                # Draw the blue polygon outline
                cv2.polylines(frame, points, isClosed=True, color=self.color, thickness=2)
                
                # Add label text near the first point of the polygon
                if len(points[0]) > 0:
                    px, py = points[0][0]
                    cv2.putText(
                        frame, 
                        "Pothole", 
                        (px, py - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        self.color, 
                        2
                    )
            output_video_frames.append(frame)
        return output_video_frames