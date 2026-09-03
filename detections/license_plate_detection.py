import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR

class LicensePlateDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def detect_frames(self, frames):
        license_plate_detections = []
        license_plate_texts = []
        for frame in frames:
            bbox_list, text_list = self.detect_frame(frame)
            license_plate_detections.append(bbox_list)
            license_plate_texts.append(text_list)
        return license_plate_detections, license_plate_texts

    def detect_frame(self, frame):
        # 1. UPDATE: predict() ki jagah track() + BoT-SORT (Kalman Filter) lagaya gaya hai
        results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
        
        id_name_dict = results.names
        license_plate_list = []
        license_plate_texts = []

        # Ensure boxes exist to prevent errors
        if results.boxes is not None:
            for box in results.boxes:
                result = box.xyxy.tolist()[0]
                cls_id = int(box.cls.tolist()[0])
                cls_name = id_name_dict[cls_id]

                if cls_name == "License_Plate":
                    license_plate_list.append(result)
                    x1, y1, x2, y2 = map(int, result)
                    cropped_plate = frame[y1:y2, x1:x2]
                    
                    # 2. UPDATE: Crash rokne ke liye check (agar crop size 0 ho jaye edge par)
                    if cropped_plate.size > 0:
                        # Preprocessing
                        gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, dsize=None, fx=2, fy=2)
                        cropped_plate = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                        
                        # RUN OCR
                        ocr_result = self.ocr.ocr(cropped_plate)
                        
                        # 3. UPDATE: Multi-line plates ke liye " ".join(texts) add kiya gaya hai
                        text = "N/A"
                        if ocr_result:
                            try:
                                res = ocr_result[0]
                                if isinstance(res, dict) and 'rec_texts' in res:
                                    texts = res['rec_texts']
                                    text = " ".join(texts) if texts else "N/A"
                                elif hasattr(res, 'get') and res.get('rec_texts'):
                                    texts = res.get('rec_texts')
                                    text = " ".join(texts) if texts else "N/A"
                                else:
                                    # Fallback for legacy format
                                    lines = [item[1][0] for item in ocr_result if item and len(item) > 1]
                                    text = " ".join(lines) if lines else "N/A"
                            except (IndexError, TypeError, KeyError):
                                text = "N/A"
                                
                        license_plate_texts.append(text)
                    else:
                        license_plate_texts.append("N/A")
                        
        return license_plate_list, license_plate_texts

    def draw_bboxes(self, video_frames, license_plate_detections, license_plate_text):
        output_video_frames = []
        for frame, plate_list, text_list in zip(video_frames, license_plate_detections, license_plate_text):
            for bbox, text in zip(plate_list, text_list):
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, text=text, org=(int(x1), int(y1-10)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0), thickness=2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            output_video_frames.append(frame)
        return output_video_frames