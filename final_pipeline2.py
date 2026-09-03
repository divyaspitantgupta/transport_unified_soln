import cv2
import csv
import os
import math
import numpy as np
import re
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 0. PADDLE OCR CRASH FIX
# ==========================================
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def is_duplicate_defect(new_gps, historical_gps_list):
    for hist_gps in historical_gps_list:
        if calculate_haversine_distance(hist_gps['lat'], hist_gps['lon'], new_gps['lat'], new_gps['lon']) < 5.0:
            return True
    return False

# ==========================================
# 2. MAIN PIPELINE CLASS
# ==========================================
class SmartTrafficPipeline:
    def __init__(self, vehicle_model_path, plate_model_path, pothole_model_path):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.pothole_model = YOLO(pothole_model_path)
        
        self.ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False)
        
        self.vehicle_history = {} 
        self.logged_track_ids = set()
        self.logged_potholes = []
        self.logged_pothole_ids = set()
        
        self.vehicle_csv = "speed_violations.csv"
        self.pothole_csv = "pothole_logs.csv"
        
        with open(self.vehicle_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Track_ID", "License_Plate", "Status"])
            
        with open(self.pothole_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Pothole_ID", "Latitude", "Longitude", "Status"])

    # REGEX VALIDATION FUNCTION
    def is_valid_plate(self, text):
        if not text or str(text).strip() == "N/A":
            return False
        # Remove all special characters, spaces, and make uppercase
        clean_text = re.sub(r'[^A-Z0-9]', '', str(text).upper())
        # Pattern: 2 Letters, 1-2 Numbers, 0-3 Letters, 4 Numbers (Matches standard Indian formats)
        pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$'
        return bool(re.match(pattern, clean_text))

    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video stream from {video_source}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_dashboard.mp4', fourcc, fps, (frame_width, frame_height))

        print("Smart Live Regex Validation Enabled. Press 'q' to exit.")
        cv2.namedWindow("Smart Traffic Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Traffic Dashboard", 1080, 720)
        
        frame_count = 0
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except IOError:
            font = ImageFont.load_default()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
                
            frame_count += 1
            if frame_count % 2 != 0:
                out.write(frame)
                continue

            current_lat = 23.2599
            current_lon = 77.4126
            texts_to_draw = []

            # ==========================================
            # 3. Pothole Tracking
            # ==========================================
            pothole_results = self.pothole_model.track(frame, conf=0.45, imgsz=1024, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if pothole_results.masks is not None and pothole_results.boxes.id is not None:
                masks = pothole_results.masks.xy
                track_ids = pothole_results.boxes.id.cpu().numpy()
                for mask, track_id in zip(masks, track_ids):
                    points = np.int32([mask])
                    p_id = int(track_id)
                    
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points], (255, 0, 0))
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
                    cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=3)
                    
                    if len(points[0]) > 0:
                        px, py = points[0][0]
                        text_str = f"Pothole: {p_id} (Logged)" if p_id in self.logged_pothole_ids else f"Pothole: {p_id}"
                        texts_to_draw.append((text_str, (px, max(0, py - 25)), (0, 0, 255)))
                        
                        if p_id not in self.logged_pothole_ids:
                            new_gps = {'lat': current_lat, 'lon': current_lon}
                            if not is_duplicate_defect(new_gps, self.logged_potholes):
                                self.logged_potholes.append(new_gps)
                                self.logged_pothole_ids.add(p_id)
                                self.log_pothole_to_csv(p_id, new_gps['lat'], new_gps['lon'])
                            else:
                                self.logged_pothole_ids.add(p_id)

            # ==========================================
            # 4. Vehicle Tracking & Strict Plate Validation
            # ==========================================
            vehicle_results = self.vehicle_model.track(frame, conf=0.35, imgsz=1024, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if vehicle_results.boxes.id is not None:
                boxes = vehicle_results.boxes.xyxy.cpu().numpy()
                track_ids = vehicle_results.boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    
                    if track_id in self.logged_track_ids:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        texts_to_draw.append((f"ID:{int(track_id)} Logged", (x1, max(0, y1 - 25)), (255, 0, 0)))
                        continue

                    is_overspeeding = False
                    
                    if track_id not in self.vehicle_history:
                        self.vehicle_history[track_id] = {'initial_area': area, 'frames': 1}
                    else:
                        self.vehicle_history[track_id]['frames'] += 1
                        if self.vehicle_history[track_id]['frames'] >= 5:
                            prev_area = self.vehicle_history[track_id]['initial_area']
                            if (area - prev_area) / prev_area > 0.15:
                                is_overspeeding = True
                    
                    if is_overspeeding and y2 > frame_height * 0.4:
                        vehicle_crop = frame[y1:y2, x1:x2]
                        plate_detected = False
                        
                        if vehicle_crop.size > 0:
                            plate_results = self.plate_model.predict(vehicle_crop, conf=0.6, imgsz=1280, verbose=False)[0]
                            for p_box in plate_results.boxes.xyxy.tolist():
                                px1, py1, px2, py2 = map(int, p_box)
                                cropped_plate = vehicle_crop[py1:py2, px1:px2]
                                if cropped_plate.size > 0:
                                    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                                    ocr_result = self.ocr.ocr(cv2.cvtColor(cv2.resize(gray, None, fx=1.5, fy=1.5), cv2.COLOR_GRAY2BGR))
                                    plate_text = self._extract_text(ocr_result)
                                    
                                    if plate_text != "N/A":
                                        clean_text = re.sub(r'[^A-Z0-9]', '', str(plate_text).upper())
                                        
                                        # Yahan LIVE regex check hoga!
                                        if self.is_valid_plate(clean_text):
                                            plate_detected = True
                                            self.log_violation(track_id, clean_text)
                                            # Sirf Regex pass hone par hi gaadi lock hogi, warna OCR try karta rahega
                                            self.logged_track_ids.add(track_id) 
                                            
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                            texts_to_draw.append((f"ID:{int(track_id)} OVERSPEED: {clean_text}", (x1, max(0, y1 - 25)), (255, 0, 0)))
                                            break
                        
                        if not plate_detected:
                            # Agar regex fail hua (jaise "399" padha) ya plate nahi mili toh orange me search jari rahega
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                            texts_to_draw.append((f"ID:{int(track_id)} Scanning Plate...", (x1, max(0, y1 - 25)), (255, 165, 0)))
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        texts_to_draw.append((f"ID: {int(track_id)}", (x1, max(0, y1 - 25)), (0, 200, 0)))

            # ==========================================
            # 5. Draw Labels with Solid Background
            # ==========================================
            if texts_to_draw:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_img)
                for text_str, pos, bg_color in texts_to_draw:
                    bbox = draw.textbbox(pos, text_str, font=font)
                    pad_x, pad_y = 4, 2
                    rect_bbox = [bbox[0] - pad_x, bbox[1] - pad_y, bbox[2] + pad_x, bbox[3] + pad_y]
                    draw.rectangle(rect_bbox, fill=bg_color)
                    draw.text(pos, text_str, font=font, fill=(255, 255, 255))
                frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            out.write(frame)
            cv2.imshow("Smart Traffic Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Done! Saved as 'output_dashboard.mp4'.")

    def _extract_text(self, ocr_result):
        if not ocr_result: return "N/A"
        try:
            res = ocr_result[0]
            if isinstance(res, dict) and 'rec_texts' in res:
                return " ".join(res['rec_texts']) if res['rec_texts'] else "N/A"
            elif hasattr(res, 'get') and res.get('rec_texts'):
                return " ".join(res.get('rec_texts')) if res.get('rec_texts') else "N/A"
            else:
                lines = [item[1][0] for item in ocr_result if item and len(item) > 1]
                return " ".join(lines) if lines else "N/A"
        except Exception:
            return "N/A"

    def log_violation(self, track_id, plate_text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.vehicle_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, int(track_id), plate_text, "Overspeeding"])
        print(f"[OVERSPEED] Time: {timestamp} | ID: {track_id} | Plate: {plate_text}")

    def log_pothole_to_csv(self, p_id, lat, lon):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.pothole_csv, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, p_id, lat, lon, "Detected"])
        print(f"[POTHOLE] Logged at Time: {timestamp} | ID: {p_id} | Lat: {lat} | Lon: {lon}")

def main():
    pipeline = SmartTrafficPipeline(
        vehicle_model_path=r"C:\Users\p\road-defect-pipeline\models\bestvehicle.pt", 
        plate_model_path=r"C:\Users\p\road-defect-pipeline\models\bestocr.pt",
        pothole_model_path=r"C:\Users\p\road-defect-pipeline\models\bestpothole.pt"
    )
    pipeline.process_video(0)

if __name__ == "__main__":
    main()