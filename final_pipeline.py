import cv2
import csv
import os
import math
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 0. PADDLE OCR CRASH FIX (MKLDNN DISABLED)
# ==========================================
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'

# ==========================================
# 1. HELPER FUNCTIONS (Deduplication & PIL Text)
# ==========================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def is_duplicate_defect(new_gps, historical_gps_list):
    """ Agar 5 meter ke andar pehle se koi defect hai, toh usko duplicate manenge """
    for hist_gps in historical_gps_list:
        if calculate_haversine_distance(hist_gps['lat'], hist_gps['lon'], new_gps['lat'], new_gps['lon']) < 5.0:
            return True
    return False

def put_professional_text(frame, text, pos, text_color=(255, 255, 255), font_size=16):
    """ OpenCV frame par PIL ka use karke clean professional text draw karne ke liye """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        
    draw.text(pos, text, font=font, fill=text_color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ==========================================
# 2. MAIN PIPELINE CLASS
# ==========================================
class SmartTrafficPipeline:
    def __init__(self, vehicle_model_path, plate_model_path, pothole_model_path):
        # 1. Load Models
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.pothole_model = YOLO(pothole_model_path)
        
        # MKLDNN Disabled in OCR initialization
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
        
        # 2. Tracking & Deduplication Variables
        self.previous_areas = {}           # {track_id: initial_area}
        self.logged_track_ids = set()      # Vehicles ke duplicate challan rokne ke liye
        self.logged_potholes = []          # List of dictionaries: [{'lat': x, 'lon': y}, ...]
        self.logged_pothole_ids = set()    # Pothole IDs track karne ke liye
        
        # 3. Initialize CSV Files
        self.vehicle_csv = "speed_violations.csv"
        self.pothole_csv = "pothole_logs.csv"
        
        with open(self.vehicle_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Track_ID", "License_Plate", "Status"])
            
        with open(self.pothole_csv, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Pothole_ID", "Latitude", "Longitude", "Status"])

    def process_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video stream from {video_source}")
            return

        # Video properties for saving output file
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30 # Fallback agar webcam ya streaming me FPS na mile

        # Initialize Video Writer (Saved as output_dashboard.mp4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output_dashboard.mp4', fourcc, fps, (frame_width, frame_height))

        print("Live detection started & recording enabled. Press 'q' to exit.")
        
        cv2.namedWindow("Smart Traffic Dashboard", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart Traffic Dashboard", 1080, 720)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # ==========================================
            # GPS DATA
            # ==========================================
            current_lat = 23.2599  # Dummy latitude (Bhopal)
            current_lon = 77.4126  # Dummy longitude (Bhopal)

            # ==========================================
            # 3. Pothole Tracking, Segmentation & Deduplication
            # ==========================================
            pothole_results = self.pothole_model.track(
                frame, 
                conf=0.6, 
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False
            )[0]
            
            if pothole_results.masks is not None and pothole_results.boxes.id is not None:
                masks = pothole_results.masks.xy
                track_ids = pothole_results.boxes.id.cpu().numpy()
                
                for mask, track_id in zip(masks, track_ids):
                    points = np.int32([mask])
                    p_id = int(track_id)
                    
                    # Pothole Fill Overlay Logic
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [points], (255, 0, 0)) # Solid blue fill
                    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame) # 40% opacity blend
                    cv2.polylines(frame, [points], isClosed=True, color=(255, 0, 0), thickness=3)
                    
                    if len(points[0]) > 0:
                        px, py = points[0][0]
                        text_pos = (px, max(0, py - 22))
                        
                        if p_id in self.logged_pothole_ids:
                            frame = put_professional_text(frame, f"Pothole ID: {p_id} (Logged)", text_pos, text_color=(255, 255, 255), font_size=16)
                        else:
                            new_gps = {'lat': current_lat, 'lon': current_lon}
                            if not is_duplicate_defect(new_gps, self.logged_potholes):
                                self.logged_potholes.append(new_gps)
                                self.logged_pothole_ids.add(p_id)
                                self.log_pothole_to_csv(p_id, new_gps['lat'], new_gps['lon'])
                                frame = put_professional_text(frame, f"Pothole ID: {p_id}", text_pos, text_color=(255, 255, 255), font_size=16)
                            else:
                                self.logged_pothole_ids.add(p_id)
                                frame = put_professional_text(frame, f"Pothole ID: {p_id} (Logged)", text_pos, text_color=(255, 255, 255), font_size=16)

            # ==========================================
            # 4. Vehicle Tracking & Overspeeding
            # ==========================================
            vehicle_results = self.vehicle_model.track(
                frame, 
                conf=0.5,
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False
            )[0]
            
            if vehicle_results.boxes.id is not None:
                boxes = vehicle_results.boxes.xyxy.cpu().numpy()
                track_ids = vehicle_results.boxes.id.cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    area = (x2 - x1) * (y2 - y1)
                    text_pos = (x1, max(0, y1 - 25))
                    
                    # Prevent duplicate OCR processing for already logged vehicles
                    if track_id in self.logged_track_ids:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        frame = put_professional_text(frame, f"ID:{int(track_id)} Logged", text_pos, text_color=(255, 255, 255), font_size=16)
                        continue

                    is_overspeeding = False
                    
                    if track_id not in self.previous_areas:
                        self.previous_areas[track_id] = area
                    else:
                        prev_area = self.previous_areas[track_id]
                        area_growth_rate = (area - prev_area) / prev_area
                        
                        if area_growth_rate > 0.15: 
                            is_overspeeding = True
                    
                    # ==========================================
                    # 5. License Plate & OCR (Only on Overspeed)
                    # ==========================================
                    if is_overspeeding:
                        vehicle_crop = frame[y1:y2, x1:x2]
                        plate_detected = False
                        
                        if vehicle_crop.size > 0:
                            plate_results = self.plate_model.predict(vehicle_crop, conf=0.6, verbose=False)[0]
                            
                            for p_box in plate_results.boxes.xyxy.tolist():
                                px1, py1, px2, py2 = map(int, p_box)
                                cropped_plate = vehicle_crop[py1:py2, px1:px2]
                                
                                if cropped_plate.size > 0:
                                    gray = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)
                                    resized = cv2.resize(gray, dsize=None, fx=2, fy=2)
                                    processed_plate = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                                    
                                    ocr_result = self.ocr.ocr(processed_plate)
                                    plate_text = self._extract_text(ocr_result)
                                    
                                    if plate_text != "N/A":
                                        plate_detected = True
                                        self.log_violation(track_id, plate_text)
                                        self.logged_track_ids.add(track_id)
                                        
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                        frame = put_professional_text(frame, f"ID:{int(track_id)} OVERSPEED: {plate_text}", text_pos, text_color=(255, 255, 255), font_size=16)
                        
                        if not plate_detected:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                            frame = put_professional_text(frame, f"ID:{int(track_id)} Speeding (No Plate)", text_pos, text_color=(255, 255, 255), font_size=16)
                            
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        frame = put_professional_text(frame, f"ID: {int(track_id)}", text_pos, text_color=(255, 255, 255), font_size=16)

            # Write the fully annotated frame to output file
            out.write(frame)

            cv2.imshow("Smart Traffic Dashboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("Video recording saved successfully as 'output_dashboard.mp4'!")

    def _extract_text(self, ocr_result):
        if not ocr_result:
            return "N/A"
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
    
    pipeline.process_video(r"C:\Users\p\Downloads\testvideo2.mp4")

if __name__ == "__main__":
    main()