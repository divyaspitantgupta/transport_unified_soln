import cv2
import math
import os
import torch
from ultralytics import YOLO

# TERA APNA ENGINE IMPORT HO RAHA HAI
from ocr_engine import extract_indian_license_plate, log_violation

# ==========================================
# 1. HELPER FUNCTIONS (Deduplication)
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

# ==========================================
# 2. MAIN PIPELINE EXECUTION (DUAL-MODEL)
# ==========================================
def run_pipeline():
    print("Initializing DUAL-AI Smart City Edge Pipeline (Defects + Vehicles)...")

    torch.set_num_threads(4) 
    
    # ENGINE 1: Tera Custom Defect Model (For Masks & Potholes)
    print("Loading Road Inspector Model (bestnew.pt)...")
    model_defect = YOLO("bestnew.pt")
    
    # ENGINE 2: Default Traffic Model (For Vehicles)
    print("Loading Traffic Police Model (yolo26n.pt)...")
    model_traffic = YOLO("yolo26n.pt") 
    
    video_path = r"C:\Users\p\Downloads\12947767_3840_2160_25fps.mp4"
    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    
    # Yahan original video ki width, height aur FPS nikal kar writer setup karenge
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_processed_video.mp4', fourcc, fps, (width, height))
    
    historical_defects = []
    plate_memory = {}  
    speed_memory = {}

    # RELATIVE SPEED CONSTANTS (Dashcam Overtake Logic)
    FOCAL_LENGTH = 800     # Tweak this based on your camera
    REAL_CAR_WIDTH = 1.8   # Average car width in meters

    # TERE PICKLE FILE WALI CLASSES (0, 1, 2 as defects)
    defect_names = {0: "Delaminator", 1: "Pothole", 2: "Chuck-hole"}

    frame_count = 0
    cv2.namedWindow("Smart City Dual-Pipeline", cv2.WINDOW_NORMAL)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1

        # =================================================
        # AI ENGINE 1: ROAD DEFECTS TRACKING (bestnew.pt)
        # =================================================
        # Maine save=True hata diya kyunki hum apna VideoWriter use kar rahe hain
        defect_results = model_defect.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.7, verbose=False)
        
        # Pehle defects ko frame par plot kar lo (Masks aur labels draw ho jayenge)
        annotated_frame = defect_results[0].plot()
        
        # Defect Logging Logic
        if defect_results[0].boxes is not None:
            for box in defect_results[0].boxes:
                cls_id = int(box.cls[0])
                if cls_id in defect_names:
                    defect_type = defect_names[cls_id]
                    dummy_gps = {'lat': 22.7196, 'lon': 75.8577} 
                    
                    if not is_duplicate_defect(dummy_gps, historical_defects):
                        historical_defects.append(dummy_gps)
                        print(f"[NEW HAZARD] {defect_type} Logged at {dummy_gps}")

        # =================================================
        # AI ENGINE 2: TRAFFIC, SPEED & OCR (yolo26n.pt)
        # =================================================
        # COCO Classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        traffic_results = model_traffic.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.5, verbose=False)
        
        if traffic_results[0].boxes is not None and traffic_results[0].boxes.id is not None:
            for box in traffic_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id[0])
                box_width = x2 - x1
                
                # Draw vehicle box on the same annotated frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                is_speeding = False
                relative_speed_kmph = 0
                
                # --- RELATIVE SPEED LOGIC ---
                if track_id not in speed_memory:
                    speed_memory[track_id] = {'start_frame': frame_count, 'start_width': box_width}
                else:
                    frames_passed = frame_count - speed_memory[track_id]['start_frame']
                    
                    if frames_passed >= 15: # Calculate every 0.5 seconds
                        w1 = speed_memory[track_id]['start_width']
                        w2 = box_width
                        
                        if w1 > 20 and w2 > 20: # Ignore very far objects
                            d1 = (REAL_CAR_WIDTH * FOCAL_LENGTH) / w1
                            d2 = (REAL_CAR_WIDTH * FOCAL_LENGTH) / w2
                            time_sec = frames_passed / fps
                            
                            # Positive speed = moving away (overtaking you)
                            relative_speed_mps = (d2 - d1) / time_sec
                            relative_speed_kmph = int(relative_speed_mps * 3.6)
                            
                            if relative_speed_kmph > 20: # 20km/h faster than dashcam
                                is_speeding = True
                                cv2.putText(annotated_frame, f"+{relative_speed_kmph} km/h", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                cv2.putText(annotated_frame, f"{relative_speed_kmph} km/h", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Reset for next calculation
                        speed_memory[track_id] = {'start_frame': frame_count, 'start_width': box_width}
                
                # --- OCR TRIGGER LOGIC ---
                if is_speeding and track_id not in plate_memory:
                    plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    temp_path = f"temp_crop_{track_id}.jpg"
                    
                    if plate_crop.size != 0:
                        cv2.imwrite(temp_path, plate_crop)
                        
                        plate_text = extract_indian_license_plate(temp_path)
                        print(f"[DEBUG OCR] Vehicle ID: {track_id} | Speed: +{relative_speed_kmph}km/h -> Read: '{plate_text}'")

                        if "Invalid" not in plate_text and plate_text != "Image Load Error":
                            log_violation(plate_text, f"Reckless Driving (+{relative_speed_kmph}km/h)", 22.7196, 75.8577)
                            plate_memory[track_id] = plate_text 
                        else:
                            plate_memory[track_id] = f"Err:{plate_text[:6]}"
                        
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                # Show OCR text above the vehicle box
                display_text = plate_memory.get(track_id, "Scanning...")
                if "Err" in display_text:
                    cv2.putText(annotated_frame, f"ID:{track_id} {display_text}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    cv2.putText(annotated_frame, f"ID:{track_id} {display_text}", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # =================================================
        # DISPLAY & SAVE OUTPUT
        # =================================================
        out.write(annotated_frame)  # Yeh frame ko nayi video file me likhega
        
        cv2.imshow("Smart City Dual-Pipeline", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Video file properly save hone ke liye zaroori
    cv2.destroyAllWindows()
    print("Pipeline Execution Complete! Nayi video 'output_processed_video.mp4' me save ho gayi hai.")

if __name__ == "__main__":
    run_pipeline()