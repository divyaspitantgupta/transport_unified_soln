import cv2
import math
import torch  # SPEED HACK: CPU threads control karne ke liye
from ultralytics import YOLO

def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    # Haversine Formula
    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2

    c = 2 * math.asin(math.sqrt(a))
    distance = R * c
    return distance

def is_duplicate_defect(new_gps, historical_gps_list):
    """
    If distance 'd' < 5 meters, the detection is flagged as a duplicate and dropped.
    """
    for hist_gps in historical_gps_list:
        d = calculate_haversine_distance(
            hist_gps['lat'], hist_gps['lon'],
            new_gps['lat'], new_gps['lon']
        )
        if d < 5.0:
            return True
    
    return False

def main():
    # SPEED HACK: PyTorch ko CPU choke karne se rokne ke liye
    torch.set_num_threads(4) 
    
    # Tera custom trained model
    model = YOLO("best1.pt")
    
    # Video path
    video_path = r"C:\Users\p\Downloads\testvideo2.mp4"
    
    print("Starting HIGH-SPEED ROI Tracking with GEOMETRY FILTER... (Live screen is OFF)")
    
    cap = cv2.VideoCapture(video_path)
    
    # Video save setup
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_tracked.mp4', fourcc, fps, (width, height))

    historical_defects = []
    
    frame_count = 0
    last_annotated_roi = None  # Purane frame ka backup

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        frame_count += 1
        h, w = frame.shape[:2]
        
        # TRUCK FIX 1: Top 55% hissa crop (Aasmaan aur truck ki unchai out)
        crop_y = int(h * 0.55)
        
        # SPEED HACK: Har 3rd frame par hi YOLO process karega
        if frame_count % 3 == 0:
            roi_frame = frame[crop_y:h, :] 
            
            # TRUCK FIX 2: conf=0.65 rakha hai
            results = model.track(roi_frame, tracker="bytetrack.yaml", conf=0.65, persist=True, verbose=False,show=True)
            
            # TRUCK FIX 3 (THE GEOMETRY FILTER): 
            annotated_roi = roi_frame.copy()
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    # Box ke coordinates nikalna
                    x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                    
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_area = box_width * box_height
                    
                    # GEOMETRY LOGIC: Area 45,000 se kam ho, aur shape chouda (wide) ho
                    if box_area < 45000 and box_width > (box_height * 0.7):
                        # Filter pass! Matlab pakka gaddha hai. Draw kar do.
                        cv2.rectangle(annotated_roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        
                        # Agar ID assign ho chuki hai toh label par ID bhi likh do
                        if boxes.id is not None:
                            track_id = int(boxes.id[i])
                            cv2.putText(annotated_roi, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Naya tracked frame save kar lo
            last_annotated_roi = annotated_roi
            frame[crop_y:h, :] = last_annotated_roi
            
        else:
            # Agar YOLO skip hua hai, toh purana wala box chipka do taaki blank na dikhe
            if last_annotated_roi is not None:
                frame[crop_y:h, :] = last_annotated_roi
        
        # Output video save karo
        out.write(frame)
        
        # SPEED HACK: Live display band kar diya taaki CPU sirf export par zor lagaye
        # cv2.imshow("Smart City - ROI Tracker", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'): 
        #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Tracking complete! Makhan jaisi fast video 'output_tracked.mp4' me save ho gayi hai.")

if __name__ == "__main__":
    main()