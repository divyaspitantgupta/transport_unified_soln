import cv2
import pytesseract
import torch
from ultralytics import YOLO

# Tesseract ka path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_plate_text(plate_img):
    """ Cropped image se text nikalne ka function """
    # Agar crop bohot chhota hai toh ignore karo
    if plate_img.shape[0] < 10 or plate_img.shape[1] < 10:
        return ""
        
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    
    return "".join(text.split())

def main():
    torch.set_num_threads(4)
    
    # Yahan apne Number Plate detection wale YOLO model ka path daalna
    # (Agar pothole wala hi use kar raha hai toh uska path, par plate ke liye alag model better hota hai)
    model = YOLO("plate_model.pt") 
    
    video_path = r"C:\Users\p\Downloads\traffic_video.mp4" # Apni traffic video
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_traffic_ocr.mp4', fourcc, fps, (width, height))

    # YAHI HAI GAME CHANGER: Dictionary jo read ki hui plates yaad rakhegi
    plate_memory = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        # YOLO Tracking (har frame par chalega kyunki traffic fast hota hai)
        results = model.track(frame, tracker="bytetrack.yaml", conf=0.5, persist=True, verbose=False)
        annotated_frame = frame.copy()
        boxes = results[0].boxes
        
        if boxes is not None and boxes.id is not None:
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes.xyxy[i])
                track_id = int(boxes.id[i])
                
                # Bounding box draw kar
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Check karo ki kya humne is ID ko pehle padha hai?
                if track_id not in plate_memory or plate_memory[track_id] == "":
                    # Nayi gaadi hai! Frame me se plate crop karo
                    # Padding de rahe hain taaki text cut na ho
                    pad = 5
                    crop_img = frame[max(0, y1-pad):min(height, y2+pad), max(0, x1-pad):min(width, x2+pad)]
                    
                    # OCR ko bhejo
                    detected_text = extract_plate_text(crop_img)
                    
                    # Agar kuch mila (minimum 4 characters to hone chahiye plate me)
                    if len(detected_text) > 3:
                        plate_memory[track_id] = detected_text
                
                # Ab memory me se text uthao (agar hai toh)
                display_text = plate_memory.get(track_id, "Reading...")
                
                # Text ko box ke upar chipka do
                label = f"ID:{track_id} {display_text}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        cv2.imshow("Smart City - Traffic OCR", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Traffic Analytics complete! Video saved as output_traffic_ocr.mp4")

if __name__ == "__main__":
    main()