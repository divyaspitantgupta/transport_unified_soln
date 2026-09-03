import cv2
import os
from ocr_engine import extract_indian_license_plate, log_violation

video_path = r"C:\Users\p\Downloads\14571162_3840_2160_60fps.mp4"
cap = cv2.VideoCapture(video_path)

# ==========================================
# VIDEO SAVER SETUP
# ==========================================
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 25

# Hum 1280x720 resolution save karenge kyunki hum screen par wahi display kar rahe hain
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_test_ocr.mp4', fourcc, fps, (1280, 720))

print("Video chal rahi hai. OCR test karne ke liye 'c' dabana, aur band karne ke liye 'q'.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video khtam ho gayi!")
        break
        
    # Screen par fit karne ke liye resize
    display_frame = cv2.resize(frame, (1280, 720))
    
    # ------------------------------------------
    # FRAME KO NAYI VIDEO FILE MEIN WRITE KARO
    # ------------------------------------------
    out.write(display_frame)
    
    cv2.imshow("OCR Video Tester", display_frame)
    
    key = cv2.waitKey(25) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('c'):
        # 'c' dabate hi video pause hogi aur ROI selector khulega
        bbox = cv2.selectROI("Select Number Plate", display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Number Plate")
        
        # Bounding box coordinates
        x, y, w, h = [int(v) for v in bbox]
        
        # Agar user ne valid box banaya hai tabhi OCR run karo
        if w > 0 and h > 0:
            plate_crop = display_frame[y:y+h, x:x+w]
            temp_path = "temp_test_plate.jpg"
            cv2.imwrite(temp_path, plate_crop)
            
            print("\n--- OCR ENGINE TRIGGERED ---")
            result = extract_indian_license_plate(temp_path)
            print(f"Extracted Text: '{result}'")
            
            if "Invalid" not in result and result != "Image Load Error" and result.strip() != "":
                print("Format Validated! Logging to database...")
                log_violation(result, "Manual Video Test", 22.7196, 75.8577)
            else:
                print("Validation Failed. Not Logged.")
            
            # Temporary file hata do
            if os.path.exists(temp_path):
                os.remove(temp_path)

cap.release()
out.release() # Video properly save karne ke liye
cv2.destroyAllWindows()

print("Testing complete! Teri video 'output_test_ocr.mp4' project folder mein save ho gayi hai.")