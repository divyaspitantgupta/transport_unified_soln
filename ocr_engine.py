import cv2
import pytesseract
import re
import pandas as pd
from datetime import datetime
import os

# WINDOWS KE LIYE YEH LINE ADD KARNA ZAROORI HAI
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# TESSDATA PREFIX (Isko wapas tessdata wale folder par set karna hai)
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR\tessdata'
def preprocess_plate_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return thresh
def extract_indian_license_plate(image_path):
    processed_img = preprocess_plate_image(image_path)
    if processed_img is None:
        return "Image Load Error"

    # Tesseract configuration for alphanumeric characters with explicit tessdata path
    custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(processed_img, config=custom_config)

    clean_text = "".join(raw_text.split()).upper()
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$'

    if re.match(pattern, clean_text):
        return clean_text
    else:
        return f"Invalid Format ({clean_text})"

def log_violation(plate_number, violation_type, lat, lon):
    if "Invalid" not in plate_number and plate_number !="Image Load Error":
        data={
            "Timestamp":[datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Plate_Number":[plate_number],
            "Violation":[violation_type],
            "Latitude":[lat],
            "Longitude":[lon]
        }
        df=pd.DataFrame(data)
        df.to_csv('plate_recognition_results.csv',mode='a',index=False, header=False)
        print(f"Logged Violation: {plate_number} | {violation_type}")

if __name__ == "__main__":
    print("OCR Engine Ready. Waiting for triggers from the tracking module...")