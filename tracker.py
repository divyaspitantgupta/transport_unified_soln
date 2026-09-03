import math
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
    
    # INDENTATION FIXED: Yeh line ab loop ke bahar hai
    return False

def main():
    # Model ki location update kar di hai taaki custom weights use hon
    model = YOLO("best1.onnx")
    
    # RAW STRING (r) ADDED: Taaki Windows backslash ka error na aaye
    video_path = r"C:\Users\p\Downloads\testvideo2.mp4"
    
    print("Starting Inference with ByteTrack...")
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",  # COMMA ADDED
        conf=0.4,
        show=True,save=True,
    )

    historical_defects = []
    for frame_result in results:
        pass

if __name__ == "__main__":
    main() # Isko call karna zaroori tha taaki inference start ho sake