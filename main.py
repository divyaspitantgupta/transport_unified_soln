import os
os.environ['PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT'] = '0'
from utils import (read_video,save_video)
from detections import CarDetection,LicensePlateDetection
def main():
    input_video_path=r"C:\Users\p\road-defect-pipeline\input_videos\video3.mp4"
    #Read Video
    video_frames=read_video(input_video_path)
    #Detect Car
    car_detector = CarDetection(model_path="yolo26n.pt")
    car_detections=car_detector.detect_frames(video_frames,read_from_stub=True,stub_path="tracker_stubs/car_detection.pkl")
    #Detect License Plate
    license_plate_detector = LicensePlateDetection(model_path=r"C:\Users\p\road-defect-pipeline\models\bestocr.pt")
    license_plate_detections,licence_plate_texts =license_plate_detector.detect_frames(video_frames)
    #Draw Car Bounding Boxes
    output_video_frames=car_detector.draw_bboxes(video_frames,car_detections)
    #Draw License Plate Bounding Boxes
    output_video_frames=license_plate_detector.draw_bboxes(video_frames,license_plate_detections,licence_plate_texts)
    #Save the Output Video
    save_video(video_frames, output_video_path="output_videos/output_video.avi")
if __name__ == "__main__":
    main() 