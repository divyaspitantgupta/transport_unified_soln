from ultralytics import YOLO

def main():
    print("Loading YOLO11 architecture...")
    model=YOLO('yolo11n.pt')

    print("Starting training on RDD2022 Indian Subset...")
    results=model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device='cpu',
        name='yolo11_road_defect_india'
    )

    print("Training Complete!")
if __name__=='__main__':
    main()