from ultralytics import YOLO

# Yahan base model nahi, balki pichli training ki 'last.pt' deni hoti hai
model = YOLO('runs/detect/yolo11_road_defect_india/weights/last.pt') 

# resume=True set karne se yeh 18th epoch se hi aage badhega
results = model.train(resume=True)