from ultralytics import YOLO

# 1. Naya model load karo
# Yahan apne asli 22 ghante wale model ka path daal
model_path = r"C:\Users\p\road-defect-pipeline\runs\detect\yolo11_road_defect_india\weights\best.pt"
model = YOLO(model_path)

# 2. Poore Test Dataset par mAP check karo
print("Final Board Exam shuru! Poore Test Dataset par testing ho rahi hai... 🚀")
metrics = model.val(
    data=r"C:\Users\p\road-defect-pipeline\data.yaml",  # Apna actual data.yaml ka path daalna
    split='test',   # YOLO ko bata rahe hain ki test folder use karna hai
    conf=0.20,      
    plots=True      # Taki graph aur confusion matrix wagaira save ho jayein
)

# 3. Final Result print karo
print("==================================")
print(f"🔥 Naye Model ka Test mAP50: {metrics.box.map50:.4f}")
print("==================================")