import os
import sys
import cv2
import torch
from ultralytics import YOLO

sys.path.append(r"e:\FYP\backend")
from app.config import settings

# Load model
yolo_path = os.path.join(r"e:\FYP\backend\weights", settings.YOLO_WEIGHTS)
model = YOLO(yolo_path)

# Load image
img_path = r"e:\FYP\datasets\us-dataset (1)\originals\malignant\us117.bmp"
img = cv2.imread(img_path)

# Run inference
results = model(img, conf=0.1, verbose=False) # check down to 0.1 confidence

print("--- YOLO DETECTIONS ---")
if len(results) == 0 or len(results[0].boxes) == 0:
    print("No detections found.")
else:
    for idx, box in enumerate(results[0].boxes):
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        label = "malignant" if cls == 1 else "benign"
        print(f"Detection {idx+1}: Class: {label} | Conf: {conf*100:.2f}% | Box: {xyxy}")
