import os
import sys
import cv2
import numpy as np
import torch

sys.path.append(r"e:\FYP\training")
sys.path.append(r"e:\FYP\backend")
from app.services.pipeline_orchestrator import PipelineOrchestrator
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = r"e:\FYP\backend\weights"

# Initialize Orchestrator
orchestrator = PipelineOrchestrator(weights_dir=weights_dir, device=device)

# Load raw YOLO
yolo_path = os.path.join(weights_dir, "yolov8m_seg_breast.pt")
yolo_model = YOLO(yolo_path)

# Image Path
img_path = r"e:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\malignant\malignant (16).png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
image = np.squeeze(image)

print("=== YOLO DIRECT RUN ===")
img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
yolo_res = yolo_model(img_3ch, conf=0.25, verbose=False)
if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
    for idx, box in enumerate(yolo_res[0].boxes):
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        conf = box.conf[0].item()
        print(f"Direct YOLO Box {idx}: {xyxy} | Conf: {conf:.4f}")
else:
    print("Direct YOLO: No boxes detected")

print("\n=== ORCHESTRATOR RUN ===")
det_res = orchestrator._run_detection(image)
print(f"Orchestrator Detected: {det_res['detected']}")
if det_res['detected']:
    print(f"Orchestrator Box: {det_res['bbox']}")
    print(f"Orchestrator ROI shape: {det_res['roi'].shape}")
    
    # Run segmentation
    seg_res = orchestrator._run_segmentation(det_res['roi'])
    print(f"Orchestrator Mask sum: {seg_res['mask'].sum()}")
    print(f"Orchestrator Mask Max: {seg_res['mask'].max()}")
    print(f"Orchestrator Mask Min: {seg_res['mask'].min()}")
    
    # Run classification
    cls_res = orchestrator._run_classification(seg_res['segmented_roi'])
    print(f"Orchestrator Pred: {cls_res['prediction']} (Efficient: {cls_res['efficient_prediction']}, ResNet: {cls_res['resnet_prediction']})")
