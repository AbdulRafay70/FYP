import os
import sys
import cv2
import numpy as np
import torch

sys.path.append(r"e:\FYP\training")
sys.path.append(r"e:\FYP\backend")
from models.classifier_heads import build_ensemble
from data.transforms import get_classification_val_transforms, get_segmentation_val_transforms
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = r"e:\FYP\backend\weights"
resnet_path = os.path.join(weights_dir, "resnet50_breast.pth")
efficient_path = os.path.join(weights_dir, "efficientnetv2_breast.pth")

ensemble = build_ensemble(num_classes=2, resnet_path=resnet_path, efficient_path=efficient_path)
ensemble.to(device).eval()

yolo_path = os.path.join(weights_dir, "yolov8m_seg_breast.pt")
yolo_model = YOLO(yolo_path)

# Load U-Net
from models.attention_unet import AttentionUNet
unet = AttentionUNet(in_channels=1, out_channels=1)
unet.load_state_dict(torch.load(os.path.join(weights_dir, "attention_unet_breast.pth"), map_location="cpu"))
unet.to(device).eval()

transform_cls = get_classification_val_transforms(224)
transform_seg = get_segmentation_val_transforms(256)

# Image Path
img_path = r"e:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\malignant\malignant (16).png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Failed to load image!")
    sys.exit()

image = np.squeeze(image)

def run_inf(img, label):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    transformed = transform_cls(image=img_rgb)
    tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        res = ensemble(tensor)
    pred = "Malignant" if res["prediction"].item() == 1 else "Benign"
    conf = res["confidence"].item()
    print(f"{label} Prediction: {pred} ({conf*100:.2f}%)")

# Run YOLO
img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
yolo_res = yolo_model(img_3ch, conf=0.25, verbose=False)
if len(yolo_res) > 0 and len(yolo_res[0].boxes) > 0:
    box = yolo_res[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    
    # 1. Orchestrator Crop (Padded 15%)
    h_img, w_img = image.shape[:2]
    pad_x = int((x2 - x1) * 0.15)
    pad_y = int((y2 - y1) * 0.15)
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(w_img, x2 + pad_x)
    y2_pad = min(h_img, y2 + pad_y)
    roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
    
    # Run U-net with exact normalization transforms
    h, w = roi.shape[:2]
    transformed = transform_seg(image=roi)
    roi_tensor = transformed["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        mask_out = torch.sigmoid(unet(roi_tensor)).squeeze().cpu().numpy()
    mask_binary = (cv2.resize(mask_out, (w, h)) > 0.5).astype(np.uint8)
    
    print(f"Mask sum (number of positive pixels): {mask_binary.sum()}")
    
    segmented_roi = roi * mask_binary
    
    # Predict on raw cropped ROI
    run_inf(roi, "Padded ROI (No Mask)")
    
    # Predict on segmented ROI
    run_inf(segmented_roi, "Padded Segmented ROI (With Mask)")
    
else:
    print("No YOLO detection found.")
