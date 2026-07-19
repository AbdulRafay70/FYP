import os
import sys
import cv2
import torch
import numpy as np

sys.path.append(r"e:\FYP\backend")
sys.path.append(r"e:\FYP\training")
from app.config import settings
from app.services.pipeline_orchestrator import PipelineOrchestrator

# Initialize pipeline
weights_dir = r"e:\FYP\backend\weights"
device = "cuda" if torch.cuda.is_available() else "cpu"
orchestrator = PipelineOrchestrator(weights_dir=weights_dir, device=device)

# Load image
img_path = r"e:\FYP\datasets\BUSI\Dataset_BUSI_with_GT\malignant\malignant (16).png"
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Failed to load image!")
    sys.exit()

image = np.squeeze(image)
print(f"Original Image Shape: {image.shape}")

# Run full pipeline
results = orchestrator.run_full_pipeline(image)

print("\n--- PIPELINE DIAGNOSTIC RESULTS ---")
print(f"Final Diagnosis:        {results['prediction']}")
print(f"Confidence:             {results['confidence'] * 100:.2f}%")
print(f"Trust Score:            {results['trust_score']}")
print(f"Tumor Detected:         {results['tumor_detected']}")
print(f"ResNet Prediction:      {results['resnet_prediction']} (Conf: {results['resnet_confidence']*100:.2f}%)")
print(f"Efficient Prediction:   {results['efficient_prediction']} (Conf: {results['efficient_confidence']*100:.2f}%)")
