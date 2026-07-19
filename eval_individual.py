import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(r"e:\FYP\training")
from models.classifier_heads import build_ensemble
from data.dataset import BreastUltrasoundDataset
from data.transforms import get_classification_val_transforms
from utils.metrics import compute_classification_metrics, print_classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = r"e:\FYP\backend\weights"
resnet_path = os.path.join(weights_dir, "resnet50_breast.pth")
efficient_path = os.path.join(weights_dir, "efficientnetv2_breast.pth")

# Load ensemble
ensemble = build_ensemble(num_classes=2, resnet_path=resnet_path, efficient_path=efficient_path)
ensemble.to(device).eval()

# Load validation dataset
data_dirs = [r"e:\FYP\datasets\BUSI\Dataset_BUSI_with_GT"]
val_transform = get_classification_val_transforms(224)
val_dataset = BreastUltrasoundDataset(
    root_dirs=data_dirs,
    mode="classification",
    transform=val_transform,
    img_size=(224, 224),
)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

def eval_model(model_name):
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)
            
            if model_name == "resnet50":
                logits = ensemble.resnet(images)
            elif model_name == "efficientnetv2":
                logits = ensemble.efficient(images)
            else:
                logits = ensemble(images)["logits"]
                
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    print(f"\n===== EVALUATION: {model_name.upper()} =====")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f} (Malignant Recall)")
    print(f"Specificity: {metrics['specificity']:.4f} (Benign Recall)")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")

eval_model("resnet50")
eval_model("efficientnetv2")
eval_model("ensemble")
