"""
Held-out Test Set Evaluation for the Classification Ensemble.

Run ONCE, after model selection, on the untouched test split. Reports
ResNet50, EfficientNetV2-S and the weighted ensemble (0.4 / 0.6) with
accuracy, precision, sensitivity, specificity, F1, ROC-AUC, the confusion
matrix and the test sample count. Results are written to JSON + PNG.

Usage:
    python evaluate_test.py --test_dir ../datasets/prepared/test \
        --weights_dir ../backend/weights --output_dir ../evaluation/results
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models.classifier_heads import build_ensemble
from data.dataset import BreastUltrasoundDataset
from data.transforms import get_classification_val_transforms
from utils.metrics import compute_classification_metrics, print_classification_report


def save_confusion_matrix(m: dict, title: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[m["true_negatives"], m["false_positives"]],
                   [m["false_negatives"], m["true_positives"]]])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["Benign", "Malignant"])
    ax.set_yticks([0, 1], ["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Evaluate classifiers on the held-out test set")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--weights_dir", default="../backend/weights")
    p.add_argument("--output_dir", default="../evaluation/results")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else ("cpu" if args.device == "auto" else args.device))
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = BreastUltrasoundDataset(
        root_dirs=[args.test_dir], mode="classification",
        transform=get_classification_val_transforms(args.img_size),
        img_size=(args.img_size, args.img_size),
    )
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    ensemble = build_ensemble(
        num_classes=2,
        resnet_path=os.path.join(args.weights_dir, "resnet50_breast.pth"),
        efficient_path=os.path.join(args.weights_dir, "efficientnetv2_breast.pth"),
    ).to(device).eval()

    labels, probs = [], {"resnet50": [], "efficientnetv2_s": [], "ensemble": []}
    with torch.no_grad():
        for images, y in loader:
            out = ensemble(images.to(device))
            probs["resnet50"].append(torch.softmax(out["resnet_logits"], 1)[:, 1].cpu())
            probs["efficientnetv2_s"].append(torch.softmax(out["efficient_logits"], 1)[:, 1].cpu())
            probs["ensemble"].append(out["probabilities"][:, 1].cpu())
            labels.append(torch.as_tensor(y))

    y_true = torch.cat(labels).numpy()
    results = {"test_samples": int(len(y_true)),
               "benign": int((y_true == 0).sum()), "malignant": int((y_true == 1).sum())}
    for name, plist in probs.items():
        p_mal = torch.cat(plist).numpy()
        y_pred = (p_mal >= 0.5).astype(int)
        m = compute_classification_metrics(y_true, y_pred, p_mal)
        results[name] = {k: float(v) for k, v in m.items()}
        print(f"\n===== {name} (test, n={len(y_true)}) =====")
        print_classification_report(y_true, y_pred)
        save_confusion_matrix(m, f"{name} — test set",
                              os.path.join(args.output_dir, f"confusion_matrix_{name}.png"))

    out_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {out_path} + confusion matrix PNGs in {args.output_dir}")


if __name__ == "__main__":
    main()
