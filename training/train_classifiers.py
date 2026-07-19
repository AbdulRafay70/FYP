"""
Classifier Ensemble Training Script.

Trains ResNet50 and EfficientNetV2-B0 classifiers on cropped tumor ROIs
for Benign vs Malignant classification.

Usage:
    python train_classifiers.py --data_dirs datasets/BUSI datasets/BUS_BRA \
                                --epochs 80 --batch_size 16 --lr 1e-4
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.insert(0, str(Path(__file__).parent))

from models.classifier_heads import ResNet50Classifier, EfficientNetV2B0Classifier
from data.dataset import BreastUltrasoundDataset
from data.transforms import get_classification_train_transforms, get_classification_val_transforms
from utils.metrics import compute_classification_metrics, print_classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train Classification Ensemble")
    parser.add_argument("--data_dirs", nargs="+", required=True, help="Dataset directories")
    parser.add_argument("--output_dir", default="weights", help="Output directory")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    return parser.parse_args()


def train_single_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
    class_weights: torch.Tensor,
) -> str:
    """
    Train a single classifier model.

    Returns:
        Path to saved best weights
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING: {model_name}")
    print(f"{'='*70}\n")

    model = model.to(device)

    # Weighted loss for class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_f1 = 0.0
    patience_counter = 0
    save_path = os.path.join(args.output_dir, f"{model_name.lower().replace(' ', '_')}_breast.pth")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels_t = torch.tensor(labels).to(device) if not isinstance(labels, torch.Tensor) else labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels_t)

                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_t.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Metrics
        metrics = compute_classification_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

        scheduler.step()
        elapsed = time.time() - t_start

        # Logging
        num_train_batches = len(train_loader)
        num_val_batches = len(val_loader)
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"| Train Loss: {train_loss/num_train_batches:.4f} Acc: {train_acc:.4f} "
            f"| Val Loss: {val_loss/num_val_batches:.4f} F1: {metrics['f1_score']:.4f} "
            f"Sens: {metrics['sensitivity']:.4f} Spec: {metrics['specificity']:.4f} "
            f"| {elapsed:.1f}s"
        )

        # Save best
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best F1: {best_f1:.4f} — Saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ✗ Early stopping at epoch {epoch}")
                break

    # Final report
    print(f"\n  {model_name} — Best F1: {best_f1:.4f}")
    print(f"  Weights: {save_path}\n")

    return save_path


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n[Device] Using: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Transforms
    train_transform = get_classification_train_transforms(args.img_size)
    val_transform = get_classification_val_transforms(args.img_size)

    # Datasets
    train_dataset = BreastUltrasoundDataset(
        root_dirs=args.data_dirs,
        mode="classification",
        transform=train_transform,
        img_size=(args.img_size, args.img_size),
    )
    val_dataset = BreastUltrasoundDataset(
        root_dirs=args.data_dirs,
        mode="classification",
        transform=val_transform,
        img_size=(args.img_size, args.img_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Class weights
    class_weights = train_dataset.get_class_weights()
    print(f"[Class Weights] {class_weights}")

    # Train ResNet50
    resnet = ResNet50Classifier(num_classes=2)
    resnet_path = train_single_model(
        resnet, "ResNet50", train_loader, val_loader, args, device, class_weights
    )

    # Train EfficientNetV2
    efficient = EfficientNetV2B0Classifier(num_classes=2)
    efficient_path = train_single_model(
        efficient, "EfficientNetV2", train_loader, val_loader, args, device, class_weights
    )

    print("\n" + "=" * 70)
    print("  ALL CLASSIFIERS TRAINED SUCCESSFULLY")
    print(f"  ResNet50:       {resnet_path}")
    print(f"  EfficientNetV2: {efficient_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
