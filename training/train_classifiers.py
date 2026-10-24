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
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split

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
    parser.add_argument("--aug_dirs", nargs="*", default=[],
                        help="Offline-augmented TRAIN-only dirs (from data/augment_offline.py)")
    parser.add_argument("--val_dirs", nargs="*", default=[],
                        help="Separate validation dirs (recommended with --aug_dirs to avoid leakage)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Stratified validation fraction")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    parser.add_argument("--mixup_alpha", type=float, default=0.2, help="Mixup alpha (0 disables)")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Linear LR warmup epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--no_tta", action="store_true", help="Disable flip test-time augmentation")
    parser.add_argument("--seed", type=int, default=42)
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

    # Weighted, label-smoothed loss for class imbalance + better calibration
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device), label_smoothing=args.label_smoothing
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # Linear warmup followed by cosine decay (stepped per epoch)
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = LambdaLR(optimizer, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1 = -1.0
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

            # Mixup regularisation
            lam = 1.0
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                perm = torch.randperm(images.size(0), device=device)
                images = lam * images + (1 - lam) * images[perm]

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
                if lam < 1.0:
                    loss = lam * loss + (1 - lam) * criterion(logits, labels[perm])

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
                if not args.no_tta:
                    # Horizontal-flip test-time augmentation
                    probs = 0.5 * (probs + torch.softmax(model(torch.flip(images, dims=[3])), dim=1))
                preds = probs.argmax(dim=1)

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


def _weights_from_labels(labels) -> torch.Tensor:
    """Inverse-frequency class weights for the binary task."""
    counts = np.bincount(np.asarray(labels), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    return torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32)


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n[Device] Using: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Stratified train/val split (previously val == train, which leaked data)
    labels_all = [smp["label"] for smp in train_dataset.samples]
    if args.val_dirs:
        train_idx = np.arange(len(labels_all))
        val_set = BreastUltrasoundDataset(
            root_dirs=args.val_dirs, mode="classification",
            transform=val_transform, img_size=(args.img_size, args.img_size),
        )
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(labels_all)), test_size=args.val_split,
            stratify=labels_all, random_state=args.seed,
        )
        val_set = Subset(val_dataset, val_idx)
    class_weights = _weights_from_labels([labels_all[i] for i in train_idx])
    train_set = Subset(train_dataset, train_idx)

    # Offline-augmented copies are added to TRAIN only
    if args.aug_dirs:
        aug_dataset = BreastUltrasoundDataset(
            root_dirs=args.aug_dirs, mode="classification",
            transform=train_transform, img_size=(args.img_size, args.img_size),
        )
        train_set = ConcatDataset([train_set, aug_dataset])
        class_weights = _weights_from_labels(
            [labels_all[i] for i in train_idx] + [smp["label"] for smp in aug_dataset.samples]
        )
    print(f"[Split] Train: {len(train_set)}  Val: {len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

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
