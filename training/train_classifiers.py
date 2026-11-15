"""
Classifier Ensemble Training Script.

Trains ResNet50 and EfficientNetV2-S classifiers on lesion ROI crops
(ground-truth mask bounding box + 15% padding, the same crop geometry the
backend uses at inference) for binary Benign (0) vs Malignant (1) diagnosis.

Usage (prepared split folders -- recommended, see run_full_training.py):
    python train_classifiers.py --data_dirs datasets/prepared/train \
        --aug_dirs datasets/prepared/train_aug --val_dirs datasets/prepared/val

Usage (raw dataset; a stratified 70/15/15 split is made and only its train/val
partitions are used -- the test partition is never trained or validated on):
    python train_classifiers.py --data_dirs datasets/BUSI
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).parent))

from models.classifier_heads import ResNet50Classifier, EfficientNetV2SClassifier
from data.split import create_train_val_test_split
from utils.run_logging import save_run_config, EpochLogger
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
    parser.add_argument("--full_image", action="store_true",
                        help="Train on whole images instead of lesion ROI crops (not recommended: "
                             "the backend classifies ROI crops)")
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
    log_name = model_name.lower().replace(" ", "_")
    save_run_config(args.output_dir, log_name, {
        **vars(args), "architecture": model.__class__.__name__,
        "optimizer": "AdamW", "scheduler": f"linear warmup ({args.warmup_epochs} ep) + cosine",
        "loss": f"weighted CrossEntropy (label_smoothing={args.label_smoothing})",
        "class_weights": class_weights.tolist(), "grad_clip": 1.0,
        "amp": device.type == "cuda", "selection": "best val F1 (Malignant = positive)",
        "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
    })
    logger = EpochLogger(args.output_dir, log_name)

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

        logger.log(epoch=epoch, train_loss=train_loss / num_train_batches, train_acc=train_acc,
                   val_loss=val_loss / num_val_batches, val_acc=metrics["accuracy"],
                   val_f1=metrics["f1_score"], val_sensitivity=metrics["sensitivity"],
                   val_specificity=metrics["specificity"], val_auc=metrics.get("auc_roc"),
                   lr=optimizer.param_groups[0]["lr"], seconds=elapsed)

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

    # Datasets (lesion ROI crops unless --full_image)
    size = (args.img_size, args.img_size)
    roi = not args.full_image
    train_dataset = BreastUltrasoundDataset(
        args.data_dirs, "classification", train_transform, size, roi_crop=roi)

    if args.val_dirs:
        val_set = BreastUltrasoundDataset(
            args.val_dirs, "classification", val_transform, size, roi_crop=roi)
        train_set = train_dataset
    else:
        # No separate val folder: make the standard 70/15/15 split and use only
        # train/val -- the test partition is held out (never trained on).
        splits = create_train_val_test_split(args.data_dirs, random_state=args.seed)
        train_paths = {s["image_path"] for s in splits["train"]}
        val_paths = {s["image_path"] for s in splits["val"]}
        val_dataset = BreastUltrasoundDataset(
            args.data_dirs, "classification", val_transform, size, roi_crop=roi)
        train_dataset.samples = [s for s in train_dataset.samples if s["image_path"] in train_paths]
        val_dataset.samples = [s for s in val_dataset.samples if s["image_path"] in val_paths]
        train_set, val_set = train_dataset, val_dataset

    train_labels = [smp["label"] for smp in train_dataset.samples]
    class_weights = _weights_from_labels(train_labels)

    # Offline-augmented copies are added to TRAIN only
    if args.aug_dirs:
        aug_dataset = BreastUltrasoundDataset(
            root_dirs=args.aug_dirs, mode="classification",
            transform=train_transform, img_size=size, roi_crop=roi,
        )
        train_set = ConcatDataset([train_set, aug_dataset])
        class_weights = _weights_from_labels(
            train_labels + [smp["label"] for smp in aug_dataset.samples]
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

    # Train EfficientNetV2-S (checkpoint keeps the name efficientnetv2_breast.pth)
    efficient = EfficientNetV2SClassifier(num_classes=2)
    efficient_path = train_single_model(
        efficient, "EfficientNetV2", train_loader, val_loader, args, device, class_weights
    )

    print("\n" + "=" * 70)
    print("  ALL CLASSIFIERS TRAINED SUCCESSFULLY")
    print(f"  ResNet50:       {resnet_path}")
    print(f"  EfficientNetV2-S: {efficient_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
