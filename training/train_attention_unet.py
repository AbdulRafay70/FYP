"""
Attention U-Net Training Script for Breast Ultrasound Segmentation.

Trains the Attention U-Net model on paired ultrasound image + mask data.
Uses combined Dice + BCE loss, early stopping, and comprehensive logging.

Usage:
    python train_attention_unet.py --data_dirs datasets/BUSI datasets/BUS_BRA \
                                   --epochs 150 --batch_size 8 --lr 1e-4
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.attention_unet import build_attention_unet, DiceBCELoss
from data.dataset import BreastUltrasoundDataset
from data.transforms import get_segmentation_train_transforms, get_segmentation_val_transforms
from data.split import create_train_val_test_split
from utils.metrics import dice_coefficient, iou_score, compute_segmentation_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train Attention U-Net")
    parser.add_argument("--data_dirs", nargs="+", required=True, help="Dataset directories")
    parser.add_argument("--output_dir", default="weights", help="Output directory for weights")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--device", default="auto", help="Device (auto/cuda/cpu)")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    """Determine the best available device."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[Device] Using CPU")
    else:
        device = torch.device(device_arg)
    return device


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and metrics."""
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        predictions = model(images)
        loss = criterion(predictions, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            pred_np = predictions.cpu().numpy()
            mask_np = masks.cpu().numpy()
            for i in range(pred_np.shape[0]):
                pred_bin = (pred_np[i, 0] > 0.5).astype(float)
                mask_bin = mask_np[i, 0]
                total_dice += dice_coefficient(pred_bin, mask_bin)
                total_iou += iou_score(pred_bin, mask_bin)

        total_loss += loss.item()
        num_batches += 1

    n_samples = num_batches * dataloader.batch_size
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / max(n_samples, 1),
        "iou": total_iou / max(n_samples, 1),
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model. Returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        predictions = model(images)
        loss = criterion(predictions, masks)

        pred_np = predictions.cpu().numpy()
        mask_np = masks.cpu().numpy()
        for i in range(pred_np.shape[0]):
            pred_bin = (pred_np[i, 0] > 0.5).astype(float)
            mask_bin = mask_np[i, 0]
            total_dice += dice_coefficient(pred_bin, mask_bin)
            total_iou += iou_score(pred_bin, mask_bin)

        total_loss += loss.item()
        num_batches += 1

    n_samples = num_batches * dataloader.batch_size
    return {
        "loss": total_loss / num_batches,
        "dice": total_dice / max(n_samples, 1),
        "iou": total_iou / max(n_samples, 1),
    }


def main():
    args = parse_args()
    device = get_device(args.device)

    print("\n" + "=" * 70)
    print("  ATTENTION U-NET TRAINING — Breast Ultrasound Segmentation")
    print("=" * 70)
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  LR:         {args.lr}")
    print(f"  Image Size: {args.img_size}x{args.img_size}")
    print(f"  Patience:   {args.patience}")
    print("=" * 70 + "\n")

    # Create data splits
    splits = create_train_val_test_split(args.data_dirs)

    # Create datasets
    train_transform = get_segmentation_train_transforms(args.img_size)
    val_transform = get_segmentation_val_transforms(args.img_size)

    train_dataset = BreastUltrasoundDataset(
        root_dirs=args.data_dirs,
        mode="segmentation",
        transform=train_transform,
        img_size=(args.img_size, args.img_size),
    )

    val_dataset = BreastUltrasoundDataset(
        root_dirs=args.data_dirs,
        mode="segmentation",
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

    # Build model
    model = build_attention_unet(in_channels=1, out_channels=1)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Training loop with early stopping
    best_dice = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    best_path = os.path.join(args.output_dir, "attention_unet_breast.pth")

    print("Starting training...\n")

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t_start

        # Logging
        print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"| Train Loss: {train_metrics['loss']:.4f} Dice: {train_metrics['dice']:.4f} "
            f"| Val Loss: {val_metrics['loss']:.4f} Dice: {val_metrics['dice']:.4f} IoU: {val_metrics['iou']:.4f} "
            f"| {elapsed:.1f}s"
        )

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best! Dice: {best_dice:.4f} — Saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  ✗ Early stopping triggered after {args.patience} epochs without improvement.")
                break

    print(f"\n{'='*70}")
    print(f"  Training Complete! Best Validation Dice: {best_dice:.4f}")
    print(f"  Best weights saved to: {best_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
