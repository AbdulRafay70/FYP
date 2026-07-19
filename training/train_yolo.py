"""
YOLOv8-Seg Training Script for Breast Lesion Detection.

Uses the Ultralytics framework to train YOLOv8m-seg for simultaneous
bounding box detection and instance segmentation of breast lesions.

Prerequisites:
    pip install ultralytics

Dataset must be in YOLO format:
    datasets/yolo_format/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── data.yaml

Usage:
    python train_yolo.py --data datasets/yolo_format/data.yaml --epochs 100
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8-Seg")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m-seg.pt", help="Base model")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--project", default="runs/segment", help="Project directory")
    parser.add_argument("--name", default="breast_yolo", help="Run name")
    parser.add_argument("--busi_dir", default=None, help="Path to raw BUSI dataset")
    return parser.parse_args()


def create_data_yaml(dataset_dir: str, output_path: str):
    """
    Create a YOLO data.yaml configuration file.

    This utility converts BUSI-style datasets into YOLO format.
    """
    yaml_content = f"""# UltraXAI — Breast Ultrasound Dataset Configuration
# Auto-generated for YOLOv8-Seg training

path: {dataset_dir}
train: images/train
val: images/val

# Classes
nc: 2
names:
  0: benign
  1: malignant
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_content)
    print(f"[YOLO] data.yaml created at: {output_path}")


def convert_busi_to_yolo(busi_dir: str, output_dir: str):
    """
    Convert BUSI dataset format to YOLO segmentation format.

    BUSI format:
        benign/image.png + benign/image_mask.png
    YOLO format:
        images/train/image.png + labels/train/image.txt
    """
    import cv2
    import numpy as np
    from sklearn.model_selection import train_test_split
    import glob

    print(f"[YOLO] Converting BUSI format to YOLO format...")

    label_map = {"benign": 0, "malignant": 1}
    all_images = []
    all_labels_data = []

    for class_name, class_id in label_map.items():
        class_dir = os.path.join(busi_dir, class_name)
        if not os.path.exists(class_dir):
            continue

        image_files = sorted(glob.glob(os.path.join(class_dir, "*.png")))
        image_files = [f for f in image_files if "_mask" not in os.path.basename(f)]

        for img_path in image_files:
            mask_path = img_path.replace(".png", "_mask.png")
            if not os.path.exists(mask_path):
                continue

            all_images.append((img_path, mask_path, class_id))

    # Split into train/val
    train_data, val_data = train_test_split(all_images, test_size=0.2, random_state=42)

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        img_dir = os.path.join(output_dir, "images", split_name)
        lbl_dir = os.path.join(output_dir, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for idx, (img_path, mask_path, class_id) in enumerate(split_data):
            # Copy image
            import shutil
            dst_img = os.path.join(img_dir, f"{split_name}_{idx:04d}.png")
            shutil.copy2(img_path, dst_img)

            # Convert mask to YOLO polygon format
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape
            contours, _ = cv2.findContours(
                (mask > 128).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            label_file = os.path.join(lbl_dir, f"{split_name}_{idx:04d}.txt")
            with open(label_file, "w") as f:
                for contour in contours:
                    if len(contour) < 3:
                        continue
                    # Normalize coordinates to [0, 1]
                    points = contour.reshape(-1, 2)
                    normalized = []
                    for px, py in points:
                        normalized.extend([px / w, py / h])
                    coords = " ".join(f"{c:.6f}" for c in normalized)
                    f.write(f"{class_id} {coords}\n")

        print(f"  {split_name}: {len(split_data)} images")

    # Create data.yaml
    create_data_yaml(output_dir, os.path.join(output_dir, "data.yaml"))
    print(f"[YOLO] Conversion complete! Output: {output_dir}")


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: Ultralytics not installed. Run: pip install ultralytics")
        return

    print("\n" + "=" * 70)
    print("  YOLOv8-SEG TRAINING — Breast Lesion Detection & Segmentation")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Data:     {args.data}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  ImgSize:  {args.imgsz}")
    print(f"  Batch:    {args.batch}")
    print("=" * 70 + "\n")

    # Auto-convert if data.yaml does not exist
    if not os.path.exists(args.data):
        print(f"[YOLO] Data config {args.data} not found.")
        
        # Check if user specified busi_dir directly
        busi_dir = None
        if args.busi_dir and os.path.exists(args.busi_dir):
            if os.path.exists(os.path.join(args.busi_dir, "benign")):
                busi_dir = args.busi_dir
            elif os.path.exists(os.path.join(args.busi_dir, "Dataset_BUSI_with_GT", "benign")):
                busi_dir = os.path.join(args.busi_dir, "Dataset_BUSI_with_GT")

        if not busi_dir:
            # Try to find BUSI dataset in candidates (including parent directory)
            parent_dir = Path(args.data).parent.parent
            busi_candidates = [
                os.path.join(parent_dir, "BUSI", "Dataset_BUSI_with_GT"),
                os.path.join(parent_dir, "BUSI"),
                "../datasets/BUSI/Dataset_BUSI_with_GT",
                "../datasets/BUSI",
                "datasets/BUSI/Dataset_BUSI_with_GT",
                "datasets/BUSI",
            ]
            for candidate in busi_candidates:
                if os.path.exists(os.path.join(candidate, "benign")):
                    busi_dir = candidate
                    break
        
        if busi_dir:
            print(f"[YOLO] Found raw BUSI dataset at: {busi_dir}")
            output_dir = str(Path(args.data).parent)
            convert_busi_to_yolo(busi_dir, output_dir)
        else:
            print(f"ERROR: Could not find BUSI dataset. Checked candidates: {[args.busi_dir] if args.busi_dir else []} + {[c for c in busi_candidates if 'busi_candidates' in locals()]}")
            print("Please either:")
            print("  1. Specify the path explicitly using --busi_dir '/path/to/BUSI'")
            print("  2. Ensure the raw BUSI dataset is uploaded to Google Drive.")
            return

    # Load model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        flipud=0.3,
        fliplr=0.5,
        mosaic=0.5,
        scale=0.3,
    )

    print("\n" + "=" * 70)
    print("  YOLOv8-SEG TRAINING COMPLETE")
    print(f"  Best weights: {args.project}/{args.name}/weights/best.pt")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
