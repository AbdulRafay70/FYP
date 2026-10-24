"""
Offline Data Augmentation — Dataset Volume Expansion.

Generates N augmented copies of every image (and its mask, if present) and
writes them to a new directory using the same class-folder layout
(benign/ malignant/ normal/), so the result can be passed directly to
BreastUltrasoundDataset via --data_dirs / --aug_dirs.

IMPORTANT: Run this on the TRAINING split only, otherwise augmented copies
of validation/test images leak into training and inflate metrics.

Usage:
    python -m data.augment_offline --input_dir datasets/BUSI \
        --output_dir datasets/BUSI_aug --copies 5 --img_size 256
"""

import argparse
import glob
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

CLASSES = ["benign", "malignant", "normal"]
EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def build_offline_pipeline(img_size: int) -> A.Compose:
    """Ultrasound-safe augmentations applied jointly to image + mask."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.08, 0.08), scale=(1-0.15, 1+0.15), rotate=(-20, 20), border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.OneOf([
            A.ElasticTransform(alpha=60, sigma=60 * 0.05, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        ], p=0.35),
        A.OneOf([
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=1.0),  # speckle
            A.GaussNoise(std_range=(0.01, 0.02), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.2),
    ])


def find_mask(image_path: str):
    base, ext = os.path.splitext(image_path)
    for cand in (f"{base}_mask{ext}", f"{base}_mask.png"):
        if os.path.exists(cand):
            return cand
    return None


def find_class_dir(root: Path, cls: str):
    direct = root / cls
    if direct.exists():
        return direct
    found = glob.glob(os.path.join(root, "**", cls), recursive=True)
    return Path(found[0]) if found else None


def augment_dataset(input_dir: str, output_dir: str, copies: int, img_size: int,
                    keep_original: bool = True, seed: int = 42) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    pipeline = build_offline_pipeline(img_size)
    stats = {}

    for cls in CLASSES:
        class_dir = find_class_dir(Path(input_dir), cls)
        if class_dir is None:
            continue
        out_dir = Path(output_dir) / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        images = []
        for ext in EXTS:
            images.extend(glob.glob(str(class_dir / ext)))
        images = sorted(f for f in images if "_mask" not in os.path.basename(f).lower())

        written = 0
        for img_path in images:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            mask_path = find_mask(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
            stem = Path(img_path).stem

            if keep_original:
                cv2.imwrite(str(out_dir / f"{stem}.png"), cv2.resize(image, (img_size, img_size)))
                if mask is not None:
                    m = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), m)
                written += 1

            for i in range(copies):
                if mask is not None:
                    res = pipeline(image=image, mask=mask)
                    cv2.imwrite(str(out_dir / f"{stem}_aug{i}_mask.png"), res["mask"])
                else:
                    res = pipeline(image=image)
                cv2.imwrite(str(out_dir / f"{stem}_aug{i}.png"), res["image"])
                written += 1

        stats[cls] = (len(images), written)
        print(f"[Augment] {cls:10s}: {len(images):5d} originals -> {written:6d} images")

    total_in = sum(v[0] for v in stats.values())
    total_out = sum(v[1] for v in stats.values())
    print(f"[Augment] TOTAL: {total_in} -> {total_out} images "
          f"({total_out / max(total_in, 1):.1f}x) saved to {output_dir}")
    return stats


def parse_args():
    p = argparse.ArgumentParser(description="Offline dataset augmentation / expansion")
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--copies", type=int, default=5, help="Augmented copies per image")
    p.add_argument("--img_size", type=int, default=256)
    p.add_argument("--no_original", action="store_true", help="Do not copy originals")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    augment_dataset(a.input_dir, a.output_dir, a.copies, a.img_size,
                    keep_original=not a.no_original, seed=a.seed)
