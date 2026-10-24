"""
UltraXAI one-click training pipeline (run on your local GPU machine).

Steps:
  1. Stratified split of the raw dataset into train / val / test folders
  2. Offline augmentation of the TRAIN folder only (dataset volume x (copies+1))
  3. Train ResNet50 + EfficientNetV2 classifiers -> backend/weights/

Usage (from the FYP folder):
    python run_full_training.py --data_dir datasets/BUSI --copies 5 --epochs 80

Developed by Abdul Rafay — UltraXAI Final Year Project.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "training"))

from data.split import create_train_val_test_split  # noqa: E402
from data.augment_offline import augment_dataset  # noqa: E402


def copy_split(samples, out_dir: Path):
    for s in samples:
        dst = out_dir / s["class_name"]
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s["image_path"], dst)
        if s.get("mask_path"):
            shutil.copy2(s["mask_path"], dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="datasets/BUSI")
    p.add_argument("--work_dir", default="datasets/prepared")
    p.add_argument("--copies", type=int, default=5)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()

    work = ROOT / a.work_dir
    if work.exists():
        shutil.rmtree(work)

    print("\n[1/3] Splitting dataset (70/15/15, stratified)...")
    splits = create_train_val_test_split([str(ROOT / a.data_dir)])
    for name in ("train", "val", "test"):
        copy_split(splits[name], work / name)

    print(f"\n[2/3] Augmenting TRAIN split with {a.copies} copies per image...")
    augment_dataset(str(work / "train"), str(work / "train_aug"), a.copies, 256, keep_original=False)

    print("\n[3/3] Training classifiers on GPU...")
    cmd = [
        sys.executable, str(ROOT / "training" / "train_classifiers.py"),
        "--data_dirs", str(work / "train"),
        "--aug_dirs", str(work / "train_aug"),
        "--output_dir", str(ROOT / "backend" / "weights"),
        "--epochs", str(a.epochs), "--batch_size", str(a.batch_size), "--lr", str(a.lr),
        "--device", "cuda",
    ]
    subprocess.run(cmd, check=True)
    print(f"\nDone. Weights saved to backend/weights. Held-out test set: {work / 'test'}")


if __name__ == "__main__":
    main()
