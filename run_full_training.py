"""
UltraXAI one-click training pipeline (run on your local GPU machine).

Steps:
  1. Stratified split of the ORIGINAL images into train / val / test folders
     and a split manifest (datasets/prepared/split_manifest.csv/.json)
  2. Offline augmentation of the TRAIN folder only (dataset volume x (copies+1))
  3. Automated check that no image / augmented variant is shared between splits
  4. Train ResNet50 + EfficientNetV2-S classifiers on lesion ROI crops -> backend/weights/
  5. (optional, --train_unet) Train Attention U-Net on the same train/val split

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
from data.manifest import write_manifest, verify_no_overlap  # noqa: E402


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_unet", action="store_true", help="Also train the Attention U-Net")
    p.add_argument("--unet_epochs", type=int, default=150)
    a = p.parse_args()

    work = ROOT / a.work_dir
    if work.exists():
        shutil.rmtree(work)

    print("\n[1/5] Splitting ORIGINAL images (70/15/15, stratified, seed %d)..." % a.seed)
    splits = create_train_val_test_split([str(ROOT / a.data_dir)], random_state=a.seed)
    for name in ("train", "val", "test"):
        copy_split(splits[name], work / name)
    write_manifest(splits, str(work))

    print(f"\n[2/5] Augmenting TRAIN split only with {a.copies} copies per image...")
    augment_dataset(str(work / "train"), str(work / "train_aug"), a.copies, 256,
                    keep_original=False, seed=a.seed)

    print("\n[3/5] Verifying that train / val / test do not overlap...")
    if not verify_no_overlap(str(work)):
        sys.exit("Data leakage detected - fix the split before training.")

    print("\n[4/5] Training classifiers on lesion ROI crops...")
    cmd = [
        sys.executable, str(ROOT / "training" / "train_classifiers.py"),
        "--data_dirs", str(work / "train"),
        "--aug_dirs", str(work / "train_aug"),
        "--val_dirs", str(work / "val"),
        "--output_dir", str(ROOT / "backend" / "weights"),
        "--epochs", str(a.epochs), "--batch_size", str(a.batch_size), "--lr", str(a.lr),
        "--seed", str(a.seed),
        "--device", os.environ.get("ULTRAXAI_DEVICE", "cuda"),
    ]
    subprocess.run(cmd, check=True)

    if a.train_unet:
        print("\n[5/5] Training Attention U-Net on the same train / val split...")
        subprocess.run([
            sys.executable, str(ROOT / "training" / "train_attention_unet.py"),
            "--train_dirs", str(work / "train"), str(work / "train_aug"),
            "--val_dirs", str(work / "val"),
            "--output_dir", str(ROOT / "backend" / "weights"),
            "--epochs", str(a.unet_epochs), "--seed", str(a.seed),
            "--device", os.environ.get("ULTRAXAI_DEVICE", "cuda"),
        ], check=True)
    else:
        print("\n[5/5] Skipped Attention U-Net (pass --train_unet to train it).")
    print(f"\nDone. Weights saved to backend/weights. Held-out test set: {work / 'test'}")


if __name__ == "__main__":
    main()
