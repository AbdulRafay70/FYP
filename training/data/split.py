"""
Stratified K-Fold Data Splitting for Breast Ultrasound Datasets.

Provides:
1. Train/Validation/Test split with stratification
2. K-Fold cross-validation splits
3. Multi-dataset combination with source tracking
4. Split statistics and verification
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split


def create_train_val_test_split(
    dataset_dirs: List[str],
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """
    Create stratified train/validation/test splits from multiple datasets.

    Args:
        dataset_dirs: List of dataset root directories
        test_size: Fraction of data for test set
        val_size: Fraction of data for validation set
        random_state: Seed for reproducibility
        output_path: Optional path to save the split configuration

    Returns:
        Dictionary with 'train', 'val', 'test' keys, each containing
        a list of sample dictionaries with 'image_path', 'label', etc.
    """
    # Collect all samples
    all_samples = _collect_samples(dataset_dirs)
    all_labels = [s["label"] for s in all_samples]

    print(f"\n{'='*60}")
    print(f"  DATASET SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total samples: {len(all_samples)}")
    _print_distribution("  Overall", all_labels)

    # First split: separate test set
    train_val_samples, test_samples, train_val_labels, _ = train_test_split(
        all_samples,
        all_labels,
        test_size=test_size,
        stratify=all_labels,
        random_state=random_state,
    )

    # Second split: separate validation from training
    relative_val_size = val_size / (1 - test_size)
    train_samples, val_samples, _, _ = train_test_split(
        train_val_samples,
        train_val_labels,
        test_size=relative_val_size,
        stratify=train_val_labels,
        random_state=random_state,
    )

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }

    # Print statistics
    for split_name, samples in splits.items():
        labels = [s["label"] for s in samples]
        print(f"\n  {split_name.upper()} set: {len(samples)} samples")
        _print_distribution(f"    {split_name}", labels)

    print(f"\n{'='*60}\n")

    # Save split configuration
    if output_path:
        _save_split_config(splits, output_path)

    return splits


def create_kfold_splits(
    dataset_dirs: List[str],
    n_folds: int = 5,
    random_state: int = 42,
    output_path: Optional[str] = None,
) -> List[Dict[str, List[Dict]]]:
    """
    Create Stratified K-Fold cross-validation splits.

    Each fold contains a 'train' and 'val' subset, with stratified
    class distribution maintained across all folds.

    Args:
        dataset_dirs: List of dataset root directories
        n_folds: Number of folds (default: 5)
        random_state: Seed for reproducibility
        output_path: Optional path to save fold configurations

    Returns:
        List of dictionaries, one per fold, each with 'train' and 'val' keys
    """
    all_samples = _collect_samples(dataset_dirs)
    all_labels = np.array([s["label"] for s in all_samples])
    all_samples_arr = np.array(all_samples, dtype=object)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    print(f"\n{'='*60}")
    print(f"  {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(all_samples_arr, all_labels)):
        fold = {
            "train": list(all_samples_arr[train_indices]),
            "val": list(all_samples_arr[val_indices]),
        }
        folds.append(fold)

        train_labels = all_labels[train_indices]
        val_labels = all_labels[val_indices]
        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Train: {len(train_indices)} samples | "
              f"Benign: {(train_labels == 0).sum()}, Malignant: {(train_labels == 1).sum()}")
        print(f"    Val:   {len(val_indices)} samples | "
              f"Benign: {(val_labels == 0).sum()}, Malignant: {(val_labels == 1).sum()}")

    print(f"\n{'='*60}\n")

    if output_path:
        _save_kfold_config(folds, output_path)

    return folds


def _collect_samples(dataset_dirs: List[str]) -> List[Dict]:
    """Scan dataset directories and collect all image sample metadata."""
    import glob

    label_map = {"benign": 0, "malignant": 1, "normal": 0}
    all_samples = []

    for root_dir in dataset_dirs:
        root_path = Path(root_dir)
        if not root_path.exists():
            print(f"  [Warning] Directory not found: {root_dir}")
            continue

        for class_name in ["benign", "malignant"]:
            class_dir = root_path / class_name
            if not class_dir.exists():
                # Search recursively for the class folder within the root path
                search_pattern = os.path.join(root_path, "**", class_name)
                found_dirs = glob.glob(search_pattern, recursive=True)
                if found_dirs:
                    class_dir = Path(found_dirs[0])
                else:
                    continue

            label = label_map[class_name]

            image_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                image_files.extend(glob.glob(str(class_dir / ext)))

            # Exclude mask files
            image_files = [
                f for f in image_files
                if "_mask" not in os.path.basename(f).lower()
            ]

            for img_path in sorted(image_files):
                # Check for corresponding mask
                mask_path = _find_mask_path(img_path)

                all_samples.append({
                    "image_path": img_path,
                    "mask_path": mask_path,
                    "label": label,
                    "class_name": class_name,
                    "source": root_path.name,
                })

    return all_samples


def _find_mask_path(image_path: str) -> Optional[str]:
    """Find the mask file corresponding to an image."""
    base = os.path.splitext(image_path)[0]
    ext = os.path.splitext(image_path)[1]

    for suffix in ["_mask", "_mask_1", "_mask1", "_segmentation", "_seg"]:
        candidate = f"{base}{suffix}{ext}"
        if os.path.exists(candidate):
            return candidate
    return None


def _print_distribution(prefix: str, labels: list):
    """Print class distribution."""
    dist = Counter(labels)
    total = len(labels)
    label_names = {0: "Benign", 1: "Malignant"}
    parts = []
    for label_id in sorted(dist.keys()):
        name = label_names.get(label_id, f"Class-{label_id}")
        count = dist[label_id]
        pct = 100.0 * count / total if total > 0 else 0
        parts.append(f"{name}: {count} ({pct:.1f}%)")
    print(f"  {prefix}: {' | '.join(parts)}")


def _save_split_config(splits: Dict, output_path: str):
    """Save split configuration to JSON for reproducibility."""
    config = {}
    for split_name, samples in splits.items():
        config[split_name] = [
            {"image": s["image_path"], "label": s["label"], "source": s["source"]}
            for s in samples
        ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Split config saved to: {output_path}")


def _save_kfold_config(folds: List[Dict], output_path: str):
    """Save K-Fold configuration to JSON."""
    config = {}
    for i, fold in enumerate(folds):
        config[f"fold_{i+1}"] = {
            split_name: [
                {"image": s["image_path"], "label": s["label"], "source": s["source"]}
                for s in samples
            ]
            for split_name, samples in fold.items()
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  K-Fold config saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    dataset_dirs = [
        "datasets/BUSI",
        "datasets/BUS_BRA",
        "datasets/UDIAT",
        "datasets/STU",
    ]

    # Create a standard train/val/test split
    splits = create_train_val_test_split(
        dataset_dirs=dataset_dirs,
        test_size=0.15,
        val_size=0.15,
        output_path="datasets/split_config.json",
    )

    # Create 5-fold cross-validation splits
    folds = create_kfold_splits(
        dataset_dirs=dataset_dirs,
        n_folds=5,
        output_path="datasets/kfold_config.json",
    )
