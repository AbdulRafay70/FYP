"""
Custom PyTorch Dataset for Breast Ultrasound Images.

Supports loading from multiple dataset sources (BUSI, BUS-BRA, UDIAT, STU)
with ground truth segmentation masks for both classification and segmentation tasks.

Handles:
- Multi-dataset loading with unified label mapping
- Paired image + mask loading for segmentation
- Image-only loading for classification (with label)
- On-the-fly Albumentations augmentation
"""

import os
import glob
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# Label mapping: string → integer
LABEL_MAP = {
    "benign": 0,
    "malignant": 1,
    "normal": 0,  # Normal cases treated as benign (non-cancerous)
}

LABEL_NAMES = {0: "Benign", 1: "Malignant"}


class BreastUltrasoundDataset(Dataset):
    """
    Unified dataset for breast ultrasound images from multiple sources.

    Supports two modes:
        1. 'segmentation': Returns (image, mask) pairs for U-Net training
        2. 'classification': Returns (image, label) pairs for classifier training

    Directory structure expected:
        dataset_root/
        ├── benign/
        │   ├── image1.png
        │   ├── image1_mask.png
        │   ├── image2.png
        │   └── image2_mask.png
        ├── malignant/
        │   ├── image1.png
        │   ├── image1_mask.png
        │   └── ...
        └── normal/  (optional)
            └── ...
    """

    def __init__(
        self,
        root_dirs: List[str],
        mode: str = "classification",
        transform: Optional[Callable] = None,
        img_size: Tuple[int, int] = (224, 224),
        include_normal: bool = False,
    ):
        """
        Args:
            root_dirs: List of dataset root directories to combine
            mode: 'classification' or 'segmentation'
            transform: Albumentations transform pipeline
            img_size: Target image size (H, W)
            include_normal: Whether to include normal (non-tumor) cases
        """
        assert mode in ("classification", "segmentation"), \
            f"Mode must be 'classification' or 'segmentation', got '{mode}'"

        self.mode = mode
        self.transform = transform
        self.img_size = img_size

        # Collect all image paths and labels
        self.samples: List[Dict] = []
        self._load_datasets(root_dirs, include_normal)

        print(
            f"[Dataset] Loaded {len(self.samples)} samples "
            f"({mode} mode) from {len(root_dirs)} source(s)"
        )
        self._print_class_distribution()

    def _load_datasets(self, root_dirs: List[str], include_normal: bool):
        """Scan all dataset directories and collect image paths."""
        for root_dir in root_dirs:
            root_path = Path(root_dir)
            if not root_path.exists():
                print(f"[Warning] Dataset directory not found: {root_dir}")
                continue

            for class_name in ["benign", "malignant", "normal"]:
                class_dir = root_path / class_name
                if not class_dir.exists():
                    # Search recursively for the class folder within the root path
                    search_pattern = os.path.join(root_path, "**", class_name)
                    found_dirs = glob.glob(search_pattern, recursive=True)
                    if found_dirs:
                        class_dir = Path(found_dirs[0])
                    else:
                        continue

                if class_name == "normal" and not include_normal:
                    continue

                label = LABEL_MAP[class_name]

                # Find all images (exclude masks)
                image_files = []
                for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                    image_files.extend(glob.glob(str(class_dir / ext)))

                # Filter out mask files
                image_files = [
                    f for f in image_files
                    if "_mask" not in os.path.basename(f).lower()
                ]

                for img_path in sorted(image_files):
                    sample = {
                        "image_path": img_path,
                        "label": label,
                        "class_name": class_name,
                        "source": root_path.name,
                    }

                    # Find corresponding mask if in segmentation mode
                    if self.mode == "segmentation":
                        mask_path = self._find_mask(img_path)
                        if mask_path is None:
                            continue  # Skip images without masks
                        sample["mask_path"] = mask_path

                    self.samples.append(sample)

    def _find_mask(self, image_path: str) -> Optional[str]:
        """
        Find the segmentation mask corresponding to an image.

        Searches for common mask naming conventions:
            - image_mask.png
            - image_mask_1.png
            - image_segmentation.png
        """
        base = os.path.splitext(image_path)[0]
        ext = os.path.splitext(image_path)[1]

        # Common mask naming patterns
        candidates = [
            f"{base}_mask{ext}",
            f"{base}_mask_1{ext}",
            f"{base}_mask1{ext}",
            f"{base}_segmentation{ext}",
            f"{base}_seg{ext}",
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return None

    def _print_class_distribution(self):
        """Print class distribution statistics."""
        from collections import Counter
        labels = [s["label"] for s in self.samples]
        dist = Counter(labels)
        total = len(labels)
        for label_id, count in sorted(dist.items()):
            name = LABEL_NAMES.get(label_id, f"Class {label_id}")
            pct = 100.0 * count / total if total > 0 else 0
            print(f"  {name}: {count} ({pct:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image: {sample['image_path']}")

        # Resize
        image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        if self.mode == "segmentation":
            return self._get_segmentation_item(image, sample)
        else:
            return self._get_classification_item(image, sample)

    def _get_segmentation_item(
        self, image: np.ndarray, sample: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image + mask pair for segmentation training."""
        # Load mask
        mask = cv2.imread(sample["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {sample['mask_path']}")

        mask = cv2.resize(mask, (self.img_size[1], self.img_size[0]))

        # Binarize mask (threshold at 128)
        mask = (mask > 128).astype(np.float32)

        # Apply augmentation (to both image and mask simultaneously)
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

            # If transform returns tensor, we're done
            if isinstance(image, torch.Tensor):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)  # Add channel dim: (1, H, W)
                return image, mask

        # Convert to tensor if not already
        image = torch.from_numpy(image).float()
        if image.ndim == 2:
            image = image.unsqueeze(0)  # (1, H, W)
        image = image / 255.0  # Normalize to [0, 1]

        mask = torch.from_numpy(mask).float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)  # (1, H, W)

        return image, mask

    def _get_classification_item(
        self, image: np.ndarray, sample: Dict
    ) -> Tuple[torch.Tensor, int]:
        """Load image + label for classification training."""
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

            if isinstance(image, torch.Tensor):
                # Duplicate grayscale to 3 channels for pretrained backbones
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                return image, sample["label"]

        # Convert to tensor
        image = torch.from_numpy(image).float()
        if image.ndim == 2:
            image = image.unsqueeze(0)
        image = image / 255.0

        # Duplicate to 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, sample["label"]

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse frequency class weights for handling class imbalance.

        Returns:
            Tensor of shape (num_classes,) with weights inversely proportional
            to class frequency.
        """
        from collections import Counter
        labels = [s["label"] for s in self.samples]
        dist = Counter(labels)
        total = len(labels)
        num_classes = max(dist.keys()) + 1

        weights = torch.zeros(num_classes)
        for cls_id in range(num_classes):
            count = dist.get(cls_id, 1)
            weights[cls_id] = total / (num_classes * count)

        return weights

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a specific sample (useful for debugging)."""
        return self.samples[idx]
