"""
Shared ROI-cropping helper.

Used by BOTH classifier training (crop from the ground-truth mask) and the
backend inference pipeline (crop from the YOLO box or U-Net mask), so the
classifiers see the same kind of input at training and inference time.
"""

from typing import Optional, Tuple

import numpy as np

ROI_PADDING = 0.15  # 15% padding around the lesion box on each side


def pad_box(
    box: Tuple[int, int, int, int],
    image_shape: Tuple[int, ...],
    padding: float = ROI_PADDING,
) -> Tuple[int, int, int, int]:
    """Expand an (x1, y1, x2, y2) box by `padding` of its size, clipped to the image."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = image_shape[:2]
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)


def mask_to_box(mask: np.ndarray, threshold: float = 0.5, min_pixels: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """Tight bounding box of the foreground of a mask, or None if it is (nearly) empty."""
    ys, xs = np.where(mask > threshold)
    if len(ys) < min_pixels:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_roi_from_mask(image: np.ndarray, mask: np.ndarray, padding: float = ROI_PADDING) -> np.ndarray:
    """Crop the lesion ROI defined by a mask (+padding). Returns the full image if the mask is empty."""
    box = mask_to_box(mask, threshold=127 if mask.max() > 1 else 0.5)
    if box is None:
        return image
    x1, y1, x2, y2 = pad_box(box, image.shape, padding)
    return image[y1:y2, x1:x2]
