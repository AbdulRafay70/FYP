"""
Image Preprocessing Pipeline.

Implements the full preprocessing chain:
1. Format detection & loading
2. Grayscale conversion
3. CLAHE contrast enhancement
4. Speckle noise reduction
5. Resize to target resolution
6. Pixel normalization [0, 1]
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (640, 640),
    apply_clahe: bool = True,
    apply_denoise: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for ultrasound images.

    Args:
        image: Raw input image (grayscale or color)
        target_size: Target (width, height)
        apply_clahe: Whether to apply CLAHE enhancement
        apply_denoise: Whether to apply speckle denoising

    Returns:
        Preprocessed grayscale image, float32, normalized [0, 1]
    """
    # Ensure grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE — Contrast Limited Adaptive Histogram Equalization
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)

    # Speckle noise reduction
    if apply_denoise:
        image = cv2.fastNlMeansDenoising(
            image, h=10, templateWindowSize=7, searchWindowSize=21
        )

    # Resize
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image


def crop_roi_from_bbox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.15,
) -> np.ndarray:
    """
    Crop a region of interest from an image using a bounding box.

    Args:
        image: Source image (H, W) or (H, W, 3)
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        padding: Fractional padding around the bbox

    Returns:
        Cropped ROI image
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    # Add padding
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return image[y1:y2, x1:x2]


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_value: int = 0,
) -> np.ndarray:
    """
    Apply a binary segmentation mask to an image.

    Args:
        image: Input image
        mask: Binary mask (0 or 1)
        background_value: Value for masked-out regions

    Returns:
        Masked image
    """
    mask_binary = (mask > 0.5).astype(np.uint8)

    if image.ndim == 2:
        result = image * mask_binary
    else:
        result = image * mask_binary[:, :, np.newaxis]

    # Set background
    if background_value != 0:
        bg = np.full_like(image, background_value)
        if image.ndim == 2:
            result = np.where(mask_binary, image, bg)
        else:
            result = np.where(mask_binary[:, :, np.newaxis], image, bg)

    return result
