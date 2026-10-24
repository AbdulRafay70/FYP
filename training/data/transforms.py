"""
Albumentations Transform Pipelines for Breast Ultrasound Images.

Provides medical-specific augmentation pipelines for:
1. Segmentation training (image + mask paired transforms)
2. Classification training (image-only transforms)
3. Inference/validation (minimal transforms — normalize only)

Key medical augmentations:
- CLAHE: Enhances tissue boundary contrast
- ElasticTransform: Simulates tissue deformation
- GridDistortion: Simulates probe pressure variations
- GaussNoise: Simulates ultrasound speckle noise
"""

import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2


def get_segmentation_train_transforms(img_size: int = 256) -> A.Compose:
    """
    Training augmentation pipeline for segmentation (Attention U-Net).

    Transforms are applied identically to both the image and mask.
    Uses medical-specific augmentations to improve model robustness
    without destroying spatial tissue structure.

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Spatial transforms (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-15, 15), border_mode=cv2.BORDER_CONSTANT, p=0.4),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.2,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.2,
            p=0.2,
        ),

        # Intensity transforms (image only, not mask)
        A.OneOf([
            A.GaussNoise(std_range=(0.012, 0.025), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.3,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        # Normalize and convert to tensor
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])


def get_segmentation_val_transforms(img_size: int = 256) -> A.Compose:
    """
    Validation/inference transforms for segmentation.
    Only resize and normalize — no augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485], std=[0.229]),
        ToTensorV2(),
    ])


def get_classification_train_transforms(img_size: int = 224) -> A.Compose:
    """
    Training augmentation pipeline for classification (ResNet50 / EfficientNet).

    More aggressive augmentations since classifiers benefit from
    greater input diversity. Includes cutout/coarse dropout to
    force the model to learn from multiple regions.

    Args:
        img_size: Target image size (square)

    Returns:
        Albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),

        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(1-0.2, 1+0.2), rotate=(-20, 20), border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            p=0.2,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            p=0.2,
        ),
        A.OpticalDistortion(
            distort_limit=0.05,
            p=0.15,
        ),

        # Intensity transforms
        A.OneOf([
            A.GaussNoise(std_range=(0.012, 0.028), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=False, p=0.2),  # speckle
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.15),

        # Regularization — force model to learn from partial information
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            fill=0,
            p=0.3,
        ),

        # Normalize and convert to tensor
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_classification_val_transforms(img_size: int = 224) -> A.Compose:
    """
    Validation/inference transforms for classification.
    Only resize and normalize — no augmentation.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_inference_transforms(img_size: int = 224) -> A.Compose:
    """
    Minimal transforms for inference (production use).
    Used by the FastAPI backend when processing uploaded images.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
