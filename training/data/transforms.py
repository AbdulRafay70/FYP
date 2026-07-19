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
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=0,  # Constant padding
            p=0.4,
        ),
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
            A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
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
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=20,
            border_mode=0,
            p=0.5,
        ),
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
            shift_limit=0.05,
            p=0.15,
        ),

        # Intensity transforms
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.4,
        ),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),

        # Regularization — force model to learn from partial information
        A.CoarseDropout(
            max_holes=8,
            max_height=img_size // 10,
            max_width=img_size // 10,
            fill_value=0,
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
