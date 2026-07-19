"""
UltraXAI Backend Configuration.
"""

import os
import torch
from pathlib import Path


class Settings:
    """Application settings."""

    # Server
    PORT: int = 8000
    HOST: str = "0.0.0.0"

    # CORS
    ALLOWED_ORIGINS: list = [
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # React dev server
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]

    # Paths
    PROJECT_ROOT: str = str(Path(__file__).parent.parent.parent)
    WEIGHTS_DIR: str = os.path.join(PROJECT_ROOT, "backend", "weights")
    UPLOADS_DIR: str = os.path.join(PROJECT_ROOT, "backend", "uploads")

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model configurations
    YOLO_WEIGHTS: str = "yolov8m_seg_breast.pt"
    UNET_WEIGHTS: str = "attention_unet_breast.pth"
    RESNET_WEIGHTS: str = "resnet50_breast.pth"
    EFFICIENT_WEIGHTS: str = "efficientnetv2_breast.pth"

    # Image processing
    YOLO_IMG_SIZE: int = 640
    CLASSIFIER_IMG_SIZE: int = 224
    UNET_IMG_SIZE: int = 256

    # Thresholds
    CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_CONF_THRESHOLD: float = 0.25
    HIGH_TRUST_THRESHOLD: float = 0.90
    MEDIUM_TRUST_THRESHOLD: float = 0.70

    # Upload limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set = {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}


settings = Settings()

# Create necessary directories
os.makedirs(settings.WEIGHTS_DIR, exist_ok=True)
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
