"""
Pipeline Orchestrator — Orchestrates the 4-Phase AI Diagnostic Pipeline.

Phase 1: Detection (YOLOv8-Seg) — Locate suspicious regions
Phase 2: Segmentation (Attention U-Net) — Precise tumor masking
Phase 3: Classification (ResNet50 + EfficientNetV2B0) — Benign vs Malignant
Phase 4: Explainability (Grad-CAM++) — Visual explanation heatmaps
"""

import os
import time
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch

# Add training module to path for model imports
training_path = str(Path(__file__).parent.parent.parent.parent / "training")
if training_path not in sys.path:
    sys.path.insert(0, training_path)

from app.config import settings
from app.preprocessing.image_pipeline import preprocess_image, crop_roi_from_bbox


class PipelineOrchestrator:
    """
    Orchestrates the complete AI diagnostic pipeline.

    Manages model loading, inference sequencing, and result aggregation
    across all 4 phases.
    """

    def __init__(self, weights_dir: str, device: str = "cpu"):
        self.weights_dir = weights_dir
        self.device = torch.device(device)
        self.models_loaded = {
            "yolo": False,
            "attention_unet": False,
            "resnet50": False,
            "efficientnetv2": False,
        }

        self._load_models()

    def _load_models(self):
        """Load all available model weights."""

        # Phase 1: YOLOv8-Seg
        yolo_path = os.path.join(self.weights_dir, settings.YOLO_WEIGHTS)
        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_path)
                self.models_loaded["yolo"] = True
                print(f"  ✓ YOLOv8-Seg loaded from {yolo_path}")
            except Exception as e:
                print(f"  ✗ YOLOv8-Seg failed: {e}")
                self.yolo_model = None
        else:
            print(f"  ⚠ YOLOv8-Seg weights not found: {yolo_path}")
            self.yolo_model = None

        # Phase 2: Attention U-Net
        unet_path = os.path.join(self.weights_dir, settings.UNET_WEIGHTS)
        if os.path.exists(unet_path):
            try:
                from models.attention_unet import build_attention_unet
                self.unet_model = build_attention_unet(
                    in_channels=1, out_channels=1, pretrained_path=unet_path
                )
                self.unet_model.to(self.device).eval()
                self.models_loaded["attention_unet"] = True
                print(f"  ✓ Attention U-Net loaded from {unet_path}")
            except Exception as e:
                print(f"  ✗ Attention U-Net failed: {e}")
                self.unet_model = None
        else:
            print(f"  ⚠ Attention U-Net weights not found: {unet_path}")
            self.unet_model = None

        # Phase 3: Classification Ensemble
        resnet_path = os.path.join(self.weights_dir, settings.RESNET_WEIGHTS)
        efficient_path = os.path.join(self.weights_dir, settings.EFFICIENT_WEIGHTS)

        if os.path.exists(resnet_path) and os.path.exists(efficient_path):
            try:
                from models.classifier_heads import build_ensemble
                self.ensemble = build_ensemble(
                    num_classes=2,
                    resnet_path=resnet_path,
                    efficient_path=efficient_path,
                )
                self.ensemble.to(self.device).eval()
                self.models_loaded["resnet50"] = True
                self.models_loaded["efficientnetv2"] = True
                print(f"  ✓ Ensemble classifier loaded")
            except Exception as e:
                print(f"  ✗ Ensemble failed: {e}")
                self.ensemble = None
        else:
            print(f"  ⚠ Classifier weights not found")
            self.ensemble = None

    def get_model_status(self) -> Dict[str, bool]:
        """Return the loading status of each model."""
        return self.models_loaded

    def run_full_pipeline(self, image: np.ndarray) -> Dict:
        """
        Run the complete 4-phase diagnostic pipeline.

        Args:
            image: Input grayscale ultrasound image (H, W)

        Returns:
            Dictionary with all results, visualizations, and metadata
        """
        results = {
            "phases": {},
            "tumor_detected": False,
            "prediction": "unknown",
            "confidence": 0.0,
            "trust_score": "UNKNOWN",
        }

        # ---- Phase 1: Detection ----
        phase1_start = time.time()
        detection_result = self._run_detection(image)
        results["phases"]["detection"] = {
            "status": "complete" if detection_result["detected"] else "no_tumor",
            "time_ms": round((time.time() - phase1_start) * 1000, 2),
        }

        if detection_result["detected"]:
            results["tumor_detected"] = True
            results["bounding_box"] = detection_result["bbox"]
            results["detection_overlay"] = detection_result["overlay"]
            roi = detection_result["roi"]
        else:
            # No tumor detected — still classify the full image
            roi = image
            results["phases"]["detection"]["status"] = "no_tumor_using_full_image"

        # ---- Phase 2: Segmentation ----
        phase2_start = time.time()
        seg_result = self._run_segmentation(roi)
        results["phases"]["segmentation"] = {
            "status": "complete",
            "time_ms": round((time.time() - phase2_start) * 1000, 2),
        }
        results["segmentation_mask"] = seg_result["mask_visualization"]
        segmented_roi = seg_result["segmented_roi"]
        
        # Fallback: If YOLO missed the tumor but U-Net segmented one, crop using U-Net mask coordinates
        if not detection_result["detected"] and seg_result["mask"] is not None:
            mask_binary = (seg_result["mask"] > 0.5).astype(np.uint8)
            y_indices, x_indices = np.where(mask_binary > 0)
            if len(y_indices) > 50:  # Minimum pixel threshold to avoid noise
                y1, y2 = y_indices.min(), y_indices.max()
                x1, x2 = x_indices.min(), x_indices.max()
                
                # Add 15% padding
                h_img, w_img = image.shape[:2]
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                x1_pad = max(0, x1 - pad_x)
                y1_pad = max(0, y1 - pad_y)
                x2_pad = min(w_img, x2 + pad_x)
                y2_pad = min(h_img, y2 + pad_y)
                
                # Crop the ROI using the U-Net mask boundary box
                segmented_roi = image[y1_pad:y2_pad, x1_pad:x2_pad]
                results["tumor_detected"] = True
                results["bounding_box"] = [int(x1_pad), int(y1_pad), int(x2_pad), int(y2_pad)]
                
                # Re-create detection overlay using U-Net bounding box
                if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                    img_3ch = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
                else:
                    img_3ch = image.copy()
                cv2.rectangle(img_3ch, (x1_pad, y1_pad), (x2_pad, y2_pad), (0, 255, 0), 2)
                cv2.putText(img_3ch, "Tumor Detected (U-Net Fallback)", (x1_pad, y1_pad - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                results["detection_overlay"] = img_3ch

        # ---- Phase 3: Classification ----
        phase3_start = time.time()
        
        # Safeguard: If the segmentation mask is empty (no tumor pixels found), bypass classifiers
        if seg_result["mask"] is not None and seg_result["mask"].sum() < 10:
            cls_result = {
                "prediction": "Benign",
                "confidence": 1.0,
                "trust_score": "HIGH",
                "resnet_prediction": "Benign",
                "efficient_prediction": "Benign",
                "resnet_confidence": 1.0,
                "efficient_confidence": 1.0,
                "message": "No tumor pixels segmented. Diagnosed as Normal / Benign."
            }
        else:
            cls_result = self._run_classification(segmented_roi)
            
        results["phases"]["classification"] = {
            "status": "complete",
            "time_ms": round((time.time() - phase3_start) * 1000, 2),
        }
        results.update(cls_result)

        # ---- Phase 4: Explainability ----
        phase4_start = time.time()
        xai_result = self._run_explainability(segmented_roi, roi)
        results["phases"]["explainability"] = {
            "status": "complete",
            "time_ms": round((time.time() - phase4_start) * 1000, 2),
        }
        results.update(xai_result)

        return results

    def run_track_a(self, image: np.ndarray) -> Dict:
        """Run Track A only (same as full pipeline for now)."""
        return self.run_full_pipeline(image)

    def run_track_b(self, image: np.ndarray) -> Dict:
        """Run Track B (SOTA pipeline). Placeholder until MedSAM/Swin-V2 are trained."""
        return {
            "prediction": "Track B not yet trained",
            "confidence": 0.0,
            "trust_score": "N/A",
            "message": "Train MedSAM and Swin-V2/ConvNeXt-V2 models first.",
        }

    def _run_detection(self, image: np.ndarray) -> Dict:
        """Phase 1: Run YOLO detection."""
        if self.yolo_model is None:
            # Fallback: return the full image as ROI
            return {"detected": False, "roi": image, "bbox": None, "overlay": None}

        # Prepare input (YOLO expects 3-channel BGR)
        if image.ndim == 2:
            img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            img_3ch = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 3:
            img_3ch = image
        else:
            img_3ch = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)

        # Run YOLO inference
        results = self.yolo_model(img_3ch, conf=settings.YOLO_CONF_THRESHOLD, verbose=False)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return {"detected": False, "roi": image, "bbox": None, "overlay": None}

        # Get the highest confidence detection
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax()
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)
        x1, y1, x2, y2 = bbox

        # Add padding (15%)
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.15)
        pad_y = int((y2 - y1) * 0.15)
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        # Crop ROI
        roi = image[y1:y2, x1:x2]

        # Create detection overlay
        overlay = img_3ch.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, "Tumor Detected", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return {
            "detected": True,
            "roi": roi,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "overlay": overlay,
        }

    def _run_segmentation(self, roi: np.ndarray) -> Dict:
        """Phase 2: Run Attention U-Net segmentation."""
        if self.unet_model is None:
            # Fallback: return the ROI as-is
            return {
                "mask": np.ones_like(roi, dtype=np.float32),
                "mask_visualization": roi,
                "segmented_roi": roi,
            }

        # Preprocess for U-Net using correct validation transforms (expects grayscale 1-channel)
        from data.transforms import get_segmentation_val_transforms
        h, w = roi.shape[:2]
        if roi.ndim == 3 and roi.shape[2] == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        elif roi.ndim == 3 and roi.shape[2] == 1:
            roi_gray = roi.squeeze(-1)
        else:
            roi_gray = roi

        transform = get_segmentation_val_transforms(settings.UNET_IMG_SIZE)
        transformed = transform(image=roi_gray)
        tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            mask_pred = self.unet_model(tensor)

        # Post-process mask
        mask = mask_pred.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (w, h))
        mask_binary = (mask > 0.5).astype(np.uint8)

        # Create mask visualization
        mask_vis = (mask_binary * 255).astype(np.uint8)
        mask_colored = cv2.applyColorMap(mask_vis, cv2.COLORMAP_OCEAN)

        # Apply mask to ROI (we pass the raw cropped ROI to classifiers to match training distribution)
        segmented = roi

        return {
            "mask": mask,
            "mask_visualization": mask_colored,
            "segmented_roi": segmented,
        }

    def _run_classification(self, segmented_roi: np.ndarray) -> Dict:
        """Phase 3: Run ensemble classification."""
        if self.ensemble is None:
            return {
                "prediction": "models_not_loaded",
                "confidence": 0.0,
                "trust_score": "N/A",
            }

        # Preprocess for classifiers using correct validation transforms
        from data.transforms import get_classification_val_transforms
        transform = get_classification_val_transforms(settings.CLASSIFIER_IMG_SIZE)
        
        # Ensure 3 channels for classification
        if segmented_roi.ndim == 2 or (segmented_roi.ndim == 3 and segmented_roi.shape[2] == 1):
            segmented_roi_rgb = cv2.cvtColor(segmented_roi.squeeze(), cv2.COLOR_GRAY2RGB)
        elif segmented_roi.ndim == 3 and segmented_roi.shape[2] == 3:
            segmented_roi_rgb = segmented_roi
        else:
            segmented_roi_rgb = cv2.cvtColor(segmented_roi.squeeze(), cv2.COLOR_GRAY2RGB)
            
        transformed = transform(image=segmented_roi_rgb)
        tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            results = self.ensemble(tensor)

        pred_label = "Malignant" if results["prediction"].item() == 1 else "Benign"
        confidence = results["confidence"].item()
        trust = results["trust_score"][0]

        return {
            "prediction": pred_label,
            "confidence": confidence,
            "trust_score": trust,
            "resnet_prediction": "Malignant" if results["resnet_logits"].argmax(1).item() == 1 else "Benign",
            "efficient_prediction": "Malignant" if results["efficient_logits"].argmax(1).item() == 1 else "Benign",
            "resnet_confidence": float(torch.softmax(results["resnet_logits"], 1).max(1).values.item()),
            "efficient_confidence": float(torch.softmax(results["efficient_logits"], 1).max(1).values.item()),
            "resnet_weight": self.ensemble.resnet_weight,
            "efficient_weight": self.ensemble.efficient_weight,
        }

    def _run_explainability(self, segmented_roi: np.ndarray, original_roi: np.ndarray) -> Dict:
        """Phase 4: Generate Grad-CAM++ heatmaps."""
        if self.ensemble is None:
            return {"gradcam_heatmap": None, "gradcam_overlay": None}

        try:
            from utils.explainability import GradCAMPlusPlus

            # Prepare input tensor using correct validation transforms
            from data.transforms import get_classification_val_transforms
            transform = get_classification_val_transforms(settings.CLASSIFIER_IMG_SIZE)
            
            # Ensure 3 channels for classification
            if segmented_roi.ndim == 2 or (segmented_roi.ndim == 3 and segmented_roi.shape[2] == 1):
                segmented_roi_rgb = cv2.cvtColor(segmented_roi.squeeze(), cv2.COLOR_GRAY2RGB)
            elif segmented_roi.ndim == 3 and segmented_roi.shape[2] == 3:
                segmented_roi_rgb = segmented_roi
            else:
                segmented_roi_rgb = cv2.cvtColor(segmented_roi.squeeze(), cv2.COLOR_GRAY2RGB)
                
            transformed = transform(image=segmented_roi_rgb)
            tensor = transformed["image"].unsqueeze(0).to(self.device)

            # Generate Grad-CAM for ResNet50
            gradcam = GradCAMPlusPlus(
                self.ensemble.resnet,
                self.ensemble.resnet.get_last_conv_layer(),
            )
            heatmap = gradcam.generate(tensor)
            overlay = GradCAMPlusPlus.overlay_heatmap(heatmap, original_roi)

            return {
                "gradcam_heatmap": (heatmap * 255).astype(np.uint8),
                "gradcam_overlay": overlay,
            }

        except Exception as e:
            print(f"  ⚠ Grad-CAM generation failed: {e}")
            return {"gradcam_heatmap": None, "gradcam_overlay": None}
