"""
Pipeline Orchestrator — Orchestrates the 4-Phase AI Diagnostic Pipeline.

Phase 1: Detection (YOLOv8-Seg) — Locate suspicious regions
Phase 2: Segmentation (Attention U-Net) — Precise tumor masking
Phase 3: Classification (ResNet50 + EfficientNetV2-S, weighted soft voting 0.40/0.60) — Benign vs Malignant
Phase 4: Explainability (ensemble Grad-CAM++) — Visual explanation heatmaps
"""

import os
import time
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import torch

# Add training module to path for model imports
training_path = str(Path(__file__).parent.parent.parent.parent / "training")
if training_path not in sys.path:
    sys.path.insert(0, training_path)

from app.config import settings
from data.roi import pad_box, mask_to_box


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
        """Return the loading status of each model (True only if its checkpoint loaded)."""
        return dict(self.models_loaded)

    def run_full_pipeline(self, image: np.ndarray, progress: Optional[Callable] = None) -> Dict:
        """
        Run the complete 4-phase diagnostic pipeline.

        Inference preprocessing (identical to the validation transforms used in
        training): grayscale decode → model-specific resize (YOLO 640, U-Net 256,
        classifiers 224, bilinear) → mean/std normalisation. CLAHE / denoising
        from app.preprocessing.image_pipeline are NOT applied, because the
        models were not trained on CLAHE-enhanced or denoised images.

        Args:
            image: Input grayscale ultrasound image (H, W)
            progress: optional callback(phase:int, name:str, status:str, time_ms:float|None)

        Returns:
            Dictionary with all results, visualizations, and metadata
        """
        def report(phase, name, status, time_ms=None):
            if progress is not None:
                progress(phase, name, status, time_ms)

        results = {
            "phases": {},
            "tumor_detected": False,
            "prediction": "unknown",
            "confidence": 0.0,
            "trust_score": "UNKNOWN",
        }

        # ---- Phase 1: Detection ----
        report(1, "Detection", "processing")
        t0 = time.time()
        detection_result = self._run_detection(image)
        results["phases"]["detection"] = {
            "status": "complete" if detection_result["detected"] else "no_detection_using_full_image",
            "time_ms": round((time.time() - t0) * 1000, 2),
        }
        report(1, "Detection", "complete", results["phases"]["detection"]["time_ms"])

        if detection_result["detected"]:
            results["tumor_detected"] = True
            results["bounding_box"] = detection_result["bbox"]
            results["detection_overlay"] = detection_result["overlay"]
            roi = detection_result["roi"]
        else:
            roi = image

        # ---- Phase 2: Segmentation ----
        report(2, "Segmentation", "processing")
        t0 = time.time()
        # U-Net was trained on whole images (256x256), so it is run on the whole image
        seg_result = self._run_segmentation(image)
        results["phases"]["segmentation"] = {
            "status": "complete" if self.unet_model is not None else "skipped_model_not_loaded",
            "time_ms": round((time.time() - t0) * 1000, 2),
        }
        report(2, "Segmentation", "complete", results["phases"]["segmentation"]["time_ms"])
        results["segmentation_mask"] = seg_result["mask_visualization"]
        segmented_roi = roi  # classifiers receive the unmasked ROI crop

        mask_pixels = 0
        if self.unet_model is not None and seg_result["mask"] is not None:
            mask_pixels = int((seg_result["mask"] > 0.5).sum())

        # Fallback: YOLO found nothing but U-Net segmented a lesion → crop the ROI from the mask
        if not detection_result["detected"] and mask_pixels > 50:
            box = pad_box(mask_to_box(seg_result["mask"]), image.shape)
            x1, y1, x2, y2 = box
            segmented_roi = image[y1:y2, x1:x2]
            results["tumor_detected"] = True
            results["bounding_box"] = [int(v) for v in box]
            img_3ch = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR) if image.ndim == 2 or image.shape[-1] == 1 else image.copy()
            cv2.rectangle(img_3ch, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_3ch, "Lesion (U-Net fallback)", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            results["detection_overlay"] = img_3ch

        # ---- Phase 3: Classification ----
        # The classifiers are binary (Benign vs Malignant) and were trained only on
        # lesion ROIs — never on Normal images. If neither YOLO nor U-Net finds a
        # lesion, the classifier is not run and the result is reported as
        # "No lesion detected" (inconclusive), NOT as a confident Benign diagnosis.
        no_lesion = (
            not results["tumor_detected"]
            and self.unet_model is not None
            and mask_pixels <= 50
        )
        report(3, "Classification", "processing")
        t0 = time.time()
        if no_lesion:
            cls_result = {
                "prediction": "No lesion detected",
                "confidence": 0.0,
                "trust_score": "N/A",
                "message": ("Neither the detector nor the segmentation model found a lesion. "
                            "The Benign/Malignant classifier was not applied. This is not a "
                            "diagnosis of normal tissue; radiologist review is required."),
            }
        else:
            cls_result = self._run_classification(segmented_roi)
        results["phases"]["classification"] = {
            "status": "skipped_no_lesion" if no_lesion else "complete",
            "time_ms": round((time.time() - t0) * 1000, 2),
        }
        report(3, "Classification", "complete", results["phases"]["classification"]["time_ms"])
        results.update(cls_result)

        # ---- Phase 4: Explainability ----
        report(4, "Explainability", "processing")
        t0 = time.time()
        if no_lesion or "prediction_index" not in cls_result:
            xai_result = {"gradcam_heatmap": None, "gradcam_overlay": None, "explained_model": None}
            xai_status = "skipped"
        else:
            xai_result = self._run_explainability(segmented_roi, segmented_roi,
                                                  target_class=cls_result["prediction_index"])
            xai_status = "complete" if xai_result["gradcam_overlay"] is not None else "failed"
        results["phases"]["explainability"] = {
            "status": xai_status,
            "time_ms": round((time.time() - t0) * 1000, 2),
        }
        report(4, "Explainability", xai_status, results["phases"]["explainability"]["time_ms"])
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

        # Add padding (15%) — same geometry as the classifier training crops
        x1, y1, x2, y2 = pad_box((x1, y1, x2, y2), image.shape)

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

        # The mask is NOT multiplied into the image: the classifiers were trained on
        # unmasked ROI crops (GT-mask box + 15% padding), so they receive the raw ROI.
        # The mask is used to (a) re-locate the ROI when YOLO finds nothing and
        # (b) decide "no lesion found", and is returned for display.
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
        
        # Ensure 3 channels for classification (same GRAY2RGB as training)
        transformed = transform(image=self._to_rgb(segmented_roi))
        tensor = transformed["image"].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            results = self.ensemble(tensor)

        pred_idx = int(results["prediction"].item())
        pred_label = "Malignant" if pred_idx == 1 else "Benign"
        confidence = results["confidence"].item()
        trust = results["trust_score"][0]

        return {
            "prediction": pred_label,
            "prediction_index": pred_idx,
            "confidence": confidence,
            "trust_score": trust,
            "resnet_prediction": "Malignant" if results["resnet_logits"].argmax(1).item() == 1 else "Benign",
            "efficient_prediction": "Malignant" if results["efficient_logits"].argmax(1).item() == 1 else "Benign",
            "resnet_confidence": float(torch.softmax(results["resnet_logits"], 1).max(1).values.item()),
            "efficient_confidence": float(torch.softmax(results["efficient_logits"], 1).max(1).values.item()),
            "resnet_weight": self.ensemble.resnet_weight,
            "efficient_weight": self.ensemble.efficient_weight,
        }

    def _run_explainability(self, segmented_roi: np.ndarray, original_roi: np.ndarray,
                            target_class: Optional[int] = None) -> Dict:
        """
        Phase 4: Grad-CAM++ explanation of the ENSEMBLE prediction.

        Grad-CAM++ maps are computed for ResNet50 (layer4[-1]) and EfficientNetV2-S
        (features[-1]) for the ensemble's predicted class and fused with the
        ensemble weights (0.40 / 0.60).
        """
        if self.ensemble is None:
            return {"gradcam_heatmap": None, "gradcam_overlay": None, "explained_model": None}

        try:
            from utils.explainability import generate_gradcam_for_ensemble
            from data.transforms import get_classification_val_transforms

            transform = get_classification_val_transforms(settings.CLASSIFIER_IMG_SIZE)
            tensor = transform(image=self._to_rgb(segmented_roi))["image"].unsqueeze(0).to(self.device)

            maps = generate_gradcam_for_ensemble(
                self.ensemble.resnet, self.ensemble.efficient, tensor, original_roi,
                target_class=target_class,
                resnet_weight=self.ensemble.resnet_weight,
                efficient_weight=self.ensemble.efficient_weight,
            )
            return {
                "gradcam_heatmap": (maps["combined_heatmap"] * 255).astype(np.uint8),
                "gradcam_overlay": maps["combined_overlay"],
                "explained_model": (
                    f"Ensemble Grad-CAM++ ({self.ensemble.resnet_weight:.2f} x ResNet50 layer4 + "
                    f"{self.ensemble.efficient_weight:.2f} x EfficientNetV2-S features[-1])"
                ),
            }

        except Exception as e:
            print(f"  ⚠ Grad-CAM generation failed: {e}")
            return {"gradcam_heatmap": None, "gradcam_overlay": None, "explained_model": None}

    @staticmethod
    def _to_rgb(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return img
        return cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
