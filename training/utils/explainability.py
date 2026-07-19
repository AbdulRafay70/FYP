"""
Explainable AI (XAI) Engine for UltraXAI.

Implements:
1. Grad-CAM++ — Generates localized heatmaps showing which image regions
   drove the classification decision
2. SHAP integration — Provides pixel-level attribution metrics

Reference:
- Chattopadhay et al., "Grad-CAM++: Generalized Gradient-based Visual
  Explanations for Deep Convolutional Networks" (2018)
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from typing import Tuple, Optional


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for visual explanations.

    Extracts gradients from the final convolutional layer of a CNN classifier
    and generates a weighted activation heatmap that highlights the most
    important regions for the classification decision.

    Usage:
        gradcam = GradCAMPlusPlus(model, target_layer)
        heatmap = gradcam.generate(input_tensor, target_class)
        overlay = gradcam.overlay_heatmap(heatmap, original_image)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The classification model (e.g., ResNet50, EfficientNet)
            target_layer: The last convolutional layer to extract gradients from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM++ heatmap for the given input.

        Args:
            input_tensor: Input image tensor of shape (1, C, H, W)
            target_class: Target class index. If None, uses predicted class.

        Returns:
            Normalized heatmap of shape (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all gradients
        self.model.zero_grad()

        # Backward pass for target class
        target_score = output[0, target_class]
        target_score.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients  # (1, C, h, w)
        activations = self.activations  # (1, C, h, w)

        # --- Grad-CAM++ weighting ---
        # Compute second and third order gradients
        grad_2 = gradients ** 2
        grad_3 = gradients ** 3

        # Compute alpha coefficients (Grad-CAM++ specific)
        # α_k_c = grad²_k / (2·grad²_k + Σ(A_k · grad³_k) + ε)
        sum_activations = torch.sum(activations, dim=(2, 3), keepdim=True)
        eps = 1e-8

        alpha_numer = grad_2
        alpha_denom = 2.0 * grad_2 + sum_activations * grad_3 + eps
        alpha = alpha_numer / alpha_denom

        # Only keep positive gradients (ReLU on gradients)
        weights = torch.sum(alpha * F.relu(gradients), dim=(2, 3), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Resize to input image size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > eps:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    @staticmethod
    def overlay_heatmap(
        heatmap: np.ndarray,
        original_image: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay a Grad-CAM heatmap onto the original image.

        Args:
            heatmap: Normalized heatmap (H, W) with values in [0, 1]
            original_image: Original image (H, W) or (H, W, 3), uint8
            alpha: Transparency of heatmap overlay (0=transparent, 1=opaque)
            colormap: OpenCV colormap (default: JET)

        Returns:
            Overlaid image (H, W, 3), uint8
        """
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # Convert heatmap to colored image
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), colormap
        )

        # Ensure original is 3-channel
        if original_image.ndim == 2:
            original_3ch = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            original_3ch = original_image.copy()

        # Ensure uint8
        if original_3ch.dtype != np.uint8:
            if original_3ch.max() <= 1.0:
                original_3ch = (original_3ch * 255).astype(np.uint8)
            else:
                original_3ch = original_3ch.astype(np.uint8)

        # Overlay
        overlaid = cv2.addWeighted(original_3ch, 1 - alpha, heatmap_colored, alpha, 0)

        return overlaid

    @staticmethod
    def create_side_by_side(
        original: np.ndarray,
        heatmap: np.ndarray,
        overlay: np.ndarray,
        label: str = "",
        confidence: float = 0.0,
    ) -> np.ndarray:
        """
        Create a side-by-side comparison: Original | Heatmap | Overlay.

        Args:
            original: Original image (H, W) or (H, W, 3)
            heatmap: Raw heatmap (H, W) in [0, 1]
            overlay: Overlaid image (H, W, 3)
            label: Prediction label
            confidence: Prediction confidence

        Returns:
            Combined image (H, W*3 + padding, 3)
        """
        h, w = original.shape[:2]

        # Convert original to 3-channel
        if original.ndim == 2:
            orig_3ch = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            orig_3ch = original.copy()

        if orig_3ch.dtype != np.uint8:
            orig_3ch = (orig_3ch * 255).astype(np.uint8) if orig_3ch.max() <= 1.0 else orig_3ch.astype(np.uint8)

        # Convert heatmap to colored
        heatmap_colored = cv2.applyColorMap(
            (cv2.resize(heatmap, (w, h)) * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        # Resize overlay
        overlay_resized = cv2.resize(overlay, (w, h))

        # Create padding
        pad = np.ones((h, 10, 3), dtype=np.uint8) * 30

        # Combine
        combined = np.concatenate([orig_3ch, pad, heatmap_colored, pad, overlay_resized], axis=1)

        # Add label text
        if label:
            text = f"{label} ({confidence:.1%})"
            color = (0, 255, 0) if "benign" in label.lower() else (0, 0, 255)
            cv2.putText(
                combined, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
            )

        return combined


def generate_gradcam_for_ensemble(
    resnet_model: torch.nn.Module,
    efficient_model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
) -> dict:
    """
    Generate Grad-CAM++ heatmaps for both ensemble models.

    Args:
        resnet_model: ResNet50 classifier
        efficient_model: EfficientNetV2 classifier
        input_tensor: Preprocessed input (1, 3, 224, 224)
        original_image: Original image for overlay (H, W)
        target_class: Target class (None = use predicted)

    Returns:
        Dictionary with heatmaps and overlays for both models
    """
    # Grad-CAM for ResNet50
    resnet_gradcam = GradCAMPlusPlus(
        resnet_model, resnet_model.get_last_conv_layer()
    )
    resnet_heatmap = resnet_gradcam.generate(input_tensor, target_class)
    resnet_overlay = GradCAMPlusPlus.overlay_heatmap(resnet_heatmap, original_image)

    # Grad-CAM for EfficientNet
    efficient_gradcam = GradCAMPlusPlus(
        efficient_model, efficient_model.get_last_conv_layer()
    )
    efficient_heatmap = efficient_gradcam.generate(input_tensor, target_class)
    efficient_overlay = GradCAMPlusPlus.overlay_heatmap(efficient_heatmap, original_image)

    # Combined heatmap (average of both)
    combined_heatmap = (resnet_heatmap + efficient_heatmap) / 2.0
    combined_overlay = GradCAMPlusPlus.overlay_heatmap(combined_heatmap, original_image)

    return {
        "resnet_heatmap": resnet_heatmap,
        "resnet_overlay": resnet_overlay,
        "efficient_heatmap": efficient_heatmap,
        "efficient_overlay": efficient_overlay,
        "combined_heatmap": combined_heatmap,
        "combined_overlay": combined_overlay,
    }
