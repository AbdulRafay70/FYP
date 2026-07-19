"""
Evaluation Metrics for Segmentation and Classification.

Provides comprehensive metrics for model evaluation:
- Segmentation: Dice, IoU, Hausdorff Distance, Pixel Accuracy
- Classification: Accuracy, Sensitivity, Specificity, F1, AUC-ROC
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


# ============================================================================
# Segmentation Metrics
# ============================================================================

def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Dice Similarity Coefficient.

    DSC = 2 × |A ∩ B| / (|A| + |B|)

    Args:
        pred: Predicted binary mask (H, W) or (B, H, W)
        target: Ground truth binary mask (H, W) or (B, H, W)
        smooth: Smoothing factor to prevent division by zero

    Returns:
        Dice coefficient in [0, 1]
    """
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)

    intersection = (pred_flat * target_flat).sum()
    return (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Intersection over Union (Jaccard Index).

    IoU = |A ∩ B| / |A ∪ B|

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask

    Returns:
        IoU score in [0, 1]
    """
    pred_flat = pred.flatten().astype(np.float32)
    target_flat = target.flatten().astype(np.float32)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute pixel-level accuracy.

    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask

    Returns:
        Pixel accuracy in [0, 1]
    """
    correct = (pred.flatten() == target.flatten()).sum()
    total = pred.flatten().shape[0]
    return correct / total


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute Hausdorff Distance between predicted and ground truth contours.

    HD = max(h(A, B), h(B, A))
    where h(A, B) = max_{a ∈ A} min_{b ∈ B} ||a - b||

    Args:
        pred: Predicted binary mask (H, W)
        target: Ground truth binary mask (H, W)

    Returns:
        Hausdorff distance in pixels
    """
    from scipy.spatial.distance import directed_hausdorff

    # Get boundary points
    pred_points = np.argwhere(pred > 0)
    target_points = np.argwhere(target > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return float("inf")

    d1 = directed_hausdorff(pred_points, target_points)[0]
    d2 = directed_hausdorff(target_points, pred_points)[0]

    return max(d1, d2)


def compute_segmentation_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
) -> Dict[str, float]:
    """
    Compute all segmentation metrics averaged over a batch.

    Args:
        predictions: List of predicted binary masks
        targets: List of ground truth binary masks

    Returns:
        Dictionary of averaged metrics
    """
    dice_scores = []
    iou_scores = []
    pixel_accs = []
    hausdorff_dists = []

    for pred, target in zip(predictions, targets):
        # Binarize predictions
        pred_bin = (pred > 0.5).astype(np.float32)
        target_bin = (target > 0.5).astype(np.float32)

        dice_scores.append(dice_coefficient(pred_bin, target_bin))
        iou_scores.append(iou_score(pred_bin, target_bin))
        pixel_accs.append(pixel_accuracy(pred_bin, target_bin))

        try:
            hd = hausdorff_distance(pred_bin, target_bin)
            if hd != float("inf"):
                hausdorff_dists.append(hd)
        except Exception:
            pass

    return {
        "dice": np.mean(dice_scores),
        "iou": np.mean(iou_scores),
        "pixel_accuracy": np.mean(pixel_accs),
        "hausdorff_distance": np.mean(hausdorff_dists) if hausdorff_dists else float("inf"),
        "dice_std": np.std(dice_scores),
        "iou_std": np.std(iou_scores),
    }


# ============================================================================
# Classification Metrics
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_probs: Prediction probabilities for positive class (N,) — for AUC-ROC

    Returns:
        Dictionary of all classification metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,  # Same as recall — catches malignant
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "npv": npv,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

    # AUC-ROC (requires probability scores)
    if y_probs is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["auc_roc"] = 0.0

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Dict[int, str] = None,
):
    """Print a formatted classification report."""
    if label_names is None:
        label_names = {0: "Benign", 1: "Malignant"}

    target_names = [label_names[i] for i in sorted(label_names.keys())]
    report = classification_report(y_true, y_pred, target_names=target_names)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60 + "\n")


def get_roc_curve_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for plotting.

    Returns:
        (fpr, tpr, thresholds) arrays
    """
    return roc_curve(y_true, y_probs)


def get_precision_recall_curve_data(
    y_true: np.ndarray,
    y_probs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get Precision-Recall curve data for plotting.

    Returns:
        (precision, recall, thresholds) arrays
    """
    return precision_recall_curve(y_true, y_probs)
