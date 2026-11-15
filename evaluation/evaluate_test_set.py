"""
UltraXAI — Held-out TEST-set evaluation (evidence for thesis Chapter 4).

Evaluates every component on the SAME held-out test partition written by
run_full_training.py (datasets/prepared/test, never trained or validated on):

  * Classification  — ResNet50, EfficientNetV2-S and the 0.40/0.60 soft-voting
                      ensemble, on lesion ROI crops (GT mask box + 15% padding,
                      exactly as in training). Binary task, Malignant = positive (1).
  * Segmentation    — Attention U-Net on whole test images vs GT masks:
                      Dice, IoU, pixel accuracy, Hausdorff distance.
  * Detection       — optional, YOLOv8 via Ultralytics `val(split="test")` when
                      --yolo_data points to a data.yaml whose `test:` entry is the
                      same test images: precision, recall, mAP@0.5, mAP@0.5:0.95.
  * Grad-CAM++      — optional sample figures (original / heatmap / overlay + prediction).

Outputs (evaluation/results/):
  predictions_classification.csv   raw per-image probabilities & predictions
  segmentation_per_image.csv       per-image Dice / IoU
  metrics.json                     all metrics + TP/TN/FP/FN + run record
  confusion_matrix_<model>.png     one per classifier and the ensemble
  roc_curves.png                   ROC of all three classifiers
  gradcam/                         Grad-CAM++ sample figures
  run_record.json                  git commit, environment, checkpoint SHA-256,
                                   split-manifest SHA-256, test-set size

Usage:
    python evaluation/evaluate_test_set.py --test_dir datasets/prepared/test \
        --weights_dir backend/weights --gradcam_samples 6 \
        [--yolo_data datasets/yolo_format/data.yaml]
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))

from data.dataset import BreastUltrasoundDataset  # noqa: E402
from data.roi import crop_roi_from_mask  # noqa: E402
from data.transforms import get_classification_val_transforms, get_segmentation_val_transforms  # noqa: E402
from utils.metrics import compute_classification_metrics, compute_segmentation_metrics, dice_coefficient, iou_score  # noqa: E402
from utils.run_logging import environment_info  # noqa: E402

CLASS_NAMES = ["Benign", "Malignant"]  # index = label; Malignant (1) is the positive class


def sha256_file(path):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_jsonable(d):
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in d.items()}


# ----------------------------------------------------------------------------- classification
@torch.no_grad()
def evaluate_classification(ensemble, samples, device):
    tf = get_classification_val_transforms(224)
    rows = []
    for s in samples:
        img = cv2.imread(s["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE)
        roi = crop_roi_from_mask(img, mask)
        x = tf(image=cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB))["image"].unsqueeze(0).to(device)
        out = ensemble(x)
        p_res = torch.softmax(out["resnet_logits"], 1)[0, 1].item()
        p_eff = torch.softmax(out["efficient_logits"], 1)[0, 1].item()
        p_ens = out["probabilities"][0, 1].item()
        rows.append({
            "image": os.path.relpath(s["image_path"], ROOT), "label": s["label"],
            "resnet_prob_malignant": p_res, "resnet_pred": int(p_res >= 0.5),
            "efficient_prob_malignant": p_eff, "efficient_pred": int(p_eff >= 0.5),
            "ensemble_prob_malignant": p_ens, "ensemble_pred": int(out["prediction"].item()),
            "trust_score": out["trust_score"][0],
        })
    return rows


def plot_confusion(cm, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"Pred {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"True {c}" for c in CLASS_NAMES])
    ax.set_title(f"{title}\n(n = {cm.sum()} test images)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score
    y = np.array([r["label"] for r in rows])
    fig, ax = plt.subplots(figsize=(5, 5))
    for key, name in (("resnet", "ResNet50"), ("efficient", "EfficientNetV2-S"), ("ensemble", "Ensemble")):
        p = np.array([r[f"{key}_prob_malignant"] for r in rows])
        if len(set(y)) > 1:
            fpr, tpr, _ = roc_curve(y, p)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"ROC — held-out test set (n = {len(rows)})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- segmentation
@torch.no_grad()
def evaluate_segmentation(unet, samples, device):
    tf = get_segmentation_val_transforms(256)
    preds, gts, rows = [], [], []
    for s in samples:
        img = cv2.imread(s["image_path"], cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        x = tf(image=img)["image"].unsqueeze(0).to(device)
        prob = unet(x).squeeze().cpu().numpy()
        pred = (cv2.resize(prob, (img.shape[1], img.shape[0])) > 0.5).astype(np.uint8)
        preds.append(pred)
        gts.append(gt)
        rows.append({"image": os.path.relpath(s["image_path"], ROOT),
                     "dice": dice_coefficient(pred, gt), "iou": iou_score(pred, gt)})
    return compute_segmentation_metrics(preds, gts), rows


# ----------------------------------------------------------------------------- grad-cam samples
def save_gradcam_samples(ensemble, rows, samples, out_dir, n):
    from utils.explainability import generate_gradcam_for_ensemble
    os.makedirs(out_dir, exist_ok=True)
    tf = get_classification_val_transforms(224)
    by_path = {os.path.relpath(s["image_path"], ROOT): s for s in samples}
    # a mix of correct and incorrect cases from both classes
    chosen = [r for r in rows if r["ensemble_pred"] != r["label"]][: n // 2]
    rest = n - len(chosen)
    chosen += [r for r in rows if r["label"] == 1 and r["ensemble_pred"] == 1][: rest - rest // 2]
    chosen += [r for r in rows if r["label"] == 0 and r["ensemble_pred"] == 0][: rest // 2]
    for r in chosen:
        s = by_path[r["image"]]
        img = cv2.imread(s["image_path"], cv2.IMREAD_GRAYSCALE)
        roi = crop_roi_from_mask(img, cv2.imread(s["mask_path"], cv2.IMREAD_GRAYSCALE))
        x = tf(image=cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB))["image"].unsqueeze(0)
        x = x.to(next(ensemble.parameters()).device)
        maps = generate_gradcam_for_ensemble(ensemble.resnet, ensemble.efficient, x, roi,
                                             target_class=r["ensemble_pred"],
                                             resnet_weight=ensemble.resnet_weight,
                                             efficient_weight=ensemble.efficient_weight)
        heat = cv2.applyColorMap((maps["combined_heatmap"] * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, (roi.shape[1], roi.shape[0]))
        panel = np.hstack([cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR), heat, maps["combined_overlay"]])
        scale = 900 / panel.shape[1]
        panel = cv2.resize(panel, (900, int(panel.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)
        panel = cv2.copyMakeBorder(panel, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        text = (f"True: {CLASS_NAMES[r['label']]} | Pred: {CLASS_NAMES[r['ensemble_pred']]} "
                f"(p_malig={r['ensemble_prob_malignant']:.2f}, trust={r['trust_score']})")
        cv2.putText(panel, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imwrite(os.path.join(out_dir, Path(r["image"]).stem.replace(" ", "_") + "_gradcam.png"), panel)
    print(f"[Grad-CAM++] {len(chosen)} sample figures saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", default="datasets/prepared/test")
    ap.add_argument("--weights_dir", default="backend/weights")
    ap.add_argument("--out_dir", default="evaluation/results")
    ap.add_argument("--manifest", default="datasets/prepared/split_manifest.csv")
    ap.add_argument("--yolo_weights", default="yolov8m_seg_breast.pt")
    ap.add_argument("--yolo_data", default=None, help="data.yaml with a test: split (optional)")
    ap.add_argument("--gradcam_samples", type=int, default=6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device(a.device)
    w = lambda name: os.path.join(a.weights_dir, name)  # noqa: E731

    test = BreastUltrasoundDataset([a.test_dir], mode="segmentation")  # images that have GT masks
    samples = test.samples
    n_ben = sum(s["label"] == 0 for s in samples)
    print(f"[Test set] {len(samples)} images ({n_ben} Benign, {len(samples) - n_ben} Malignant)")
    metrics = {"task": "binary classification: Benign (0) vs Malignant (1); positive class = Malignant",
               "test_set": {"dir": a.test_dir, "n": len(samples), "benign": n_ben,
                            "malignant": len(samples) - n_ben}}

    # ---- classification ----
    from models.classifier_heads import build_ensemble
    ensemble = build_ensemble(2, w("resnet50_breast.pth"), w("efficientnetv2_breast.pth")).to(device).eval()
    t0 = time.time()
    rows = evaluate_classification(ensemble, samples, device)
    metrics["classification_ms_per_image"] = 1000 * (time.time() - t0) / max(len(rows), 1)
    with open(os.path.join(a.out_dir, "predictions_classification.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    y = np.array([r["label"] for r in rows])
    metrics["classification"] = {}
    for key, name in (("resnet", "ResNet50"), ("efficient", "EfficientNetV2-S"), ("ensemble", "Ensemble (0.40/0.60)")):
        pred = np.array([r[f"{key}_pred"] for r in rows])
        prob = np.array([r[f"{key}_prob_malignant"] for r in rows])
        m = compute_classification_metrics(y, pred, prob)
        metrics["classification"][name] = to_jsonable(m)
        cm = np.array([[m["true_negatives"], m["false_positives"]],
                       [m["false_negatives"], m["true_positives"]]])
        plot_confusion(cm, f"{name} — confusion matrix", os.path.join(a.out_dir, f"confusion_matrix_{key}.png"))
        print(f"  {name:22s} acc={m['accuracy']:.4f} sens={m['sensitivity']:.4f} "
              f"spec={m['specificity']:.4f} F1={m['f1_score']:.4f} AUC={m.get('auc_roc', 0):.4f} "
              f"TP={m['true_positives']} TN={m['true_negatives']} FP={m['false_positives']} FN={m['false_negatives']}")
    plot_roc(rows, os.path.join(a.out_dir, "roc_curves.png"))

    # ---- segmentation ----
    if os.path.exists(w("attention_unet_breast.pth")):
        from models.attention_unet import build_attention_unet
        unet = build_attention_unet(1, 1, pretrained_path=w("attention_unet_breast.pth")).to(device).eval()
        seg, seg_rows = evaluate_segmentation(unet, samples, device)
        metrics["segmentation"] = {"model": "Attention U-Net", **to_jsonable(seg)}
        with open(os.path.join(a.out_dir, "segmentation_per_image.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(seg_rows[0].keys()))
            wr.writeheader()
            wr.writerows(seg_rows)
        print(f"  Attention U-Net        Dice={seg['dice']:.4f} IoU={seg['iou']:.4f}")
    else:
        metrics["segmentation"] = "not evaluated: attention_unet_breast.pth missing"

    # ---- detection ----
    if a.yolo_data and os.path.exists(w(a.yolo_weights)):
        from ultralytics import YOLO
        r = YOLO(w(a.yolo_weights)).val(data=a.yolo_data, split="test", plots=True,
                                       project=a.out_dir, name="yolo_test", exist_ok=True)
        metrics["detection"] = {"model": a.yolo_weights, "precision": float(r.box.mp),
                                "recall": float(r.box.mr), "mAP50": float(r.box.map50),
                                "mAP50_95": float(r.box.map)}
        print(f"  YOLO                   P={r.box.mp:.4f} R={r.box.mr:.4f} mAP50={r.box.map50:.4f}")
    else:
        metrics["detection"] = "not evaluated: pass --yolo_data (data.yaml with test split) and YOLO weights"

    if a.gradcam_samples > 0:
        save_gradcam_samples(ensemble, rows, samples, os.path.join(a.out_dir, "gradcam"), a.gradcam_samples)

    record = {
        "environment": environment_info(),
        "checkpoints_sha256": {n: sha256_file(w(n)) for n in (
            "resnet50_breast.pth", "efficientnetv2_breast.pth", "attention_unet_breast.pth", a.yolo_weights)},
        "split_manifest_sha256": sha256_file(a.manifest),
        "ensemble_weights": {"resnet50": ensemble.resnet_weight, "efficientnetv2_s": ensemble.efficient_weight},
    }
    metrics["run_record"] = record
    with open(os.path.join(a.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(a.out_dir, "run_record.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"\nAll results written to {a.out_dir}")


if __name__ == "__main__":
    main()
