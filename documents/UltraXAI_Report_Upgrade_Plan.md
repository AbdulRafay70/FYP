# Roadmap Upgrade Plan: Pushing the UltraXAI Report to 9.8+/10

This document outlines the concrete technical tasks to be completed in the coming days to achieve a publication-grade (9.8+/10) Final Year Project (FYP) report and thesis chapter.

---

## Task List & Timeline

### Task 1: Expand Object Detection Metrics (YOLOv8)
* **Goal**: Transition from only reporting $mAP_{50}$ to standard computer vision publication metrics.
* **Actions**:
  1. Extract $mAP_{50-95}$ from the training runs.
  2. Add Precision and Recall columns to the detection baseline table.
  3. Include the YOLOv8-Seg training graphs (box loss, seg loss, cls loss, and mAP curves).

### Task 2: Expand Classification Metrics & Visual Validation
* **Goal**: Add standard clinical diagnostic validation graphs.
* **Actions**:
  1. Generate and plot a **Confusion Matrix** for the test split.
  2. Generate and plot the **ROC Curve** and calculate the **AUC (Area Under the Curve)**.
  3. Calculate the **Matthews Correlation Coefficient (MCC)** to mathematically justify the high accuracy despite the imbalance:
     $$MCC = \frac{TP \times TN - FP \times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

### Task 3: Compare Against Published Literature (State-of-the-Art)
* **Goal**: Prove the system is competitive with recent published work using the same dataset.
* **Actions**:
  1. Add a comparison table in the report comparing UltraXAI to 3-5 peer-reviewed papers using the BUSI dataset.
  2. *Example Reference Table*:
     * *Al-Dhabyani et al. (2020)*: 86.0% accuracy (Baseline)
     * *Proposed UltraXAI (2026)*: **99.30%** accuracy (Proposed)

### Task 4: Explainability Qualitative Analysis
* **Goal**: Make the Grad-CAM++ section scientifically rigorous.
* **Actions**:
  1. Add a dedicated figures page showing 3 side-by-side examples:
     * Column 1: Original Scan.
     * Column 2: Attention U-Net Segmented Margin.
     * Column 3: Grad-CAM++ Heatmap.
  2. Write a paragraph explaining how the Grad-CAM++ highlights correct margins (confirming that the model is not overfitting to background noise or label text).

---

## Action Plan Checklist

- [ ] Run evaluation scripts to output confusion matrix and save it as `confusion_matrix.png`.
- [ ] Run ROC calculation script and save the curve as `roc_curve.png`.
- [ ] Compile the literature benchmarking table from recent journals (2022-2025).
- [ ] Add the actual generated plots (`confusion_matrix.png`, `loss_curves.png`, and `gradcam_sample.png`) into the `documents/assets/` folder and link them in the report.
- [ ] Integrate the baseline comparison metrics directly into the React "Benchmark" tab.
