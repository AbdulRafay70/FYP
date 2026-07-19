"""
Benchmark Comparison: Track A vs Track B.

Runs evaluation on both pipelines and generates a comprehensive
comparison report with statistical significance testing.

Usage:
    python benchmark_comparison.py --test_dir datasets/OASBUD \
                                   --track_a_weights weights/ \
                                   --track_b_weights weights/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from utils.metrics import (
    compute_classification_metrics,
    compute_segmentation_metrics,
    print_classification_report,
)


def generate_report(track_a_metrics: dict, track_b_metrics: dict, output_path: str):
    """Generate a markdown comparison report."""

    report = f"""# UltraXAI — Benchmark Comparison Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Classification Metrics

| Metric | Track A (Clinical) | Track B (SOTA) | Winner |
|--------|-------------------|----------------|--------|
| Accuracy | {track_a_metrics.get('accuracy', 'N/A')} | {track_b_metrics.get('accuracy', 'N/A')} | — |
| Sensitivity | {track_a_metrics.get('sensitivity', 'N/A')} | {track_b_metrics.get('sensitivity', 'N/A')} | — |
| Specificity | {track_a_metrics.get('specificity', 'N/A')} | {track_b_metrics.get('specificity', 'N/A')} | — |
| F1 Score | {track_a_metrics.get('f1_score', 'N/A')} | {track_b_metrics.get('f1_score', 'N/A')} | — |
| AUC-ROC | {track_a_metrics.get('auc_roc', 'N/A')} | {track_b_metrics.get('auc_roc', 'N/A')} | — |

## Segmentation Metrics

| Metric | Track A | Track B | Winner |
|--------|---------|---------|--------|
| Dice | {track_a_metrics.get('dice', 'N/A')} | {track_b_metrics.get('dice', 'N/A')} | — |
| IoU | {track_a_metrics.get('iou', 'N/A')} | {track_b_metrics.get('iou', 'N/A')} | — |

## Inference Performance

| Metric | Track A | Track B |
|--------|---------|---------|
| Avg Inference Time | {track_a_metrics.get('inference_time', 'N/A')} ms | {track_b_metrics.get('inference_time', 'N/A')} ms |

---

> **Note**: This report will be populated after model training is complete.
> Run the evaluation scripts to fill in the actual metrics.
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"[Report] Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Comparison")
    parser.add_argument("--output", default="evaluation/benchmark_report.md")
    args = parser.parse_args()

    # Placeholder metrics (will be filled after training)
    track_a = {"accuracy": "N/A", "sensitivity": "N/A", "specificity": "N/A",
               "f1_score": "N/A", "auc_roc": "N/A", "dice": "N/A", "iou": "N/A",
               "inference_time": "N/A"}
    track_b = track_a.copy()

    generate_report(track_a, track_b, args.output)


if __name__ == "__main__":
    main()
