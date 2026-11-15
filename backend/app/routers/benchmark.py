"""Benchmark data router — serves only verified evaluation output."""

import json
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import settings

router = APIRouter()

RESULTS_FILE = os.path.join(settings.PROJECT_ROOT, "evaluation", "results", "metrics.json")


@router.get("/benchmark")
async def get_benchmark():
    """
    Return the test-set metrics written by evaluation/evaluate_test_set.py.

    No numbers are hard-coded: if the evaluation has not been run, all metrics
    are null and the status is "awaiting_evaluation".
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return JSONResponse(content={"status": "evaluated", **json.load(f)})

    empty = {k: None for k in ("accuracy", "sensitivity", "specificity", "f1_score",
                               "auc_roc", "dice", "iou", "inference_time_ms")}
    return JSONResponse(content={
        "status": "awaiting_evaluation",
        "track_a": {"name": "Clinical Pipeline (YOLOv8m-Seg + Attention U-Net + ResNet50/EfficientNetV2-S ensemble)",
                    "metrics": empty, "status": "awaiting_evaluation"},
        "track_b": {"name": "SOTA Pipeline (MedSAM + Swin-V2 + ConvNeXt-V2) — not implemented",
                    "metrics": empty, "status": "not_implemented"},
    })
