"""Benchmark data router."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/benchmark")
async def get_benchmark():
    """
    Get benchmark comparison data between Track A and Track B.
    Returns placeholder data until actual training results are available.
    """
    return JSONResponse(content={
        "track_a": {
            "name": "Clinical Pipeline (YOLO + Att-UNet + Ensemble)",
            "metrics": {
                "accuracy": None,
                "sensitivity": None,
                "specificity": None,
                "f1_score": None,
                "auc_roc": None,
                "dice": None,
                "iou": None,
                "inference_time_ms": None,
            },
            "status": "awaiting_training",
        },
        "track_b": {
            "name": "SOTA Pipeline (MedSAM + Swin-V2 + ConvNeXt-V2)",
            "metrics": {
                "accuracy": None,
                "sensitivity": None,
                "specificity": None,
                "f1_score": None,
                "auc_roc": None,
                "dice": None,
                "iou": None,
                "inference_time_ms": None,
            },
            "status": "awaiting_training",
        },
    })
