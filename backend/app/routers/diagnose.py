"""
Diagnosis API Router.

Handles image upload, pipeline execution, and result delivery.
Supports WebSocket for real-time progress streaming.
"""

import os
import time
import uuid
import base64
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, Request
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings

router = APIRouter()

# In-memory history (in production, use a database)
diagnosis_history = []


@router.post("/diagnose")
async def diagnose_image(request: Request, file: UploadFile = File(...)):
    """
    Upload an ultrasound image and run the full AI diagnostic pipeline.

    Returns:
        JSON with diagnosis results, confidence, trust score,
        and base64-encoded visualization images.
    """
    # Validate file
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Read image
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    # Convert to numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    # Save uploaded file
    file_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(settings.UPLOADS_DIR, f"{file_id}_{file.filename}")
    with open(save_path, "wb") as f:
        f.write(contents)

    # Run pipeline
    try:
        pipeline = request.app.state.pipeline
        start_time = time.time()
        results = pipeline.run_full_pipeline(image)
        inference_time = time.time() - start_time
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Encode visualization images as base64
    visualizations = {}
    for key in ["detection_overlay", "segmentation_mask", "gradcam_heatmap", "gradcam_overlay"]:
        if key in results and results[key] is not None:
            img = results[key]
            if isinstance(img, np.ndarray):
                _, buffer = cv2.imencode(".png", img)
                visualizations[key] = base64.b64encode(buffer).decode("utf-8")

    # Encode original image
    _, orig_buffer = cv2.imencode(".png", image)
    visualizations["original"] = base64.b64encode(orig_buffer).decode("utf-8")

    # Build response
    response = {
        "id": file_id,
        "filename": file.filename,
        "prediction": results.get("prediction", "unknown"),
        "confidence": float(results.get("confidence", 0.0)),
        "trust_score": results.get("trust_score", "UNKNOWN"),
        "inference_time_ms": round(inference_time * 1000, 2),
        "phases": results.get("phases", {}),
        "visualizations": visualizations,
        "details": {
            "resnet_prediction": results.get("resnet_prediction", "N/A"),
            "efficient_prediction": results.get("efficient_prediction", "N/A"),
            "resnet_confidence": float(results.get("resnet_confidence", 0.0)),
            "efficient_confidence": float(results.get("efficient_confidence", 0.0)),
            "resnet_weight": float(results.get("resnet_weight", 0.5)),
            "efficient_weight": float(results.get("efficient_weight", 0.5)),
            "tumor_detected": results.get("tumor_detected", False),
            "bounding_box": results.get("bounding_box", None),
            "explained_model": results.get("explained_model"),
            "message": results.get("message"),
        },
    }

    # Store in history
    history_entry = {**response}
    history_entry.pop("visualizations", None)  # Don't store large base64 in memory
    diagnosis_history.append(history_entry)

    return JSONResponse(content=response)


@router.post("/diagnose/track-a")
async def diagnose_track_a(request: Request, file: UploadFile = File(...)):
    """Run Track A pipeline only (YOLO → Attention U-Net → ResNet+EfficientNet)."""
    # Reuse main endpoint with track parameter
    return await _run_track(request, file, track="A")


@router.post("/diagnose/track-b")
async def diagnose_track_b(request: Request, file: UploadFile = File(...)):
    """Run Track B pipeline only (MedSAM → Swin-V2 + ConvNeXt-V2)."""
    return await _run_track(request, file, track="B")


async def _run_track(request: Request, file: UploadFile, track: str):
    """Internal helper to run a specific track."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    pipeline = request.app.state.pipeline
    start_time = time.time()

    if track == "A":
        results = pipeline.run_track_a(image)
    else:
        results = pipeline.run_track_b(image)

    inference_time = time.time() - start_time

    return JSONResponse(content={
        "track": track,
        "prediction": results.get("prediction", "unknown"),
        "confidence": float(results.get("confidence", 0.0)),
        "trust_score": results.get("trust_score", "UNKNOWN"),
        "inference_time_ms": round(inference_time * 1000, 2),
    })


@router.get("/history")
async def get_history():
    """Get past diagnosis history."""
    return JSONResponse(content={"history": diagnosis_history[-50:]})  # Last 50


@router.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint that runs the pipeline and streams real per-phase progress.

    Client → server: one JSON message per image
        {"image": "<base64-encoded PNG/JPG>"}
    Server → client:
        {"type": "connected"}
        {"type": "progress", "phase": 1, "name": "Detection", "status": "processing"}
        {"type": "progress", "phase": 1, "name": "Detection", "status": "complete", "time_ms": 150.2}
        ... phases 2-4 ...
        {"type": "result", "prediction": ..., "confidence": ..., "trust_score": ..., ...}
        {"type": "error", "detail": "..."}   on failure
    """
    import asyncio
    from starlette.websockets import WebSocketDisconnect

    await websocket.accept()
    await websocket.send_json({"type": "connected", "message": "WebSocket connected to UltraXAI pipeline"})
    loop = asyncio.get_running_loop()
    pipeline = websocket.app.state.pipeline

    try:
        while True:
            msg = await websocket.receive_json()
            try:
                raw = base64.b64decode(msg["image"])
                image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:
                image = None
            if image is None:
                await websocket.send_json({"type": "error", "detail": "Failed to decode image"})
                continue

            queue: asyncio.Queue = asyncio.Queue()

            def on_progress(phase, name, status, time_ms=None):
                loop.call_soon_threadsafe(queue.put_nowait, {
                    "type": "progress", "phase": phase, "name": name,
                    "status": status, "time_ms": time_ms,
                })

            start = time.time()
            task = loop.run_in_executor(None, pipeline.run_full_pipeline, image, on_progress)
            while not (task.done() and queue.empty()):
                try:
                    await websocket.send_json(await asyncio.wait_for(queue.get(), timeout=0.1))
                except asyncio.TimeoutError:
                    pass
            try:
                results = task.result()
            except Exception as e:
                await websocket.send_json({"type": "error", "detail": f"Pipeline error: {e}"})
                continue
            await websocket.send_json({
                "type": "result",
                "prediction": results.get("prediction", "unknown"),
                "confidence": float(results.get("confidence", 0.0)),
                "trust_score": results.get("trust_score", "UNKNOWN"),
                "tumor_detected": results.get("tumor_detected", False),
                "bounding_box": results.get("bounding_box"),
                "phases": results.get("phases", {}),
                "inference_time_ms": round((time.time() - start) * 1000, 2),
            })
    except WebSocketDisconnect:
        pass
