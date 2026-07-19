"""Health check router."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return JSONResponse(content={"status": "healthy", "service": "UltraXAI Backend"})


@router.get("/models/status")
async def model_status(request: Request):
    """Check which models are loaded and ready."""
    pipeline = getattr(request.app.state, "pipeline", None)

    if pipeline is None:
        return JSONResponse(content={"status": "not_initialized", "models": {}})

    return JSONResponse(content={
        "status": "ready",
        "models": pipeline.get_model_status(),
    })
