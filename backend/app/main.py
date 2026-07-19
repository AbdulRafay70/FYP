"""
UltraXAI — FastAPI Backend Main Application.

Serves the AI pipeline via REST API with WebSocket support for
real-time processing progress streaming.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import diagnose, health, benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle manager."""
    print("\n" + "=" * 60)
    print("  🧬 UltraXAI Backend Starting...")
    print("=" * 60)

    # Load models on startup
    from app.services.pipeline_orchestrator import PipelineOrchestrator
    app.state.pipeline = PipelineOrchestrator(
        weights_dir=settings.WEIGHTS_DIR,
        device=settings.DEVICE,
    )

    print("  ✓ All models loaded successfully")
    print(f"  ✓ Device: {settings.DEVICE}")
    print(f"  ✓ Server ready at http://localhost:{settings.PORT}")
    print("=" * 60 + "\n")

    yield

    # Cleanup
    print("\n  Shutting down UltraXAI Backend...")


app = FastAPI(
    title="UltraXAI — Breast Cancer Detection API",
    description="AI-powered breast cancer detection from ultrasound images with explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(diagnose.router, prefix="/api/v1", tags=["Diagnosis"])
app.include_router(benchmark.router, prefix="/api/v1", tags=["Benchmark"])

# Serve uploaded files
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
