# ================================================================
#  VoiceAuthentix — FastAPI Backend
#  File: main.py
#  Entry point — runs the full server
# ================================================================

import uvicorn
import os
import sys

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── App init ────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceAuthentix API",
    description="DeepFake Audio Detection using Mel-Spectrogram Deep Learning",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────
try:
    from routers import analysis, streaming, training, health
    app.include_router(health.router,    prefix="/api", tags=["Health"])
    app.include_router(analysis.router,  prefix="/api", tags=["Analysis"])
    app.include_router(streaming.router, prefix="/api", tags=["Streaming"])
    app.include_router(training.router,  prefix="/api", tags=["Training"])
except ImportError as e:
    print(f"\n❌ ERROR: Could not import routers.")
    print(f"   Detail : {e}")

# ── Root ─────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "app": "VoiceAuthentix",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }

# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
