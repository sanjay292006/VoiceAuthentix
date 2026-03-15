# ================================================================
#  VoiceAuthentix — Health Router
#  File: routers/health.py
# ================================================================

from fastapi import APIRouter
from datetime import datetime
import platform, sys

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status":     "ok",
        "app":        "VoiceAuthentix",
        "version":    "2.0.0",
        "timestamp":  datetime.utcnow().isoformat(),
        "python":     sys.version,
        "platform":   platform.system(),
        "endpoints": {
            "health":    "GET  /api/health",
            "analyze":   "POST /api/analyze",
            "stream":    "WS   /api/stream",
            "train":     "POST /api/train/start",
            "status":    "GET  /api/train/status",
            "docs":      "GET  /docs"
        }
    }

@router.get("/model/info")
def model_info():
    from core.model_engine import engine
    return {
        "model_version":   engine.model_version,
        "threshold":       engine.threshold,
        "architecture":    "CNN-BiLSTM + Attention",
        "mel_bins":        128,
        "sample_rate":     22050,
        "n_fft":           2048,
        "hop_length":      512,
        "chunk_duration":  1.0,
        "overlap":         0.5,
        "training_data":   "ASVspoof 2021 (simulated)",
        "accuracy":        "96.4% (simulated)",
        "eer":             "2.3% (simulated)",
    }
