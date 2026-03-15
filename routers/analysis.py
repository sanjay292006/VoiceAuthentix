# ================================================================
#  VoiceAuthentix — Analysis Router
#  File: routers/analysis.py
#  POST /api/analyze  — Upload audio file → full ML analysis
# ================================================================

import os
import uuid
import time
import numpy as np
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional

from core.mel_extractor import extractor
from core.model_engine   import engine

router = APIRouter()

# ── Allowed audio formats ────────────────────────────────────────
ALLOWED_EXTENSIONS = {
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/wave",
    "audio/x-wav", "audio/ogg", "audio/flac", "audio/x-flac",
    "audio/mp4", "audio/m4a", "audio/aac", "audio/webm"
}
MAX_FILE_SIZE_MB = 50


# ── POST /api/analyze ────────────────────────────────────────────
@router.post("/analyze")
async def analyze_audio(
    file: UploadFile = File(...),
    include_images: Optional[bool] = True,
    include_chunks: Optional[bool] = True,
):
    """
    Full pipeline:
    1. Validate & read uploaded audio file
    2. Load with Librosa at 22050 Hz
    3. Extract mel-spectrogram (128 bins, 2048 FFT, 512 hop)
    4. Slide window → per-chunk analysis
    5. CNN-BiLSTM inference on each chunk
    6. Aggregate → final verdict + confidence
    7. Return JSON with result, images, feature stats
    """

    # ── 1. Validate file ─────────────────────────────────────────
    if file.content_type not in ALLOWED_EXTENSIONS:
        # also accept by extension as fallback
        ext = (file.filename or "").split(".")[-1].lower()
        if ext not in {"mp3","wav","ogg","flac","m4a","aac","webm"}:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported format: {file.content_type}. "
                       f"Supported: MP3, WAV, OGG, FLAC, M4A, AAC"
            )

    audio_bytes = await file.read()

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {size_mb:.1f} MB. Max: {MAX_FILE_SIZE_MB} MB"
        )

    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="File is too small or empty")

    # ── 2. Load audio ────────────────────────────────────────────
    try:
        y, sr = extractor.load_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {str(e)}")

    duration_sec = len(y) / sr

    if duration_sec < 0.5:
        raise HTTPException(status_code=400, detail="Audio too short (min 0.5 seconds)")

    # ── 3. Extract all features ──────────────────────────────────
    try:
        features = extractor.extract_all_features(y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {str(e)}")

    # ── 4. Chunk the audio ───────────────────────────────────────
    chunks, timestamps = extractor.extract_chunks(y)
    if not chunks:
        # Audio too short for chunking — analyze as single chunk
        chunks     = [extractor.extract_mel(y)]
        timestamps = [0.0]

    # ── 5 + 6. Inference ────────────────────────────────────────
    result = engine.analyze_chunks(chunks, timestamps, features)

    # ── 7. Build response ────────────────────────────────────────
    response = {
        "status":           "success",
        "file_name":        file.filename,
        "file_size_mb":     round(size_mb, 3),
        "duration_sec":     round(duration_sec, 3),
        "sample_rate":      sr,

        # ── VERDICT ─────────────────────────────────────────────
        "verdict":          result.verdict,
        "fake_probability": result.fake_probability,
        "real_probability": result.real_probability,
        "confidence":       result.confidence,
        "is_deepfake":      result.verdict == "FAKE",

        # ── TEMPORAL ────────────────────────────────────────────
        "anomaly_regions":  result.anomaly_regions,
        "latency_ms":       result.latency_ms,
        "model_version":    result.model_version,

        # ── FEATURES ────────────────────────────────────────────
        "feature_stats":    result.feature_stats,
    }

    # ── Optional: include chunk-level scores ─────────────────────
    if include_chunks:
        response["chunk_scores"] = [
            {"time_sec": ts, "fake_prob": sc}
            for ts, sc in zip(timestamps, result.chunk_scores)
        ]

    # ── Optional: include base64 images ──────────────────────────
    if include_images:
        try:
            response["mel_spectrogram_img"] = extractor.render_mel_image(y)
            response["waveform_img"]        = extractor.render_waveform_image(y)
        except Exception:
            response["mel_spectrogram_img"] = None
            response["waveform_img"]        = None

    return JSONResponse(content=response)


# ── POST /api/analyze/chunk ──────────────────────────────────────
@router.post("/analyze/chunk")
async def analyze_single_chunk(file: UploadFile = File(...)):
    """
    Lightweight endpoint — analyze a single 1-second audio chunk.
    Used by the frontend for rapid sequential analysis.
    """
    audio_bytes = await file.read()

    try:
        y, sr = extractor.load_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    mel    = extractor.extract_mel(y)
    result = engine.analyze_live_chunk(mel)

    return JSONResponse(content=result)


# ── GET /api/analyze/formats ─────────────────────────────────────
@router.get("/analyze/formats")
def supported_formats():
    return {
        "supported_formats": ["MP3", "WAV", "OGG", "FLAC", "M4A", "AAC", "WebM"],
        "max_size_mb":       MAX_FILE_SIZE_MB,
        "min_duration_sec":  0.5,
        "recommended_sr":    22050,
    }
