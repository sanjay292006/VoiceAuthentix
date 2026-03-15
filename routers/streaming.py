# ================================================================
#  VoiceAuthentix — Live Streaming Router
#  File: routers/streaming.py
#  WebSocket /api/stream  — Real-time mic chunk analysis
# ================================================================

import numpy as np
import json
import time
import asyncio
import struct
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

from core.mel_extractor import extractor
from core.model_engine   import engine

router = APIRouter()

# ── Session tracker ──────────────────────────────────────────────
class StreamSession:
    def __init__(self, session_id: str):
        self.session_id     = session_id
        self.chunks_analyzed = 0
        self.fake_count     = 0
        self.real_count     = 0
        self.score_sum      = 0.0
        self.start_time     = time.time()
        self.audio_buffer   = []
        self.sample_rate    = 22050
        self.chunk_samples  = 22050   # 1 second

    def push_samples(self, samples: np.ndarray):
        self.audio_buffer.extend(samples.tolist())

    def has_chunk(self) -> bool:
        return len(self.audio_buffer) >= self.chunk_samples

    def pop_chunk(self) -> np.ndarray:
        chunk = np.array(self.audio_buffer[:self.chunk_samples], dtype=np.float32)
        self.audio_buffer = self.audio_buffer[self.chunk_samples // 2:]  # 50% overlap
        return chunk

    def record_result(self, fake_prob: float):
        self.chunks_analyzed += 1
        self.score_sum       += fake_prob
        if fake_prob > engine.threshold:
            self.fake_count += 1
        else:
            self.real_count += 1

    def session_stats(self) -> Dict[str, Any]:
        elapsed = round(time.time() - self.start_time, 1)
        avg     = round(self.score_sum / max(self.chunks_analyzed, 1), 4)
        return {
            "session_id":      self.session_id,
            "elapsed_sec":     elapsed,
            "chunks_analyzed": self.chunks_analyzed,
            "fake_count":      self.fake_count,
            "real_count":      self.real_count,
            "average_score":   avg,
        }


# ── Active sessions ──────────────────────────────────────────────
active_sessions: Dict[str, StreamSession] = {}


# ── WS /api/stream ───────────────────────────────────────────────
@router.websocket("/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
    ─ Client sends: binary PCM float32 audio frames (22050 Hz, mono)
    ─ Server sends: JSON result per analyzed chunk

    Connection flow:
    1. Client connects → server sends {"type":"connected"}
    2. Client streams PCM chunks as binary frames
    3. Server accumulates → when 1 second buffered → inference
    4. Server sends JSON: {"type":"result", "fake_probability": ..., ...}
    5. Client disconnects → server sends session summary
    """

    await websocket.accept()

    import uuid
    session_id = str(uuid.uuid4())[:8]
    session    = StreamSession(session_id)
    active_sessions[session_id] = session

    # ── Send connection confirmation ─────────────────────────────
    await websocket.send_json({
        "type":       "connected",
        "session_id": session_id,
        "message":    "VoiceAuthentix stream ready",
        "config": {
            "sample_rate":    session.sample_rate,
            "chunk_duration": 1.0,
            "overlap":        0.5,
            "model":          engine.model_version,
        }
    })

    try:
        while True:
            # ── Receive audio data ───────────────────────────────
            try:
                data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "message": "keep-alive"})
                continue

            # ── Handle text messages (control commands) ──────────
            if "text" in data:
                msg = json.loads(data["text"])

                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

                elif msg.get("type") == "stats":
                    await websocket.send_json({
                        "type":  "stats",
                        "stats": session.session_stats()
                    })

                elif msg.get("type") == "stop":
                    await websocket.send_json({
                        "type":    "session_end",
                        "summary": session.session_stats()
                    })
                    break

                continue

            # ── Handle binary PCM audio ──────────────────────────
            if "bytes" in data:
                raw_bytes = data["bytes"]

                if len(raw_bytes) < 4:
                    continue

                # Parse float32 PCM
                num_samples = len(raw_bytes) // 4
                samples     = np.array(
                    struct.unpack(f"{num_samples}f", raw_bytes[:num_samples * 4]),
                    dtype=np.float32
                )

                # Clamp to [-1, 1]
                samples = np.clip(samples, -1.0, 1.0)

                session.push_samples(samples)

                # ── If we have a full 1-second chunk → analyze ───
                if session.has_chunk():
                    chunk = session.pop_chunk()

                    # Extract mel-spectrogram
                    try:
                        mel = extractor.extract_mel(chunk)
                    except Exception as e:
                        await websocket.send_json({
                            "type":    "error",
                            "message": f"Feature extraction failed: {str(e)}"
                        })
                        continue

                    # Run inference
                    result = engine.analyze_live_chunk(mel)
                    session.record_result(result["fake_probability"])

                    # Send result to frontend
                    await websocket.send_json({
                        "type":             "result",
                        "chunk_index":      session.chunks_analyzed,
                        "fake_probability": result["fake_probability"],
                        "real_probability": result["real_probability"],
                        "verdict":          result["verdict"],
                        "confidence":       result["confidence"],
                        "latency_ms":       result["latency_ms"],
                        "session_stats":    session.session_stats(),
                    })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # Cleanup session
        active_sessions.pop(session_id, None)


# ── GET /api/stream/sessions ─────────────────────────────────────
@router.get("/stream/sessions")
def active_stream_sessions():
    """Return stats on all currently active WebSocket sessions."""
    return {
        "active_sessions": len(active_sessions),
        "sessions": [s.session_stats() for s in active_sessions.values()]
    }
