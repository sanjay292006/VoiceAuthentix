# ================================================================
#  VoiceAuthentix — Real Model Engine (Updated)
#  File: core/model_engine.py
#  Replaces dummy model with real CNN-BiLSTM PyTorch inference
#
#  HOW TO USE:
#  1. Run:  python train.py          (trains and saves models/model.pt)
#  2. Copy this file over:  core/model_engine.py
#  3. Restart:  python main.py
# ================================================================

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from dataclasses import dataclass

from models.cnn_bilstm import build_model, CNNBiLSTM

# ── Constants (must match training config) ────────────────────────
N_MELS       = 128
SAMPLE_RATE  = 22050
LSTM_HIDDEN  = 256
LSTM_LAYERS  = 2
DROPOUT      = 0.4
THRESHOLD    = 0.55


# ── Result schema ────────────────────────────────────────────────
@dataclass
class InferenceResult:
    fake_probability:  float
    real_probability:  float
    verdict:           str
    confidence:        float
    latency_ms:        float
    model_version:     str
    chunk_scores:      List[float]
    anomaly_regions:   List[Dict]
    feature_stats:     Dict[str, Any]


# ── Real PyTorch Model Wrapper ────────────────────────────────────
class RealCNNBiLSTM:
    def __init__(self, model_path: str = "models/model.pt"):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path  = model_path
        self.trained     = False

        print(f"Initializing RealCNNBiLSTM on {self.device}")

        self.model = build_model(
            n_mels          = N_MELS,
            lstm_hidden     = LSTM_HIDDEN,
            lstm_layers     = LSTM_LAYERS,
            dropout         = DROPOUT,
            pretrained_path = model_path if os.path.exists(model_path) else None,
            device          = str(self.device),
        )
        self.model.eval()

        if os.path.exists(model_path):
            self.trained = True
            ckpt = torch.load(model_path, map_location=self.device)
            epoch   = ckpt.get("epoch", "?")
            val_acc = ckpt.get("val_acc", "?")
            val_eer = ckpt.get("val_eer", "?")
            self.model_version = (
                f"{CNNBiLSTM.VERSION} | epoch={epoch} | "
                f"val_acc={val_acc} | eer={val_eer}"
            )
            print(f"Loaded trained model: {self.model_version}")
        else:
            self.model_version = f"{CNNBiLSTM.VERSION} | untrained (run train.py)"
            print(f"No checkpoint at {model_path} — run: python train.py")

    def _preprocess(self, mel: np.ndarray) -> torch.Tensor:
        if mel.shape[0] != N_MELS:
            mel = mel[:N_MELS, :] if mel.shape[0] > N_MELS else \
                  np.pad(mel, ((0, N_MELS - mel.shape[0]), (0, 0)))
        if mel.shape[1] < 8:
            mel = np.pad(mel, ((0, 0), (0, 8 - mel.shape[1])), mode="reflect")
        tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict_single(self, mel: np.ndarray) -> float:
        tensor = self._preprocess(mel)
        logits = self.model(tensor)
        probs  = F.softmax(logits, dim=1)
        return float(probs[0, 1].item())

    @torch.no_grad()
    def predict_batch(self, mels: List[np.ndarray]) -> List[float]:
        if not mels:
            return []
        max_t = max(m.shape[1] for m in mels)
        tensors = []
        for mel in mels:
            if mel.shape[0] != N_MELS:
                mel = mel[:N_MELS, :] if mel.shape[0] > N_MELS else \
                      np.pad(mel, ((0, N_MELS - mel.shape[0]), (0, 0)))
            if mel.shape[1] < max_t:
                mel = np.pad(mel, ((0, 0), (0, max_t - mel.shape[1])), mode="reflect")
            tensors.append(mel)
        batch  = torch.FloatTensor(np.stack(tensors)).unsqueeze(1).to(self.device)
        logits = self.model(batch)
        probs  = F.softmax(logits, dim=1)
        return [float(p) for p in probs[:, 1].cpu().numpy()]


# ── Inference Engine ─────────────────────────────────────────────
class InferenceEngine:
    def __init__(self, model_path: str = "models/model.pt"):
        self.model_wrapper = RealCNNBiLSTM(model_path)
        self.threshold     = THRESHOLD
        self.model_version = self.model_wrapper.model_version

    def analyze_chunks(self, chunks, timestamps, features) -> InferenceResult:
        t_start = time.perf_counter()
        if not chunks:
            return self._empty_result()

        chunk_scores = self.model_wrapper.predict_batch(chunks)
        weights      = np.linspace(0.5, 1.0, len(chunk_scores))
        fake_prob    = float(np.average(chunk_scores, weights=weights))
        real_prob    = 1.0 - fake_prob
        verdict      = "FAKE" if fake_prob > self.threshold else "REAL"
        confidence   = max(fake_prob, real_prob)

        anomaly_regions = [
            {"start_sec": ts, "end_sec": round(ts+1.0,3),
             "score": round(sc,4), "severity": "high" if sc>0.80 else "medium"}
            for sc, ts in zip(chunk_scores, timestamps)
            if sc > self.threshold
        ]

        feature_stats = {
            "rms_energy":         round(features.get("rms_energy", 0), 6),
            "zero_crossing_rate": features.get("zero_crossing_rate", 0),
            "tempo_bpm":          round(features.get("tempo", 0), 2),
            "duration_sec":       round(features.get("duration_sec", 0), 3),
            "num_mel_bins":       features.get("num_mel_bins", 128),
            "num_frames":         features.get("num_frames", 0),
            "chunks_analyzed":    len(chunks),
            "fake_chunks":        sum(1 for s in chunk_scores if s > self.threshold),
            "model_trained":      self.model_wrapper.trained,
        }

        return InferenceResult(
            fake_probability = round(fake_prob, 4),
            real_probability = round(real_prob, 4),
            verdict          = verdict,
            confidence       = round(confidence, 4),
            latency_ms       = round((time.perf_counter()-t_start)*1000, 2),
            model_version    = self.model_version,
            chunk_scores     = [round(s,4) for s in chunk_scores],
            anomaly_regions  = anomaly_regions,
            feature_stats    = feature_stats,
        )

    def analyze_live_chunk(self, mel: np.ndarray) -> Dict[str, Any]:
        t_start   = time.perf_counter()
        fake_prob = self.model_wrapper.predict_single(mel)
        return {
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(1.0-fake_prob, 4),
            "verdict":          "FAKE" if fake_prob > self.threshold else "REAL",
            "confidence":       round(max(fake_prob, 1.0-fake_prob), 4),
            "latency_ms":       round((time.perf_counter()-t_start)*1000, 2),
            "model_trained":    self.model_wrapper.trained,
        }

    def _empty_result(self) -> InferenceResult:
        return InferenceResult(
            fake_probability=0.0, real_probability=1.0,
            verdict="REAL", confidence=1.0, latency_ms=0.0,
            model_version=self.model_version,
            chunk_scores=[], anomaly_regions=[], feature_stats={},
        )


# ── Singleton ────────────────────────────────────────────────────
engine = InferenceEngine()
