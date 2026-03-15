# ================================================================
#  VoiceAuthentix — Mel-Spectrogram Feature Extractor
#  File: core/mel_extractor.py
#  Extracts mel-spectrogram features from raw audio
# ================================================================

import numpy as np
import librosa
import librosa.display
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any


# ── Constants ───────────────────────────────────────────────────
SAMPLE_RATE     = 22050
N_MELS          = 128
N_FFT           = 2048
HOP_LENGTH      = 512
PRE_EMPHASIS    = 0.97
CHUNK_DURATION  = 1.0      # seconds per analysis window
OVERLAP         = 0.5      # 50% overlap between chunks


class MelExtractor:
    """
    Full mel-spectrogram extraction pipeline.
    Mirrors the Python backend of a real CNN-BiLSTM deepfake detector.
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        pre_emphasis: float = PRE_EMPHASIS,
    ):
        self.sr          = sr
        self.n_mels      = n_mels
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.pre_emphasis = pre_emphasis

    # ── Load audio from bytes ────────────────────────────────────
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Load audio from raw bytes (any format librosa supports)."""
        audio_io = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_io, sr=self.sr, mono=True)
        return y, sr

    # ── Pre-emphasis filter ──────────────────────────────────────
    def apply_pre_emphasis(self, y: np.ndarray) -> np.ndarray:
        """Boost high-frequency components — enhances artifact detection."""
        return librosa.effects.preemphasis(y, coef=self.pre_emphasis)

    # ── Mel spectrogram ──────────────────────────────────────────
    def extract_mel(self, y: np.ndarray) -> np.ndarray:
        """
        Full mel-spectrogram extraction pipeline:
        1. Pre-emphasis
        2. STFT → Power Spectrum
        3. Mel filter bank (128 bins)
        4. Convert to dB scale
        5. Per-instance normalization
        """
        y = self.apply_pre_emphasis(y)

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=20,
            fmax=self.sr // 2
        )

        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [-1, 1]
        mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        return mel_norm

    # ── Additional features ──────────────────────────────────────
    def extract_mfcc(self, y: np.ndarray, n_mfcc: int = 40) -> np.ndarray:
        """MFCC features for supplementary analysis."""
        return librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """Chroma features — harmonic content."""
        return librosa.feature.chroma_stft(
            y=y, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length
        )

    def extract_spectral_contrast(self, y: np.ndarray) -> np.ndarray:
        """Spectral contrast — peaks vs valleys in spectrum."""
        return librosa.feature.spectral_contrast(
            y=y, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length
        )

    # ── Full feature bundle ──────────────────────────────────────
    def extract_all_features(self, y: np.ndarray) -> Dict[str, Any]:
        """Extract all features used by the model."""
        mel       = self.extract_mel(y)
        mfcc      = self.extract_mfcc(y)
        chroma    = self.extract_chroma(y)
        contrast  = self.extract_spectral_contrast(y)

        # Scalar audio statistics
        rms           = float(np.sqrt(np.mean(y ** 2)))
        zero_crossings = int(np.sum(librosa.zero_crossings(y)))
        tempo_result  = librosa.beat.beat_track(y=y, sr=self.sr)
        tempo         = float(np.atleast_1d(tempo_result[0])[0])

        return {
            "mel_spectrogram":    mel,
            "mfcc":               mfcc,
            "chroma":             chroma,
            "spectral_contrast":  contrast,
            "rms_energy":         rms,
            "zero_crossing_rate": zero_crossings,
            "tempo":              float(tempo),
            "duration_sec":       float(len(y) / self.sr),
            "sample_rate":        self.sr,
            "num_mel_bins":       self.n_mels,
            "num_frames":         mel.shape[1],
        }

    # ── Chunked analysis for long audio ─────────────────────────
    def extract_chunks(
        self,
        y: np.ndarray,
        chunk_dur: float = CHUNK_DURATION,
        overlap: float = OVERLAP
    ):
        """
        Slide a window over long audio and yield mel chunks.
        Used for file-level analysis with temporal resolution.
        """
        chunk_samples   = int(chunk_dur * self.sr)
        hop_samples     = int(chunk_samples * (1 - overlap))
        chunks          = []
        timestamps      = []

        start = 0
        while start + chunk_samples <= len(y):
            chunk = y[start: start + chunk_samples]
            mel   = self.extract_mel(chunk)
            chunks.append(mel)
            timestamps.append(round(start / self.sr, 3))
            start += hop_samples

        return chunks, timestamps

    # ── Render spectrogram as base64 image ───────────────────────
    def render_mel_image(self, y: np.ndarray) -> str:
        """
        Render mel-spectrogram as a base64 PNG string
        for embedding directly in API JSON responses.
        """
        mel    = self.extract_mel(y)
        fig, ax = plt.subplots(figsize=(10, 3), facecolor="#020510")
        ax.set_facecolor("#020510")

        librosa.display.specshow(
            mel,
            sr=self.sr,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="mel",
            cmap="magma",
            ax=ax
        )
        ax.set_xlabel("Time (s)", color="#5a6a8a")
        ax.set_ylabel("Mel Bins", color="#5a6a8a")
        ax.tick_params(colors="#5a6a8a")
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#020510")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    # ── Waveform as base64 ───────────────────────────────────────
    def render_waveform_image(self, y: np.ndarray) -> str:
        """Render waveform as a base64 PNG."""
        fig, ax = plt.subplots(figsize=(10, 2), facecolor="#020510")
        ax.set_facecolor("#020510")
        times = np.linspace(0, len(y) / self.sr, num=len(y))
        ax.plot(times, y, color="#00c8ff", linewidth=0.5, alpha=0.8)
        ax.axhline(0, color="#1a2340", linewidth=0.5)
        ax.set_xlabel("Time (s)", color="#5a6a8a")
        ax.tick_params(colors="#5a6a8a")
        ax.spines[:].set_color("#1a2340")
        plt.tight_layout(pad=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor="#020510")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# ── Singleton instance ───────────────────────────────────────────
extractor = MelExtractor()
