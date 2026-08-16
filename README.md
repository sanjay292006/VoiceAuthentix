# 🎙️ VoiceAuthentix – Deepfake Audio Detection System

> **AI-powered audio forensics system for detecting AI-generated and deepfake speech using Deep Learning.**

VoiceAuthentix is an AI-based deepfake audio detection platform designed to identify whether an audio sample is **Real (Human)** or **Fake (AI-Generated)**.

The system uses a **CNN-BiLSTM architecture with Temporal Attention** to analyze audio features represented as Mel-Spectrograms and classify suspicious or synthetic speech.

---

## 🚀 Key Highlights

- 🎙️ Real-time deepfake audio detection
- 🤖 CNN + BiLSTM deep learning architecture
- 🧠 Temporal Attention mechanism
- 🎵 Mel-Spectrogram based audio analysis
- ⚡ Fast inference for audio classification
- 🔌 FastAPI backend
- 🌐 REST API architecture
- 📡 WebSocket support for real-time audio streaming
- 🖥️ Audio file and microphone-based analysis
- 📊 Model performance monitoring
- 🧪 Training pipeline included
- 🚀 Deployment-ready project structure

---

# 🎯 Problem Statement

With the rapid development of Generative AI and voice synthesis technologies, AI-generated speech has become increasingly realistic.

Deepfake audio can potentially be used for:

- Identity impersonation
- Voice fraud
- Social engineering
- Fake interviews
- Financial scams
- Misinformation
- Unauthorized voice cloning

Traditional audio verification methods may struggle to distinguish highly realistic synthetic speech.

**VoiceAuthentix addresses this problem using Deep Learning-based audio analysis.**

---

# 💡 Proposed Solution

VoiceAuthentix converts an input audio signal into a **Mel-Spectrogram**, which provides a visual representation of the audio's frequency characteristics.

The extracted features are processed through a hybrid deep learning architecture:

```text
Audio Input
     │
     ▼
Audio Preprocessing
     │
     ▼
Mel-Spectrogram
     │
     ▼
CNN Feature Extraction
     │
     ▼
BiLSTM Temporal Modeling
     │
     ▼
Temporal Attention
     │
     ▼
Classification Layer
     │
     ▼
┌───────────────┐
│ REAL / FAKE   │
└───────────────┘
