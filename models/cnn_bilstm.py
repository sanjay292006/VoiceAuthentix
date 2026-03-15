# ================================================================
#  VoiceAuthentix — Real PyTorch CNN-BiLSTM Model Architecture
#  File: models/cnn_bilstm.py
#
#  Architecture:
#    Input  : Mel-Spectrogram  (1, 128, T)
#    Block 1: Conv2D(32)  → BatchNorm → ReLU → MaxPool → Dropout
#    Block 2: Conv2D(64)  → BatchNorm → ReLU → MaxPool → Dropout
#    Block 3: Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout
#    Reshape : (B, T', 128*H') → BiLSTM input
#    BiLSTM  : 2 layers, hidden=256, bidirectional → output (B, T', 512)
#    Attention: Weighted context vector  (B, 512)
#    Head    : Linear(512→256) → ReLU → Dropout → Linear(256→2)
#    Output  : Softmax → [P(real), P(fake)]
# ================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Squeeze-and-Excitation block (channel attention) ─────────────
class SEBlock(nn.Module):
    """
    Channel-wise attention — lets the model focus on the
    most discriminative mel frequency bands.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# ── CNN Feature Extraction Block ─────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 pool_size=(2, 2), dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se      = SEBlock(out_ch)
        self.pool    = nn.MaxPool2d(pool_size)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ── Temporal Self-Attention ───────────────────────────────────────
class TemporalAttention(nn.Module):
    """
    Learns which time frames are most important for the verdict.
    Fake audio often has specific temporal regions with artifacts.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, lstm_out):
        # lstm_out: (B, T, hidden_dim)
        scores  = self.attn(lstm_out)              # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        context = (weights * lstm_out).sum(dim=1)  # (B, hidden_dim)
        return context, weights.squeeze(-1)


# ── Full CNN-BiLSTM Model ─────────────────────────────────────────
class CNNBiLSTM(nn.Module):
    """
    Main deepfake audio detection model.

    Input  : (B, 1, N_MELS=128, T_FRAMES)
    Output : (B, 2)  →  [logit_real, logit_fake]

    Usage:
        model  = CNNBiLSTM()
        logits = model(mel_tensor)          # shape (B, 2)
        probs  = torch.softmax(logits, 1)   # shape (B, 2)
        fake_p = probs[:, 1]                # fake probability
    """

    VERSION = "cnn-bilstm-v2.0"

    def __init__(
        self,
        n_mels:       int   = 128,
        lstm_hidden:  int   = 256,
        lstm_layers:  int   = 2,
        num_classes:  int   = 2,
        dropout:      float = 0.4,
    ):
        super().__init__()
        self.n_mels      = n_mels
        self.lstm_hidden = lstm_hidden

        # ── CNN Backbone ─────────────────────────────────────────
        # Input: (B, 1, 128, T)
        self.cnn = nn.Sequential(
            CNNBlock(1,   32,  pool_size=(2, 2), dropout=0.15),   # → (B, 32,  64, T/2)
            CNNBlock(32,  64,  pool_size=(2, 2), dropout=0.20),   # → (B, 64,  32, T/4)
            CNNBlock(64,  128, pool_size=(2, 2), dropout=0.25),   # → (B, 128, 16, T/8)
        )

        # After 3 MaxPool(2,2): mel_dim = 128/8 = 16
        self.cnn_mel_out = n_mels // 8       # = 16
        self.lstm_input  = 128 * self.cnn_mel_out   # = 128 * 16 = 2048

        # ── BiLSTM Temporal Encoder ──────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = self.lstm_input,
            hidden_size   = lstm_hidden,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if lstm_layers > 1 else 0.0,
        )

        # BiLSTM output dim = hidden * 2 (forward + backward)
        self.lstm_out_dim = lstm_hidden * 2   # = 512

        # ── Temporal Attention ───────────────────────────────────
        self.attention = TemporalAttention(self.lstm_out_dim)

        # ── Classification Head ──────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_out_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

        # ── Weight initialization ────────────────────────────────
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 1, N_MELS, T_FRAMES)
        """
        B = x.size(0)

        # ── CNN: extract spatial frequency features ───────────────
        cnn_out = self.cnn(x)                           # (B, 128, mel/8, T/8)
        _, C, H, T = cnn_out.shape

        # Reshape for LSTM: (B, T_frames, features)
        cnn_out = cnn_out.permute(0, 3, 1, 2)          # (B, T, C, H)
        cnn_out = cnn_out.reshape(B, T, C * H)          # (B, T, 2048)

        # ── BiLSTM: encode temporal patterns ─────────────────────
        lstm_out, _ = self.lstm(cnn_out)                # (B, T, 512)

        # ── Attention: focus on artifact-heavy frames ─────────────
        context, attn_weights = self.attention(lstm_out) # (B, 512)

        # ── Classify ─────────────────────────────────────────────
        logits = self.classifier(context)               # (B, 2)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities. (B, 2)"""
        return F.softmax(self.forward(x), dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Model factory ────────────────────────────────────────────────
def build_model(
    n_mels:      int   = 128,
    lstm_hidden: int   = 256,
    lstm_layers: int   = 2,
    dropout:     float = 0.4,
    pretrained_path: str = None,
    device: str = None,
) -> CNNBiLSTM:
    """
    Build and optionally load a pretrained model.

    Args:
        n_mels          : number of mel bins (must match training)
        lstm_hidden     : BiLSTM hidden size
        lstm_layers     : number of LSTM layers
        dropout         : dropout rate
        pretrained_path : path to .pt checkpoint file
        device          : 'cuda', 'cpu', or None (auto-detect)

    Returns:
        CNNBiLSTM model (eval mode if pretrained loaded)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CNNBiLSTM(
        n_mels      = n_mels,
        lstm_hidden = lstm_hidden,
        lstm_layers = lstm_layers,
        dropout     = dropout,
    ).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        checkpoint = torch.load(pretrained_path, map_location=device)
        # Support both raw state_dict and full checkpoint dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Loaded checkpoint from {pretrained_path}")
            print(f"   Epoch     : {checkpoint.get('epoch', 'N/A')}")
            print(f"   Val Acc   : {checkpoint.get('val_acc', 'N/A')}")
            print(f"   Val EER   : {checkpoint.get('val_eer', 'N/A')}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ Loaded state dict from {pretrained_path}")
        model.eval()
    else:
        print(f"ℹ️  Built fresh model ({model.count_parameters():,} parameters) on {device}")

    return model


import os

if __name__ == "__main__":
    # ── Quick architecture test ──────────────────────────────────
    print("=" * 60)
    print("  VoiceAuthentix — CNN-BiLSTM Architecture Test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    model = build_model(device=device)
    print(f"  Params : {model.count_parameters():,}")
    print(f"  Version: {model.VERSION}")

    # Test forward pass
    dummy = torch.randn(4, 1, 128, 87).to(device)  # batch=4, 1ch, 128 mels, 87 frames (~1s)
    with torch.no_grad():
        logits = model(dummy)
        probs  = torch.softmax(logits, dim=1)

    print(f"\n  Input  : {list(dummy.shape)}")
    print(f"  Output : {list(logits.shape)}")
    print(f"  Sample probs (real, fake):")
    for i in range(4):
        print(f"    Sample {i+1}: real={probs[i,0]:.3f}  fake={probs[i,1]:.3f}")
    print("\n  ✅ Architecture test passed!")
