# ================================================================
#  VoiceAuthentix — Full PyTorch Training Pipeline
#  File: train.py
#
#  Usage:
#    python train.py                          # synthetic data
#    python train.py --data_dirs "path1" "path2" "path3" "path4"
#    python train.py --epochs 50 --batch_size 64
#
#  Output:
#    models/model.pt          ← best checkpoint
#    models/final_model.pt    ← final epoch model
#    models/training_log.json ← full epoch history
# ================================================================

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

import librosa
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from models.cnn_bilstm import build_model, CNNBiLSTM


# ================================================================
#  CONFIG
# ================================================================
class Config:
    SAMPLE_RATE  = 22050
    N_MELS       = 128
    N_FFT        = 2048
    HOP_LENGTH   = 512
    DURATION     = 1.0
    PRE_EMPHASIS = 0.97
    LSTM_HIDDEN  = 256
    LSTM_LAYERS  = 2
    DROPOUT      = 0.4
    EPOCHS       = 50
    BATCH_SIZE   = 64
    LR           = 1e-3
    WEIGHT_DECAY = 1e-4
    VAL_SPLIT    = 0.15
    TEST_SPLIT   = 0.10
    PATIENCE     = 8
    MIN_LR       = 1e-6
    GRAD_CLIP    = 1.0
    MODEL_DIR    = "models"
    LOG_PATH     = "models/training_log.json"
    BEST_MODEL   = "models/model.pt"
    FINAL_MODEL  = "models/final_model.pt"
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS  = 0   # 0 is safest on Windows
    PIN_MEMORY   = torch.cuda.is_available()

cfg = Config()


# ================================================================
#  DATASET
# ================================================================
class AudioDataset(Dataset):
    EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

    def __init__(self, data_dir: str, config: Config = cfg, augment: bool = True):
        self.config  = config
        self.augment = augment
        self.samples : List[Tuple[str, int]] = []

        data_path = Path(data_dir)
        real_dir  = data_path / "real"
        fake_dir  = data_path / "fake"

        if real_dir.exists():
            for f in real_dir.rglob("*"):
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), 0))
        else:
            print(f"   ⚠️  real/ not found in {data_dir}")

        if fake_dir.exists():
            for f in fake_dir.rglob("*"):
                if f.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(f), 1))
        else:
            print(f"   ⚠️  fake/ not found in {data_dir}")

        n_real = sum(1 for _, l in self.samples if l == 0)
        n_fake = sum(1 for _, l in self.samples if l == 1)
        print(f"   📁 {data_dir.split(os.sep)[-1]}: {len(self.samples)} samples (real={n_real}, fake={n_fake})")

        total = len(self.samples)
        self.class_weights = torch.tensor([
            total / (2 * n_real) if n_real else 1.0,
            total / (2 * n_fake) if n_fake else 1.0,
        ], dtype=torch.float32)

    def _load_audio(self, path: str) -> np.ndarray:
        target_len = int(self.config.DURATION * self.config.SAMPLE_RATE)
        try:
            y, sr = librosa.load(path, sr=self.config.SAMPLE_RATE, mono=True,
                                  duration=self.config.DURATION * 3)
        except Exception:
            return np.zeros(target_len, dtype=np.float32)

        # Fix: handle empty or too-short audio
        if len(y) == 0:
            return np.zeros(target_len, dtype=np.float32)

        if len(y) < target_len:
            # Use constant padding for very short files
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        elif len(y) > target_len:
            if self.augment:
                start = np.random.randint(0, len(y) - target_len)
            else:
                start = (len(y) - target_len) // 2
            y = y[start: start + target_len]

        return y[:target_len].astype(np.float32)

    def _extract_mel(self, y: np.ndarray) -> np.ndarray:
        y   = librosa.effects.preemphasis(y, coef=self.config.PRE_EMPHASIS)
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS, n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH, fmin=20,
            fmax=self.config.SAMPLE_RATE // 2,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return ((mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)).astype(np.float32)

    def _augment(self, y: np.ndarray) -> np.ndarray:
        if np.random.random() < 0.4:
            y = y + np.random.randn(len(y)) * np.random.uniform(0.001, 0.015)
        if np.random.random() < 0.4:
            y = y * np.random.uniform(0.7, 1.3)
        return np.clip(y, -1.0, 1.0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        y   = self._load_audio(path)
        if self.augment:
            y = self._augment(y)
        mel = self._extract_mel(y)
        return torch.FloatTensor(mel).unsqueeze(0), torch.tensor(label, dtype=torch.long)


# ================================================================
#  METRICS
# ================================================================
def compute_eer(y_true, y_scores):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    with torch.no_grad():
        for mels, labels in loader:
            mels, labels = mels.to(device), labels.to(device)
            logits = model(mels)
            loss   = criterion(logits, labels)
            probs  = F.softmax(logits, dim=1)
            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    acc = float((all_preds == all_labels).mean())

    try:
        auc = roc_auc_score(all_labels, all_probs)
        eer = compute_eer(all_labels, all_probs)
    except Exception:
        auc = 0.0
        eer = 0.5

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "loss": round(total_loss / len(loader), 4),
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ================================================================
#  TRAINING LOOP
# ================================================================
def train(args):
    os.makedirs(cfg.MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 65)
    print("   VoiceAuthentix — CNN-BiLSTM Training")
    print("=" * 65)
    print(f"   Device     : {cfg.DEVICE}")
    print(f"   Epochs     : {args.epochs}")
    print(f"   Batch size : {args.batch_size}")
    print(f"   LR         : {args.lr}")
    print(f"   Data dirs  : {len(args.data_dirs)} folder(s)")
    print("=" * 65 + "\n")

    # ── Load all datasets and combine ───────────────────────────
    print("📂 Loading datasets...")
    all_datasets = []
    total_weights = None

    for data_dir in args.data_dirs:
        ds = AudioDataset(data_dir.strip(), config=cfg, augment=True)
        if len(ds) > 0:
            all_datasets.append(ds)
            if total_weights is None:
                total_weights = ds.class_weights
            else:
                total_weights = (total_weights + ds.class_weights) / 2

    if not all_datasets:
        print("❌ No valid datasets found! Check your paths.")
        return

    # Combine all datasets
    full_dataset = ConcatDataset(all_datasets)
    n_total = len(full_dataset)
    n_val   = int(n_total * cfg.VAL_SPLIT)
    n_test  = int(n_total * cfg.TEST_SPLIT)
    n_train = n_total - n_val - n_test

    print(f"\n   ✅ Total combined: {n_total} samples")
    print(f"   Train: {n_train}  Val: {n_val}  Test: {n_test}\n")

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=cfg.NUM_WORKERS,
                              pin_memory=cfg.PIN_MEMORY, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=cfg.PIN_MEMORY)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=cfg.NUM_WORKERS,
                              pin_memory=cfg.PIN_MEMORY)

    # ── Model ────────────────────────────────────────────────────
    print("🧠 Building CNN-BiLSTM model...")
    model = build_model(
        n_mels=cfg.N_MELS, lstm_hidden=cfg.LSTM_HIDDEN,
        lstm_layers=cfg.LSTM_LAYERS, dropout=cfg.DROPOUT,
        device=cfg.DEVICE,
    )
    print(f"   Parameters: {model.count_parameters():,}\n")

    weights   = (total_weights if total_weights is not None else torch.ones(2)).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=cfg.MIN_LR)

    best_val_acc = 0.0
    best_eer     = 1.0
    patience_ctr = 0
    epoch_logs   = []

    print("🚀 Training started...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'Val EER':>7} | {'LR':>8}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total   = 0
        t0 = time.time()

        for mels, labels in train_loader:
            mels, labels = mels.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            optimizer.zero_grad()
            logits = model(mels)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()
            train_loss    += loss.item()
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        scheduler.step()
        curr_lr        = optimizer.param_groups[0]["lr"]
        avg_train_loss = round(train_loss / len(train_loader), 4)
        avg_train_acc  = round(train_correct / train_total, 4)
        val_metrics    = evaluate(model, val_loader, criterion, cfg.DEVICE)
        elapsed        = round(time.time() - t0, 1)

        print(f"{epoch:>6} | {avg_train_loss:>10.4f} | {avg_train_acc:>9.4f} | "
              f"{val_metrics['loss']:>8.4f} | {val_metrics['accuracy']:>7.4f} | "
              f"{val_metrics['eer']:>7.4f} | {curr_lr:>8.6f}  [{elapsed}s]")

        log = {
            "epoch": epoch, "train_loss": avg_train_loss,
            "train_acc": avg_train_acc, "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"], "val_auc": val_metrics["auc"],
            "val_eer": val_metrics["eer"], "lr": round(curr_lr, 8),
            "time_sec": elapsed,
        }
        epoch_logs.append(log)

        if val_metrics["accuracy"] > best_val_acc or val_metrics["eer"] < best_eer:
            best_val_acc = max(val_metrics["accuracy"], best_val_acc)
            best_eer     = min(val_metrics["eer"], best_eer)
            patience_ctr = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "val_eer": val_metrics["eer"],
                "val_auc": val_metrics["auc"],
                "config": {
                    "n_mels": cfg.N_MELS, "lstm_hidden": cfg.LSTM_HIDDEN,
                    "lstm_layers": cfg.LSTM_LAYERS, "dropout": cfg.DROPOUT,
                    "sample_rate": cfg.SAMPLE_RATE,
                },
                "model_version": CNNBiLSTM.VERSION,
            }, cfg.BEST_MODEL)
            print(f"         ✅ New best model saved  "
                  f"(acc={best_val_acc:.4f}, EER={best_eer:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.PATIENCE and args.early_stop:
                print(f"\n⏹  Early stopping at epoch {epoch}")
                break

    # ── Final model ───────────────────────────────────────────────
    torch.save({
        "epoch": args.epochs, "model_state_dict": model.state_dict(),
        "val_acc": val_metrics["accuracy"], "val_eer": val_metrics["eer"],
        "model_version": CNNBiLSTM.VERSION,
    }, cfg.FINAL_MODEL)

    # ── Test evaluation ───────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 Final evaluation on TEST set:")
    print("=" * 65)
    ckpt = torch.load(cfg.BEST_MODEL, map_location=cfg.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, cfg.DEVICE)
    print(f"   Accuracy : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"   AUC-ROC  : {test_metrics['auc']:.4f}")
    print(f"   EER      : {test_metrics['eer']:.4f} ({test_metrics['eer']*100:.2f}%)")
    print(f"   TP={test_metrics['tp']} TN={test_metrics['tn']} "
          f"FP={test_metrics['fp']} FN={test_metrics['fn']}")
    print(f"\n   Best model  → {cfg.BEST_MODEL}")
    print(f"   Final model → {cfg.FINAL_MODEL}")

    with open(cfg.LOG_PATH, "w") as f:
        json.dump({
            "model_version": CNNBiLSTM.VERSION,
            "best_val_acc": best_val_acc,
            "best_eer": best_eer,
            "test_metrics": test_metrics,
            "total_epochs": len(epoch_logs),
            "epoch_logs": epoch_logs,
            "data_dirs": args.data_dirs,
        }, f, indent=2)
    print(f"   Training log → {cfg.LOG_PATH}")
    print("=" * 65 + "\n")


# ================================================================
#  ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoiceAuthentix Training")
    parser.add_argument("--data_dirs", nargs="+", required=False,
                        default=["data"], help="One or more dataset directories")
    parser.add_argument("--epochs",     type=int,   default=cfg.EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=cfg.BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=cfg.LR)
    parser.add_argument("--early_stop", action="store_true", default=True)
    parser.add_argument("--no_early_stop", dest="early_stop", action="store_false")
    args = parser.parse_args()
    train(args)
