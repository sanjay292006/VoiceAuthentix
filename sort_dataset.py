# ================================================================
#  VoiceAuthentix — Dataset Sorter
#  File: sort_dataset.py
#  Reads ASVspoof 2019 protocol file and sorts audio into
#  data/real/ and data/fake/ folders automatically
# ================================================================

import os
import shutil
from pathlib import Path

# ── CONFIG — update these paths to match your setup ─────────────
PROTOCOL_FILE = r"D:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
AUDIO_DIR     = r"D:\ASVspoof2019\LA\ASVspoof2019_LA_train\flac"
OUTPUT_REAL   = r"D:\VoiceAuthentix\BACKEND\data\real"
OUTPUT_FAKE   = r"D:\VoiceAuthentix\BACKEND\data\fake"

# ── Create output folders ────────────────────────────────────────
os.makedirs(OUTPUT_REAL, exist_ok=True)
os.makedirs(OUTPUT_FAKE, exist_ok=True)

print("📂 Reading protocol file...")
real_count = 0
fake_count = 0
not_found  = 0

with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        file_id = parts[1]        # e.g. LA_T_1000137
        label   = parts[4]        # bonafide or spoof

        # Find the audio file
        src = os.path.join(AUDIO_DIR, file_id + ".flac")
        if not os.path.exists(src):
            not_found += 1
            continue

        # Copy to correct folder
        if label == "bonafide":
            dst = os.path.join(OUTPUT_REAL, file_id + ".flac")
            real_count += 1
        else:
            dst = os.path.join(OUTPUT_FAKE, file_id + ".flac")
            fake_count += 1

        shutil.copy2(src, dst)

        # Progress
        total = real_count + fake_count
        if total % 500 == 0:
            print(f"   Sorted {total} files... (real={real_count}, fake={fake_count})")

print("\n✅ Dataset sorting complete!")
print(f"   Real files : {real_count}")
print(f"   Fake files : {fake_count}")
print(f"   Not found  : {not_found}")
print(f"\n   Real folder: {OUTPUT_REAL}")
print(f"   Fake folder: {OUTPUT_FAKE}")
print("\n▶ Now run:  python train.py")
