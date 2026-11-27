"""Shared constants and default paths.

Every value here matches the original Colab run, so the released checkpoint
(`siamese_best.pt`) works with the preprocessing in this package unchanged.
Paths can be overridden with the MUSICSIM_DATA environment variable or the
command-line flags of each script.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Audio / spectrogram parameters (identical to the training run) ---
TARGET_SR = 16_000      # MelodySim is distributed at 16 kHz
N_FFT = 1024            # ~64 ms window
HOP_LENGTH = 256        # ~16 ms hop
N_MELS = 128
POWER = 2.0
TARGET_T = 685          # 95th percentile of training clip lengths (frames, ~11 s)

# --- Model ---
IN_CHANNELS = 3         # [log-mel, HPSS harmonic, HPSS percussive]
EMB_DIM = 256
CHANNEL_NAMES = ("log-mel", "harmonic", "percussive")

# --- Dataset ---
MELODYSIM_REPO = "amaai-lab/melodySim"
VERSIONS = ("original", "version_0", "version_1", "version_2")

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("MUSICSIM_DATA", REPO_ROOT / "data"))

SHARD_LIST = REPO_ROOT / "data" / "melodysim_shards.txt"
REFERENCE_MANIFEST = REPO_ROOT / "data" / "melodysim_manifest.csv"

PARQUET_DIR = DATA_ROOT / "melodysim_parquets"
MULTISPEC_DIR = DATA_ROOT / "multispec"
MANIFEST = DATA_ROOT / "manifest_multispec.csv"

MCIC_DIR = DATA_ROOT / "mcic"
MCIC_CSV = MCIC_DIR / "MCIC-ISMIR2024.csv"
MCIC_MIDI_DIR = MCIC_DIR / "MIDI FILES"
MCIC_AUDIO_DIR = MCIC_DIR / "AUDIO"

CHECKPOINT = REPO_ROOT / "checkpoints" / "siamese_best.pt"
OUTPUTS = REPO_ROOT / "outputs"
