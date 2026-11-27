"""MCIC benchmark: real music-copyright cases (Park et al., ISMIR 2024).

Each case has two MIDI files (`File ID`) and a court `Decision`
(1 = infringement found, 0 = not found). MIDI is rendered to audio with
FluidSynth, converted to the same 3-channel multispec, and scored by the model.

Two scoring modes:
* `center` (original run): one ~11 s window from the middle of each song.
* `max-window` (added): slide ~11 s windows over both songs (50 % hop) and keep
  the highest similarity over all window pairs, so the score is not tied to
  whichever passage happens to sit in the middle of the file.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

from . import config as C
from .audio import audio_to_logmel, build_multichannel, crop_or_pad

_STRIP = "'’‘´`ʼ\"“”«»‚„()[]!?;,"


def normalize_filename(name: str) -> str:
    """Make CSV `File ID`s and on-disk names comparable (quotes, brackets, spaces...)."""
    name = str(name).strip()
    for ch in _STRIP:
        name = name.replace(ch, "")
    name = re.sub(r"\.{2,}", ".", name)
    return re.sub(r"_+\.", ".", name)


def _stem(name: str) -> str:
    """Matching key: lower-case letters and digits only, accents removed.

    Handles CSV/disk differences such as "L'Annee" vs "L_Annee" or "‘E.T.’" vs "ET".
    """
    stem = Path(str(name).strip()).stem
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", stem.lower())


def load_pairs(csv_path: Path = C.MCIC_CSV) -> pd.DataFrame:
    """One row per case with a clear 0/1 decision: Case, File_A, File_B, Decision."""
    raw = pd.read_csv(csv_path, dtype=str)
    raw = raw.dropna(subset=["File ID"])
    raw = raw[raw["Decision"].str.strip().isin(["0", "1"])]
    pairs = []
    for case, g in raw.groupby("Case", sort=False):
        if len(g) != 2 or g["Decision"].nunique() != 1:
            continue
        a, b = g["File ID"].tolist()
        pairs.append(dict(Case=case, File_A=a.strip(), File_B=b.strip(), Decision=int(g["Decision"].iloc[0])))
    return pd.DataFrame(pairs)


class MCICAudio:
    """Resolves File IDs to audio, rendering MIDI with FluidSynth when no WAV exists yet."""

    def __init__(self, midi_dir: Path = C.MCIC_MIDI_DIR, audio_dir: Path = C.MCIC_AUDIO_DIR,
                 soundfont: str | None = None):
        self.midi_dir, self.audio_dir = Path(midi_dir), Path(audio_dir)
        self.soundfont = soundfont
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.midi_index = {_stem(p.name): p for p in self.midi_dir.glob("*.mid")} if self.midi_dir.exists() else {}
        self.wav_index = {_stem(p.name): p for p in self.audio_dir.glob("*.wav")}

    def _render(self, midi: Path, wav: Path):
        if shutil.which("fluidsynth") is None:
            raise RuntimeError("fluidsynth not found (brew install fluid-synth / apt-get install fluidsynth).")
        if not self.soundfont:
            raise RuntimeError("Pass a General MIDI soundfont (e.g. FluidR3_GM.sf2) with --soundfont.")
        subprocess.run(["fluidsynth", "-ni", self.soundfont, str(midi), "-F", str(wav),
                        "-r", str(C.TARGET_SR)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def load(self, file_id: str) -> np.ndarray:
        key = _stem(file_id)
        wav = self.wav_index.get(key)
        if wav is None:
            midi = self.midi_index.get(key)
            if midi is None:
                raise FileNotFoundError(f"No MIDI/WAV for {file_id!r}")
            wav = self.audio_dir / f"{midi.stem}.wav"
            self._render(midi, wav)
            self.wav_index[key] = wav
        y, _ = librosa.load(wav, sr=C.TARGET_SR, mono=True)
        return y.astype(np.float32)


def song_multispec(y: np.ndarray) -> np.ndarray:
    """Full-length (3, F, T) multispec for a whole rendered song."""
    return build_multichannel(audio_to_logmel(y, C.TARGET_SR))


def windows(multi: np.ndarray, hop: int = C.TARGET_T // 2) -> list[np.ndarray]:
    t = multi.shape[-1]
    if t <= C.TARGET_T:
        return [crop_or_pad(multi)]
    starts = list(range(0, t - C.TARGET_T + 1, hop))
    if starts[-1] != t - C.TARGET_T:
        starts.append(t - C.TARGET_T)
    return [multi[..., s:s + C.TARGET_T] for s in starts]


@torch.no_grad()
def score_pair(model, multi_a: np.ndarray, multi_b: np.ndarray, mode: str = "center",
               batch: int = 64) -> float:
    """Similarity probability for one case."""
    device = next(model.parameters()).device
    if mode == "center":
        a = torch.from_numpy(crop_or_pad(multi_a))[None].to(device)
        b = torch.from_numpy(crop_or_pad(multi_b))[None].to(device)
        return torch.sigmoid(model(a, b)).item()
    if mode != "max-window":
        raise ValueError(mode)
    za = torch.cat([model.encoder(torch.from_numpy(np.stack(w)).to(device))[0]
                    for w in _chunks(windows(multi_a), batch)])
    zb = torch.cat([model.encoder(torch.from_numpy(np.stack(w)).to(device))[0]
                    for w in _chunks(windows(multi_b), batch)])
    diff = torch.abs(za[:, None, :] - zb[None, :, :]).reshape(-1, za.shape[1])
    return torch.sigmoid(model.head(diff)).max().item()


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
