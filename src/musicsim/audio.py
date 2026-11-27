"""Audio -> 3-channel spectrogram ("multispec") used by the Siamese CNN.

Channel 0: log-mel spectrogram (dB), min-max normalised to [0, 1]
Channel 1: harmonic component of HPSS, normalised to [0, 1]
Channel 2: percussive component of HPSS, normalised to [0, 1]

Note: HPSS is applied to the *shifted dB* log-mel (made non-negative), not to a
linear-magnitude spectrogram. That is what the original run did, and the
checkpoint was trained on it, so it is kept as is (see README, "Known issues").
"""
from __future__ import annotations

import numpy as np
import librosa

from . import config as C


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    # channels-first or channels-last
    return audio.mean(axis=0) if audio.shape[0] < audio.shape[-1] else audio.mean(axis=-1)


def audio_to_logmel(audio: np.ndarray, sr: int) -> np.ndarray:
    """Waveform -> log-mel in dB, shape (N_MELS, T)."""
    audio = to_mono(np.asarray(audio, dtype=np.float32))
    if sr != C.TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=C.TARGET_SR)
    mel = librosa.feature.melspectrogram(
        y=audio, sr=C.TARGET_SR, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH,
        n_mels=C.N_MELS, power=C.POWER,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    mx = x.max()
    return x / mx if mx > 0 else x


def build_multichannel(logmel_db: np.ndarray) -> np.ndarray:
    """(F, T) log-mel in dB -> (3, F, T) float32 multispec."""
    spec = logmel_db.astype(np.float32)
    harm, perc = librosa.decompose.hpss(spec - spec.min())
    return np.stack([_minmax(spec), _minmax(harm), _minmax(perc)]).astype(np.float32)


def crop_or_pad(multi: np.ndarray, target_t: int = C.TARGET_T, offset: int | None = None) -> np.ndarray:
    """Center-crop (or crop at `offset`) / symmetrically zero-pad the time axis."""
    t = multi.shape[-1]
    if t > target_t:
        start = (t - target_t) // 2 if offset is None else min(max(offset, 0), t - target_t)
        return multi[..., start:start + target_t]
    if t < target_t:
        left = (target_t - t) // 2
        pad = [(0, 0)] * (multi.ndim - 1) + [(left, target_t - t - left)]
        return np.pad(multi, pad, mode="constant")
    return multi


def audio_to_multispec(audio: np.ndarray, sr: int, crop: bool = True) -> np.ndarray:
    """Waveform -> (3, F, TARGET_T) multispec (or full length if crop=False)."""
    multi = build_multichannel(audio_to_logmel(audio, sr))
    return crop_or_pad(multi) if crop else multi
