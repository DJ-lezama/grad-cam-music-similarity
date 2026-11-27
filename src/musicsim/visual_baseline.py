"""Classical computer-vision baseline on spectrograms (poster Figure 1).

Before any deep learning, two recordings are compared by treating their
spectrograms as images: CLAHE contrast, directional median filters (an
HPSS-like harmonic view), Canny edges, probabilistic Hough lines, SSIM and a
multi-dimensional DTW alignment.
"""
from __future__ import annotations

import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from dtaidistance import dtw, dtw_ndim
from scipy.ndimage import median_filter
from skimage.metrics import structural_similarity

from . import config as C


def stft_db(y: np.ndarray) -> np.ndarray:
    return librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=C.N_FFT, hop_length=C.HOP_LENGTH)), ref=np.max)


def to_u8(x: np.ndarray) -> np.ndarray:
    return cv2.normalize(x.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def harmonic_view(spec_db: np.ndarray) -> np.ndarray:
    """Median denoise -> CLAHE -> median along time (keeps horizontal, harmonic ridges)."""
    u8 = to_u8(cv2.medianBlur(spec_db.astype(np.float32), 3))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8).astype(np.float32) / 255
    return median_filter(clahe, size=(1, 31))


def hough_lines(view: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(to_u8(view), 50, 150)
    out = np.zeros_like(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    for x1, y1, x2, y2 in (lines.reshape(-1, 4) if lines is not None else []):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    return out


def compare(path_a: str, path_b: str, duration: float = 9.0) -> dict:
    """Compute views and similarity scores for two audio files."""
    ya, _ = librosa.load(path_a, sr=C.TARGET_SR, duration=duration)
    yb, _ = librosa.load(path_b, sr=C.TARGET_SR, duration=duration)
    sa, sb = stft_db(ya), stft_db(yb)
    t = min(sa.shape[1], sb.shape[1])
    ha, hb = harmonic_view(sa), harmonic_view(sb)
    dist, paths = dtw_ndim.warping_paths(sa.T.astype(np.double), sb.T.astype(np.double), window=25, psi=2)
    return dict(
        y=(ya, yb), stft=(sa, sb), harmonic=(ha, hb), hough=(hough_lines(ha), hough_lines(hb)),
        ssim=structural_similarity(ha[:, :t], hb[:, :t], data_range=1.0),
        dtw_distance=dist, dtw_path=dtw.best_path(paths),
    )


def plot(result: dict, names=("A", "B")):
    fig, axes = plt.subplots(2, 4, figsize=(18, 6), constrained_layout=True)
    for r, name in enumerate(names):
        librosa.display.waveshow(result["y"][r], sr=C.TARGET_SR, ax=axes[r, 0])
        for c, (key, cmap) in enumerate([("stft", "magma"), ("harmonic", "viridis"), ("hough", "gray")], 1):
            axes[r, c].imshow(result[key][r], aspect="auto", origin="lower", cmap=cmap)
            axes[r, c].set(xticks=[], yticks=[], title=f"{name}: {key}")
        axes[r, 0].set_title(f"{name}: waveform")
    fig.suptitle(f"SSIM (harmonic view) = {result['ssim']:.3f}   |   DTW distance = {result['dtw_distance']:.1f}")
    return fig
