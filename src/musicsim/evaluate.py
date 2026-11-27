"""Inference, metrics and channel ablation."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config as C


@torch.no_grad()
def predict(model, dataset, batch_size: int = 32, zero_channels: Iterable[int] = (),
            num_workers: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, probabilities). `zero_channels` blanks input channels on both branches."""
    device = next(model.parameters()).device
    model.eval()
    labels, probs = [], []
    zero_channels = list(zero_channels)
    for a, b, y in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers), leave=False):
        a, b = a.to(device), b.to(device)
        if zero_channels:
            a[:, zero_channels] = 0.0
            b[:, zero_channels] = 0.0
        probs.append(torch.sigmoid(model(a, b)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    out = dict(
        n=len(labels),
        accuracy=accuracy_score(labels, preds),
        precision=precision_score(labels, preds, zero_division=0),
        recall=recall_score(labels, preds, zero_division=0),
        f1=f1_score(labels, preds, zero_division=0),
        confusion=confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
    )
    out["roc_auc"] = roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan")
    return out


ABLATIONS = [
    ("all channels", []),
    ("log-mel only", [1, 2]),
    ("harmonic only", [0, 2]),
    ("percussive only", [0, 1]),
    ("log-mel + harmonic", [2]),
    ("log-mel + percussive", [1]),
    ("harmonic + percussive", [0]),
]


def ablation(model, dataset, **kw) -> list[dict]:
    """Zero out channels at inference time (no retraining).

    This measures how much the *trained* model relies on each channel, not how
    well a model trained on that channel alone would do.
    """
    rows = []
    for name, zero in ABLATIONS:
        y, p = predict(model, dataset, zero_channels=zero, **kw)
        m = metrics(y, p)
        rows.append(dict(inputs=name, roc_auc=m["roc_auc"], accuracy=m["accuracy"]))
        print(f"{name:<24} ROC-AUC {m['roc_auc']:.4f}  acc {m['accuracy']:.4f}")
    return rows
