"""Siamese CNN (architecture identical to the trained checkpoint)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class Encoder(nn.Module):
    """4 conv blocks -> global average pool -> L2-normalised embedding.

    `forward` also returns the last conv feature maps, which Grad-CAM uses.
    """

    def __init__(self, in_ch: int = C.IN_CHANNELS, emb_dim: int = C.EMB_DIM):
        super().__init__()

        def block(cin, cout, pool=True):
            layers = [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]
            return layers + ([nn.MaxPool2d(2)] if pool else [])

        self.features = nn.Sequential(
            *block(in_ch, 16), *block(16, 32), *block(32, 64), *block(64, 128, pool=False)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor):
        fmap = self.features(x)                         # (B, 128, F/8, T/8)
        emb = self.fc(self.gap(fmap).flatten(1))        # (B, emb_dim)
        return F.normalize(emb, p=2, dim=1), fmap


class SiameseNet(nn.Module):
    """Shared encoder; |z_a - z_b| -> MLP -> similarity logit."""

    def __init__(self, in_ch: int = C.IN_CHANNELS, emb_dim: int = C.EMB_DIM):
        super().__init__()
        self.encoder = Encoder(in_ch, emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        za, _ = self.encoder(a)
        zb, _ = self.encoder(b)
        return self.head(torch.abs(za - zb)).squeeze(1)   # logits, (B,)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str | Path = C.CHECKPOINT, device: torch.device | None = None) -> SiameseNet:
    device = device or get_device()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path}. Download siamese_best.pt from the "
            "GitHub release (see README) or train one with scripts/train.py."
        )
    model = SiameseNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.eval()
