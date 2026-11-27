"""Grad-CAM for the Siamese network.

The target is the similarity logit. Gradients are taken with respect to the last
conv feature maps of each branch, so each input gets its own heat map showing
which time-frequency regions pushed the pair towards "similar".
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class SiameseGradCAM:
    def __init__(self, model):
        self.model = model.eval()

    @staticmethod
    def _cam(fmap: torch.Tensor, grads: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
        weights = grads.mean(dim=(2, 3), keepdim=True)            # channel importance
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()

    def __call__(self, spec_a: torch.Tensor, spec_b: torch.Tensor):
        """spec_a, spec_b: (C, F, T) or (1, C, F, T).

        Returns (heatmap_a, heatmap_b, probability) with heat maps shaped (F, T).
        """
        device = next(self.model.parameters()).device
        a = (spec_a if spec_a.dim() == 4 else spec_a[None]).to(device)
        b = (spec_b if spec_b.dim() == 4 else spec_b[None]).to(device)
        size = tuple(a.shape[-2:])
        with torch.enable_grad():
            za, fa = self.model.encoder(a)
            zb, fb = self.model.encoder(b)
            logit = self.model.head(torch.abs(za - zb)).squeeze()
            ga, gb = torch.autograd.grad(logit, [fa, fb])
        return self._cam(fa, ga, size), self._cam(fb, gb, size), torch.sigmoid(logit).item()
