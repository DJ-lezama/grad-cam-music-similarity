"""Figures used in the README and notebook."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def roc(labels, probs, title="ROC curve", ax=None):
    ax = ax or plt.subplots(figsize=(5, 5))[1]
    fpr, tpr, _ = roc_curve(labels, probs)
    from sklearn.metrics import auc
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    return ax


def confusion(cm, labels=("dissimilar", "similar"), title="Confusion matrix", ax=None):
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    cm = np.asarray(cm)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=[f"pred {l}" for l in labels],
           yticklabels=[f"true {l}" for l in labels], title=title)
    return ax


def ablation_bars(rows, ax=None):
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    names = [r["inputs"] for r in rows]
    vals = [r["roc_auc"] for r in rows]
    ax.barh(names, vals, color=["#2e7d32"] + ["#90caf9"] * (len(rows) - 1))
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center")
    ax.invert_yaxis()
    ax.set(xlim=(0.45, max(vals) + 0.05), xlabel="ROC-AUC", title="Channel ablation (inference-time)")
    return ax


def gradcam_overlay(spec, heatmap, title="", ax=None, alpha=0.55):
    """spec: (C, F, T) tensor/array; channel 0 (log-mel) is the background."""
    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    bg = spec[0].detach().cpu().numpy() if hasattr(spec, "detach") else np.asarray(spec)[0]
    ax.imshow(bg, aspect="auto", origin="lower", cmap="gray")
    im = ax.imshow(heatmap, aspect="auto", origin="lower", cmap="jet", alpha=alpha, vmin=0, vmax=1)
    ax.set(title=title, xlabel="time frame", ylabel="mel bin")
    return im


def gradcam_pair(spec_a, spec_b, cam_a, cam_b, suptitle=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.6), constrained_layout=True)
    gradcam_overlay(spec_a, cam_a, "A", axes[0])
    im = gradcam_overlay(spec_b, cam_b, "B", axes[1])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Grad-CAM")
    fig.suptitle(suptitle)
    return fig
