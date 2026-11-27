"""Pair datasets for the Siamese CNN.

Positive pair: the same (track_id, chunk_id) segment in two *different* versions
               (e.g. `original` vs `version_1`) -> label 1.
Negative pair: segments from two different tracks -> label 0.

Because positives are "same segment, different rendition", the task the model is
trained on is *version identity* rather than general melodic similarity, which
is the central point of the project.

Changes from the original notebook (see README):
* evaluation pairs are fixed once with a seed, so metrics are repeatable
  (the notebook re-sampled them on every pass over the loader);
* anchors with no other version available are skipped for positive pairs
  (the notebook paired such a segment with itself and labelled it similar).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config as C
from .audio import crop_or_pad


def load_multispec(path: str, target_t: int = C.TARGET_T) -> torch.Tensor:
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    return torch.from_numpy(crop_or_pad(arr, target_t))


class _PairIndex:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.tracks = self.df["track_id"].to_numpy()
        self.versions = self.df["version"].to_numpy()
        groups: dict[tuple, list[int]] = defaultdict(list)
        for i, key in enumerate(zip(self.df["track_id"], self.df["chunk_id"])):
            groups[key].append(i)
        self.keys = list(zip(self.df["track_id"], self.df["chunk_id"]))
        self.groups = groups

    def positives(self, i: int) -> list[int]:
        return [j for j in self.groups[self.keys[i]] if self.versions[j] != self.versions[i]]

    def negative(self, i: int, rng: np.random.Generator) -> int:
        if len(set(self.tracks)) < 2:
            raise ValueError("Need at least two tracks to sample negative pairs.")
        while True:
            j = int(rng.integers(len(self.df)))
            if self.tracks[j] != self.tracks[i]:
                return j


class TrainPairs(Dataset):
    """Each item draws a positive (p=0.5) or negative pair for anchor i on the fly."""

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.idx = _PairIndex(df)
        self.anchors = [i for i in range(len(self.idx.df)) if self.idx.positives(i)]
        self.paths = self.idx.df["path"].tolist()
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, k):
        i = self.anchors[k]
        if self.rng.random() < 0.5:
            j, y = int(self.rng.choice(self.idx.positives(i))), 1.0
        else:
            j, y = self.idx.negative(i, self.rng), 0.0
        return load_multispec(self.paths[i]), load_multispec(self.paths[j]), torch.tensor(y)


class FixedPairs(Dataset):
    """Deterministic, balanced evaluation pairs (alternating positive / negative)."""

    def __init__(self, df: pd.DataFrame, seed: int = 1234):
        idx = _PairIndex(df)
        rng = np.random.default_rng(seed)
        self.paths = idx.df["path"].tolist()
        self.meta = idx.df
        self.pairs: list[tuple[int, int, float]] = []
        for i in range(len(idx.df)):
            if i % 2 == 0:
                pos = idx.positives(i)
                if pos:
                    self.pairs.append((i, int(rng.choice(pos)), 1.0))
            else:
                self.pairs.append((i, idx.negative(i, rng), 0.0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, y = self.pairs[k]
        return load_multispec(self.paths[i]), load_multispec(self.paths[j]), torch.tensor(y)

    def describe(self, k) -> dict:
        i, j, y = self.pairs[k]
        a, b = self.meta.iloc[i], self.meta.iloc[j]
        return {"a": f"{a.track_id}/{a.version}/{a.chunk_id}",
                "b": f"{b.track_id}/{b.version}/{b.chunk_id}", "label": int(y)}
