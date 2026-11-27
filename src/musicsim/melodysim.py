"""MelodySim: download the shards used in the original run and build multispecs.

MelodySim (Lu et al., 2025) is on Hugging Face as `amaai-lab/melodySim`
(CC-BY 3.0, 539 train / 45 validation / 28 test parquet shards, ~296 GB).
The original experiment used 103 of the *train-* shards, listed in
`data/melodysim_shards.txt`. Each segment's train/validation/test split comes
from its `relative_path` (e.g. `test/Track01962/version_2/1.wav`), not from the
shard name, so those 103 shards contain segments from all three splits.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from . import config as C
from .audio import audio_to_logmel, build_multichannel

MANIFEST_COLUMNS = ["split", "track_id", "version", "chunk_id", "relative_path",
                    "parquet_file", "row_index", "multispec_file"]


# ---------------------------------------------------------------- download --
def read_shard_list(path: Path = C.SHARD_LIST) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]


def download_shards(shards: list[str], out_dir: Path = C.PARQUET_DIR) -> list[Path]:
    """Download parquet shards one by one (skips files already present)."""
    from huggingface_hub import hf_hub_download

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in tqdm(shards, desc="MelodySim shards"):
        target = out_dir / "data" / name
        if not target.exists():
            hf_hub_download(C.MELODYSIM_REPO, filename=f"data/{name}",
                            repo_type="dataset", local_dir=out_dir)
        paths.append(target)
    return paths


# -------------------------------------------------------------- preprocess --
def parse_relative_path(relpath: str):
    """'train/Track00000/version_0/1.wav' -> ('train', 'Track00000', 'version_0', '1')."""
    parts = (relpath or "").split("/")
    if len(parts) < 4:
        return None, None, None, None
    return parts[0], parts[1], parts[2], Path(parts[3]).stem


def iter_parquet_rows(path: Path, batch_size: int = 128) -> Iterator[tuple[np.ndarray, int, str, int]]:
    """Yield (audio, sr, relative_path, row_index) from one MelodySim shard."""
    row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        cols = batch.to_pydict()
        for raw, relpath in zip(cols["audio"], cols.get("relative_path", [None] * len(cols["audio"]))):
            if isinstance(raw, dict) and raw.get("bytes") is not None:
                audio, sr = sf.read(io.BytesIO(raw["bytes"]), dtype="float32")
            elif isinstance(raw, dict) and "array" in raw:
                audio, sr = np.asarray(raw["array"], np.float32), int(raw.get("sampling_rate", C.TARGET_SR))
            else:
                audio, sr = np.asarray(raw, np.float32), C.TARGET_SR
            yield audio, sr, relpath, row
            row += 1


def multispec_filename(split, track_id, version, chunk_id) -> str:
    return f"{split}_{track_id}_{version}_{chunk_id}_mspec.npy"


def preprocess_shards(parquet_paths: list[Path], out_dir: Path = C.MULTISPEC_DIR,
                      manifest_path: Path = C.MANIFEST, dtype: str = "float16") -> pd.DataFrame:
    """Turn every segment into a (3, 128, T) multispec .npy and write a manifest.

    Already-processed files are skipped, so the step can be resumed.
    `dtype=float16` halves disk use (~0.5 MB per segment); values are in [0, 1].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in parquet_paths:
        for audio, sr, relpath, idx in tqdm(iter_parquet_rows(path), desc=Path(path).name, leave=False):
            split, track, version, chunk = parse_relative_path(relpath)
            if split is None:
                continue
            fname = multispec_filename(split, track, version, chunk)
            out = out_dir / fname
            if not out.exists():
                # The original run stored the log-mel as float16 before HPSS; keep that step.
                logmel = audio_to_logmel(audio, sr).astype(np.float16).astype(np.float32)
                np.save(out, build_multichannel(logmel).astype(dtype))
            rows.append(dict(split=split, track_id=track, version=version, chunk_id=chunk,
                             relative_path=relpath, parquet_file=Path(path).name,
                             row_index=idx, multispec_file=fname))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


# ------------------------------------------------------------------ loading --
def load_manifest(manifest_path: Path = C.MANIFEST, spec_dir: Path = C.MULTISPEC_DIR,
                  require_files: bool = True) -> pd.DataFrame:
    """Load the manifest and attach absolute multispec paths.

    With `require_files`, rows whose .npy is missing are dropped and the count is
    reported, so a partial copy of the data is visible instead of silent.
    """
    df = pd.read_csv(manifest_path, dtype={"chunk_id": str})
    if "multispec_file" not in df.columns:      # manifests from the original Colab run
        df["multispec_file"] = [multispec_filename(*r) for r in
                                df[["split", "track_id", "version", "chunk_id"]].itertuples(index=False)]
    df["path"] = [str(spec_dir / f) for f in df["multispec_file"]]
    if require_files:
        exists = df["path"].map(lambda p: Path(p).exists())
        if (~exists).any():
            print(f"[manifest] {int((~exists).sum())} of {len(df)} multispecs missing on disk; using {int(exists.sum())}.")
        df = df[exists].reset_index(drop=True)
    return df


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test")}
