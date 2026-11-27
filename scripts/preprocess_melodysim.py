"""Convert downloaded MelodySim shards into 3-channel multispecs + a manifest.

    python scripts/preprocess_melodysim.py
"""
import argparse
from pathlib import Path

from musicsim import config as C
from musicsim.melodysim import preprocess_shards

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--parquets", default=str(C.PARQUET_DIR / "data"))
p.add_argument("--out", default=str(C.MULTISPEC_DIR))
p.add_argument("--manifest", default=str(C.MANIFEST))
p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
args = p.parse_args()

paths = sorted(Path(args.parquets).glob("*.parquet"))
if not paths:
    raise SystemExit(f"No parquet files in {args.parquets}. Run scripts/download_melodysim.py first.")
df = preprocess_shards(paths, Path(args.out), Path(args.manifest), args.dtype)
print(df["split"].value_counts().to_string())
print(f"Wrote {len(df)} rows to {args.manifest}")
