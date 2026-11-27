"""Download the MelodySim shards used in the original run (~50 GB) or a quick subset.

    python scripts/download_melodysim.py --quick 3     # ~1.5 GB, for a smoke test
    python scripts/download_melodysim.py               # all 103 shards
"""
import argparse
from pathlib import Path

from musicsim import config as C
from musicsim.melodysim import download_shards, read_shard_list

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--quick", type=int, default=None, help="only download the first N shards")
p.add_argument("--out", default=str(C.PARQUET_DIR))
args = p.parse_args()

shards = read_shard_list()
if args.quick:
    shards = shards[: args.quick]
print(f"Downloading {len(shards)} shard(s) from {C.MELODYSIM_REPO} (~0.48 GB each) to {args.out}")
download_shards(shards, Path(args.out))
