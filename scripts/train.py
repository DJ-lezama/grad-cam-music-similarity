"""Train the Siamese CNN on MelodySim (train split), early-stopping on validation loss.

    python scripts/train.py --ckpt checkpoints/siamese_retrained.pt
"""
import argparse
import json
from pathlib import Path

from musicsim import config as C
from musicsim.melodysim import load_manifest, split_frames
from musicsim.pairs import FixedPairs, TrainPairs
from musicsim.train import TrainConfig, train

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", default=str(C.MANIFEST))
p.add_argument("--ckpt", default=str(C.REPO_ROOT / "checkpoints" / "siamese_retrained.pt"))
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

splits = split_frames(load_manifest(Path(args.manifest)))
train_ds = TrainPairs(splits["train"], seed=args.seed)
val_ds = FixedPairs(splits["validation"])
print(f"train anchors {len(train_ds)} | validation pairs {len(val_ds)}")

cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  num_workers=args.workers, seed=args.seed)
history = train(train_ds, val_ds, cfg, Path(args.ckpt))
out = Path(args.ckpt).with_suffix(".history.json")
out.write_text(json.dumps(history, indent=2))
print(f"Best checkpoint: {args.ckpt}\nHistory: {out}")
