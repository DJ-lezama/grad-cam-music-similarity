"""Evaluate a checkpoint on fixed MelodySim pairs, with optional channel ablation.

    python scripts/evaluate_melodysim.py --split test --ablation
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from musicsim import config as C, plots  # noqa: E402
from musicsim.evaluate import ablation, metrics, predict  # noqa: E402
from musicsim.melodysim import load_manifest, split_frames  # noqa: E402
from musicsim.model import load_model  # noqa: E402
from musicsim.pairs import FixedPairs  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", default=str(C.MANIFEST))
p.add_argument("--ckpt", default=str(C.CHECKPOINT))
p.add_argument("--split", default="test", choices=["train", "validation", "test"])
p.add_argument("--ablation", action="store_true")
p.add_argument("--seed", type=int, default=1234)
p.add_argument("--workers", type=int, default=2)
p.add_argument("--out", default=str(C.OUTPUTS / "melodysim"))
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model = load_model(args.ckpt)
ds = FixedPairs(split_frames(load_manifest(Path(args.manifest)))[args.split], seed=args.seed)
print(f"{args.split}: {len(ds)} pairs")

labels, probs = predict(model, ds, num_workers=args.workers)
m = metrics(labels, probs)
print(json.dumps({k: v for k, v in m.items()}, indent=2))
result = {"split": args.split, "metrics": m}

plots.roc(labels, probs, f"MelodySim {args.split}")
plt.savefig(out / f"roc_{args.split}.png", dpi=150, bbox_inches="tight")
plots.confusion(m["confusion"], title=f"MelodySim {args.split}")
plt.savefig(out / f"confusion_{args.split}.png", dpi=150, bbox_inches="tight")

if args.ablation:
    rows = ablation(model, ds, num_workers=args.workers)
    result["ablation"] = rows
    plots.ablation_bars(rows)
    plt.savefig(out / f"ablation_{args.split}.png", dpi=150, bbox_inches="tight")

(out / f"results_{args.split}.json").write_text(json.dumps(result, indent=2))
print(f"Saved to {out}")
