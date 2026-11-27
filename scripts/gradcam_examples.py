"""Save Grad-CAM overlays for a few MelodySim pairs (similar and dissimilar).

    python scripts/gradcam_examples.py --split test --n 3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from musicsim import config as C, plots  # noqa: E402
from musicsim.gradcam import SiameseGradCAM  # noqa: E402
from musicsim.melodysim import load_manifest, split_frames  # noqa: E402
from musicsim.model import load_model  # noqa: E402
from musicsim.pairs import FixedPairs  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--manifest", default=str(C.MANIFEST))
p.add_argument("--ckpt", default=str(C.CHECKPOINT))
p.add_argument("--split", default="test")
p.add_argument("--n", type=int, default=3, help="examples per label")
p.add_argument("--out", default=str(C.OUTPUTS / "gradcam"))
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
ds = FixedPairs(split_frames(load_manifest(Path(args.manifest)))[args.split])
cam = SiameseGradCAM(load_model(args.ckpt))

for label, name in [(1, "similar"), (0, "dissimilar")]:
    picks = [k for k, (_, _, y) in enumerate(ds.pairs) if y == label][: args.n]
    for k in picks:
        a, b, _ = ds[k]
        ha, hb, prob = cam(a, b)
        d = ds.describe(k)
        fig = plots.gradcam_pair(a, b, ha, hb, f"{name}: {d['a']} vs {d['b']}  (p_similar = {prob:.2f})")
        fig.savefig(out / f"{name}_{k}.png", dpi=130)
        plt.close(fig)
print(f"Saved to {out}")
