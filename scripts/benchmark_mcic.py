"""Score the MCIC copyright cases with a trained checkpoint.

    python scripts/benchmark_mcic.py --mode center        # as in the original run
    python scripts/benchmark_mcic.py --mode max-window    # best-matching ~11 s windows

Needs data/mcic/MCIC-ISMIR2024.csv plus either rendered WAVs in data/mcic/AUDIO
or MIDI files in "data/mcic/MIDI FILES" and --soundfont for FluidSynth.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from musicsim import config as C, plots  # noqa: E402
from musicsim.evaluate import metrics  # noqa: E402
from musicsim.mcic import MCICAudio, load_pairs, score_pair, song_multispec  # noqa: E402
from musicsim.model import load_model  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt", default=str(C.CHECKPOINT))
p.add_argument("--csv", default=str(C.MCIC_CSV))
p.add_argument("--mode", default="center", choices=["center", "max-window"])
p.add_argument("--soundfont", default=None)
p.add_argument("--out", default=str(C.OUTPUTS / "mcic"))
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
model = load_model(args.ckpt)
pairs = load_pairs(Path(args.csv))
audio = MCICAudio(Path(args.csv).parent / "MIDI FILES", Path(args.csv).parent / "AUDIO", args.soundfont)
print(f"{len(pairs)} cases with a 0/1 decision")

rows = []
for r in tqdm(pairs.itertuples(), total=len(pairs)):
    try:
        a, b = song_multispec(audio.load(r.File_A)), song_multispec(audio.load(r.File_B))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"skip {r.Case}: {e}")
        continue
    rows.append(dict(Case=r.Case, File_A=r.File_A, File_B=r.File_B, label=r.Decision,
                     prob=score_pair(model, a, b, args.mode)))

df = pd.DataFrame(rows)
df.to_csv(out / f"scores_{args.mode}.csv", index=False)
m = metrics(df["label"].to_numpy(), df["prob"].to_numpy())
print(json.dumps(m, indent=2))
(out / f"results_{args.mode}.json").write_text(json.dumps(m, indent=2))

plots.roc(df["label"], df["prob"], f"MCIC ({args.mode})")
plt.savefig(out / f"roc_{args.mode}.png", dpi=150, bbox_inches="tight")
plots.confusion(m["confusion"], ("no infr.", "infr."), f"MCIC ({args.mode})")
plt.savefig(out / f"confusion_{args.mode}.png", dpi=150, bbox_inches="tight")
print(f"Saved to {out}")
