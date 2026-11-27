# From Versionness to Infringement?
### A Grad-CAM study of audio-based musical similarity models

**Question:** when a deep audio-similarity model says two songs are "similar", *what is it actually looking at*, and does
that match what matters in a music-copyright dispute?

**Approach:** train a Siamese CNN on 3-channel spectrograms (log-mel + HPSS harmonic + HPSS percussive) from
[MelodySim](https://huggingface.co/datasets/amaai-lab/melodySim), test it on real court cases from MCIC, and use
**Grad-CAM** to see which time-frequency regions drive its decisions.

**Finding:** the model learns **version identity** (recognising re-arrangements of the *same* segment), not melodic
similarity. It does well in-domain (ROC-AUC 0.83) but is at chance on real copyright cases (ROC-AUC 0.44), and Grad-CAM
shows attention on low-level, rendition-specific texture rather than melodic contour.

![poster](docs/poster.png)

<sub>Course project, Artificial Vision (LIS4042), Universidad de las Américas Puebla, Fall 2025. The audio/music side of
the project was designed and carried out independently.</sub>

---

## Method

```
audio (≈9 s @16 kHz)
  └─ log-mel (128 × T) ─┬─ min-max ───────────────┐
                        └─ HPSS ─┬─ harmonic ──────┤→ (3, 128, 685) multispec
                                 └─ percussive ────┘
two multispecs ─► shared CNN encoder ─► |z_a − z_b| ─► MLP ─► similarity logit
                          └─ last conv maps ─► Grad-CAM heat map per input
```

| Step | Details |
|---|---|
| Visual baseline | Spectrograms as images: CLAHE, directional median filters, Canny, Hough lines, SSIM, multi-dimensional DTW |
| Data | 103 MelodySim shards → 28,826 segments, 351 tracks; splits taken from each segment's path |
| Pairs | positive = same segment in two different versions; negative = segments from different tracks |
| Model | 4 conv blocks (16→128 ch), GAP, 256-d L2-normalised embedding, MLP on \|z_a − z_b\| (≈164 k params) |
| Training | BCE, Adam lr 1e-4, ReduceLROnPlateau, early stopping (stopped at epoch 18, best val loss at epoch 10) |
| Evaluation | MelodySim validation pairs, inference-time channel ablation, 95 MCIC court cases |
| Explainability | Grad-CAM on the similarity logit, one map per branch |

## Results (original run)

**MelodySim, validation split (2,099 pairs):**

| metric | value |
|---|---|
| ROC-AUC | **0.833** |
| accuracy (threshold 0.5) | 0.732 |
| precision / recall, *similar* | 0.752 / 0.694 |
| precision / recall, *dissimilar* | 0.716 / 0.770 |

**Channel ablation** (channels zeroed at inference; ROC-AUC): all three 0.813 · log-mel + harmonic 0.722 · log-mel +
percussive 0.652 · log-mel 0.610 · harmonic 0.591 · harmonic + percussive 0.581 · percussive 0.526.
The combination matters: no single view carries the decision.

<img src="docs/figures/fig2_melodysim_roc.png" width="330"> <img src="docs/figures/fig2_ablation.png" width="470">

**MCIC, real copyright cases (95 evaluated):** ROC-AUC **0.442**. The model rates nearly every pair as similar
(32/32 infringement cases and 62/63 non-infringement cases called "similar"), consistent with it reacting to shared
rendering (every MCIC song is synthesised from MIDI with the same soundfont) rather than to melody.

<img src="docs/figures/fig3_mcic_roc.png" width="330"> <img src="docs/figures/fig3_mcic_confusion.png" width="330">

### Grad-CAM

| MelodySim, similar pair | MCIC, false positive |
|---|---|
| <img src="docs/figures/fig4_gradcam_melodysim_similar_A.png" width="400"> | <img src="docs/figures/fig5_gradcam_mcic_fp_A.png" width="400"> |
| <img src="docs/figures/fig4_gradcam_melodysim_similar_B.png" width="400"> | <img src="docs/figures/fig5_gradcam_mcic_fp_B.png" width="400"> |

* **MelodySim:** attention is scattered over note onsets and dense spectral spots and is not aligned between the two
  versions: evidence of version-specific cues, not melodic structure.
* **False negatives / positives:** misses ignore melodic contours; false alarms highlight incidental timbre overlap.
* **MCIC:** occasionally highlights musically relevant note clusters, but inconsistently.

### Interpretation

Musical similarity forms a hierarchy: *identical audio* → *different versions of the same song* → *melodically similar
songs* → *vaguely similar (style, harmony)* → *unrelated*. The training pairs only test level 2 (versionness), while legal
similarity lives at levels 3–4. The in-domain score therefore does not transfer.

## Reproducing

```bash
git clone https://github.com/DJ-lezama/grad-cam-music-similarity.git
cd grad-cam-music-similarity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**1. Trained weights** (≈0.7 MB): download `siamese_best.pt` from the
[latest release](https://github.com/DJ-lezama/grad-cam-music-similarity/releases) into `checkpoints/`.

```bash
gh release download --repo DJ-lezama/grad-cam-music-similarity -p siamese_best.pt -D checkpoints
```

**2. MelodySim** (public, CC-BY 3.0). The full set used here is ~50 GB; `--quick N` takes only the first N shards
(~0.5 GB each) for a fast end-to-end check.

```bash
python scripts/download_melodysim.py --quick 3
python scripts/preprocess_melodysim.py
```

**3. MCIC** (optional). Place the dataset in `data/mcic/` as `MCIC-ISMIR2024.csv` plus either rendered WAVs in
`data/mcic/AUDIO/` or MIDI files in `data/mcic/MIDI FILES/` (rendered with FluidSynth: pass a General MIDI soundfont such
as FluidR3_GM via `--soundfont`).

**4. Run**

```bash
python scripts/evaluate_melodysim.py --split test --ablation   # metrics + ablation on held-out test pairs
python scripts/gradcam_examples.py --split test                # Grad-CAM overlays
python scripts/benchmark_mcic.py --mode center                 # MCIC, as in the original run
python scripts/benchmark_mcic.py --mode max-window             # MCIC, best-matching windows
python scripts/train.py                                        # retrain from scratch (GPU recommended)
```

Or open [`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb), which shows the saved results and re-runs each step
when data and weights are present. Set `MUSICSIM_DATA=/path/to/data` to keep the large files elsewhere.

## Repository layout

```
src/musicsim/
  audio.py            log-mel + HPSS → 3-channel multispec
  melodysim.py        shard download, preprocessing, manifest
  pairs.py            training / fixed evaluation pair datasets
  model.py            Siamese CNN, checkpoint loading
  train.py            training loop
  evaluate.py         metrics and channel ablation
  gradcam.py          Grad-CAM for both Siamese branches
  mcic.py             MCIC loading, MIDI rendering, center / max-window scoring
  visual_baseline.py  classical CV baseline
  plots.py            figures
scripts/              command-line entry points
notebooks/            walkthrough
data/                 shard list + reference manifest of the original run (large data is git-ignored)
docs/                 poster and figures
```

## Changes from the original Colab notebook

The model, preprocessing and the released checkpoint are unchanged. The code was reorganised and these issues fixed:

| Issue in the original notebook | Effect | Now |
|---|---|---|
| Evaluation pairs were re-sampled at random on every pass | metrics varied between runs (0.833 vs 0.813 ROC-AUC on the same split) | fixed, seeded pairs (`FixedPairs`) |
| A segment with no other version available was paired with itself as a "positive" | identical inputs counted as easy positives | such anchors are skipped |
| Validation split used both for early stopping and for the reported metrics; test split unused | optimistic estimate | `evaluate_melodysim.py --split test` |
| MCIC loop thresholded the raw **logit** at 0.5 (named `prob`) | confusion matrix used a different cut-off than intended (ROC-AUC unaffected) | sigmoid applied before thresholding |
| MCIC scored only the middle ~11 s of each full song | the disputed passage may not be in that window | optional `max-window` scoring |
| Training data depended on how many files a Colab session managed to copy locally | training set size varied between sessions | manifest-driven; missing files are reported |
| Visual baseline passed a 0–255 image as dB to HPSS | meaningless HPSS view | operates on the dB spectrogram |

Note on the "flipped score" check from the poster: reversing the score gives ROC-AUC = 1 − 0.442 = 0.558 by definition,
so it does not add information beyond "the ranking is close to chance".

### Known limitations

* HPSS is applied to the shifted dB log-mel rather than a linear spectrogram; kept for compatibility with the checkpoint.
* MCIC songs are MIDI renderings with one soundfont, while MelodySim is rendered audio with varied instrumentation.
* Grad-CAM at the last conv layer has coarse resolution (1/8 of the input in each axis).

## Future work

* Train on pairs with stylistic and melodic variation (e.g. augmented MCIC-like data) rather than versions only.
* Sequence models (transformers) that encode long-range melodic structure.
* Combine symbolic (MIDI / pitch contour) features with spectrogram input.
* Check attribution with SHAP and occlusion in addition to Grad-CAM.
* Optimise directly for human-aligned similarity rather than version matching.

## References

1. T. Lu, C.-M. Geist, J. Melechovsky, A. Roy, D. Herremans. *MelodySim: Measuring melody-aware music similarity for
   plagiarism detection.* arXiv:2505.20979, 2025.
2. D. FitzGerald. *Harmonic/percussive separation using median filtering.* DAFx, 2010.
3. S. Park, H. Kim, J. Jung, J. Park, J. Kim, J. Nam. *Quantitative analysis of melodic similarity in music copyright
   infringement cases.* ISMIR, 2024.
4. Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image quality assessment: From error visibility to structural
   similarity.* IEEE TIP 13(4), 2004.
5. R. R. Selvaraju et al. *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV, 2017.

Datasets are not redistributed here; see their own licences. MelodySim is CC-BY 3.0.
