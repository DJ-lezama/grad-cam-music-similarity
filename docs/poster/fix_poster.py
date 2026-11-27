"""Patch typos and outdated numbers in the poster PDF (exported from PowerPoint).

The original slide source is not available, so affected text lines are removed
and re-typeset in the same font (Aptos, shipped with Microsoft Office).

    python docs/poster/fix_poster.py docs/poster/poster_original.pdf docs/poster/poster.pdf
"""
import sys

import pymupdf

FONT_DIR = "/Applications/Microsoft PowerPoint.app/Contents/Resources/DFonts"
FONTS = {"r": "Aptos.ttf", "b": "Aptos-Bold.ttf", "i": "Aptos-Italic.ttf"}

# (baseline_y, x0, size, align, right_or_center, [lines of [(text, font), ...]], line_gap)
# align: "left" | "justify" (to right edge) | "center" (around x)
EDITS = [
    # headings
    (140.23, 292.4, 10.8, "center", 365.25, [[("Evaluation Results (MelodySim)", "b")]]),
    (251.90, 99.7, 10.8, "center", 127.95, [[("Methodology", "b")]]),
    (355.43, 305.8, 10.8, "center", 365.65, [[("Explainability (Grad-CAM)", "b")]]),
    # introduction
    (144.77, 21.1, 7.2, "justify", 255.5, [[("detection show strong performance but behave as ", "r"), ("black-box", "b")]]),
    (180.82, 21.1, 7.2, "left", None, [[("criteria", "b"), (", and ", "r"), ("blurry definitions ", "b"),
                                        ("of originality and creativity.", "r")]]),
    # visual baseline: DWT -> DTW (dynamic time warping)
    (396.07, 6.1, 7.2, "justify", 244.9, [[("detection, Hough transforms, DTW translation alignment) to inspect", "r")]]),
    (536.10, 72.7, 5.8, "center", 136.25, [[("edge detection, Hough lines, and DTW alignment.", "i")]]),
    # methodology: segment count from the training run
    (309.65, 78.5, 7.2, "left", None, [[("8,928 MelodySim training segments.", "r")]]),
    # evaluation results
    (155.97, 287.1, 7.2, "justify", 447.4, [
        [("The model reached 0.83 ROC-AUC (73% accuracy)", "r")],
        [("on controlled version-based validation pairs.", "r")],
    ], [0, 8.64]),
    (182.66, 287.1, 7.2, "justify", 447.4, [[("Ablation tests show the combined 3-channel input", "r")]]),
    # benchmarking: flipped-score bullet
    (303.59, 285.2, 7.2, "justify", 450.1, [
        [("Inverting the score only mirrors ROC-AUC", "r")],
        [("(1 − 0.44 = 0.56): ranking stays near chance,", "r")],
        [("confirming that the learned representation", "r")],
        [("does not map cleanly to legal similarity.", "r")],
    ], [0, 8.64, 18.03, 26.69]),
    (322.04, 561.9, 5.8, "center", 589.35, [[("real copyright cases.", "i")]]),
    # future work
    (343.07, 738.6, 7.2, "left", None, [[("variations (like an augmented MCIC dataset).", "r")]]),
    (406.53, 738.6, 7.2, "justify", 946.1, [[("Develop and evaluate models explicitly optimized for human-", "r")]]),
    # references
    (491.59, 741.3, 5.8, "left", None, [[("[3] Park, S., Kim, H., Jung, J., Park, J., Kim, J., and Nam, J. (2024). ", "r"),
                                         ("Quantitative", "i")]]),
    (527.64, 741.3, 5.8, "left", None, [[("quality assessment: From error visibility to structural similarity. ", "i"),
                                         ("IEEE", "r")]]),
]



def draw_line(page, fonts, runs, x0, y, size, align, ref, last=False):
    width = lambda t, f: fonts[f].text_length(t, fontsize=size)
    if align == "justify" and not last:
        ws = [(w, f) for t, f in runs for w in t.split(" ") if w]
        total = sum(width(w, f) for w, f in ws)
        gap = (ref - x0 - total) / max(len(ws) - 1, 1)
        x = x0
        for w, f in ws:
            page.insert_text((x, y), w, fontsize=size, fontname=f"apt{f}", color=(0, 0, 0))
            x += width(w, f) + gap
        return
    full = sum(width(t, f) for t, f in runs)
    x = ref - full / 2 if align == "center" else x0
    for t, f in runs:
        page.insert_text((x, y), t, fontsize=size, fontname=f"apt{f}", color=(0, 0, 0))
        x += width(t, f)


def main(src, dst):
    doc = pymupdf.open(src)
    page = doc[0]
    fonts = {k: pymupdf.Font(fontfile=f"{FONT_DIR}/{v}") for k, v in FONTS.items()}

    # 1) remove the old text of every edited line (text only; images and shapes stay)
    for e in EDITS:
        y, size, lines = e[0], e[2], e[5]
        offsets = e[6] if len(e) > 6 else [0]
        for off in offsets[: len(lines)]:
            b = y + off
            page.add_redact_annot(pymupdf.Rect(e[1] - 0.5 if e[3] != "center" else e[1] - 5, b - size * 0.55,
                                               page.rect.width if e[3] != "center" else e[4] * 2 - e[1] + 5,
                                               b - size * 0.15) & _column(e[1]), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)

    # 2) typeset the corrected lines
    for k, v in FONTS.items():
        page.insert_font(fontname=f"apt{k}", fontfile=f"{FONT_DIR}/{v}")
    for e in EDITS:
        y, x0, size, align, ref, lines = e[:6]
        offsets = e[6] if len(e) > 6 else [0]
        for n, (runs, off) in enumerate(zip(lines, offsets)):
            draw_line(page, fonts, runs, x0, y + off, size, align, ref, last=(n == len(lines) - 1 and len(lines) > 1))
    doc.save(dst, garbage=3, deflate=True)


def _column(x):
    """Restrict redaction to the poster column containing x (avoids touching neighbours)."""
    for lo, hi in [(0, 262), (262, 480), (480, 715), (715, 960)]:
        if lo <= x < hi:
            return pymupdf.Rect(lo, 0, hi, 540)
    return pymupdf.Rect(0, 0, 960, 540)


if __name__ == "__main__":
    main(*sys.argv[1:3])
