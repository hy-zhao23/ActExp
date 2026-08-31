"""
Stage-1 pretraining effect on generation metrics — two separate bar plots.

Writes:
    out/plots/stage1_effect_500k.pdf   (+ Stage-1 vs w/o Stage-1 at 500k)
    out/plots/stage1_effect_951k.pdf   (+ Stage-1 vs w/o Stage-1 at 951k)

Each plot shows 4 generation metrics (Tok-F1, ROUGE-L, chrF, BERTScore).
Numbers come from metric.md aggregate (nostop=False, default scoring).

Usage:
    python experiments/plot/stage1_effect_plot.py [--out-dir out/plots]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


INTER_DIR = Path.home() / ".fonts" / "Inter"


def setup_inter():
    for w in ("Regular", "Medium", "Bold"):
        f = INTER_DIR / f"Inter-{w}.ttf"
        if f.exists():
            fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"] = "Inter"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"]        = 10
    plt.rcParams["axes.labelsize"]   = 10
    plt.rcParams["xtick.labelsize"]  = 10
    plt.rcParams["ytick.labelsize"]  = 10


# ──────────────────────────────────────────────────────────────────────────────
# Source rows:
#   500k          ours_v1-2_500k             0.302 / 0.291 / 0.288 / 0.354
#   500k_nostage1 ours_v1-2_500k_nostage1    0.297 / 0.284 / 0.279 / 0.340
#   951k          ours_v1-2_500k_fullqa      0.321 / 0.308 / 0.306 / 0.374
#   951k_nostage1 ours_v1-2_fullqa_nostage1  0.298 / 0.287 / 0.285 / 0.356
# ──────────────────────────────────────────────────────────────────────────────

METRICS = ["Tok-F1", "ROUGE-L", "chrF", "BERTScore"]

DATA = {
    "500k": {
        "+ Stage-1":  [0.302, 0.291, 0.288, 0.354],
        "w/o Stage-1": [0.297, 0.284, 0.279, 0.340],
    },
    "951k": {
        "+ Stage-1":  [0.321, 0.308, 0.306, 0.374],
        "w/o Stage-1": [0.298, 0.287, 0.285, 0.356],
    },
}

COLOR_WO   = "#B9D6E9"  # soft blue (w/o Stage-1)
COLOR_PLUS = "#9A9DC7"  # lavender (+ Stage-1)


def plot_single(size_label: str, data: dict, out_pdf: Path,
                ylim: tuple = (0.26, 0.38)) -> None:
    """One figure per Stage-2 training size — grouped bars for 4 metrics."""
    x = np.arange(len(METRICS))
    bar_w = 0.22          # narrower bars
    offset = 0.14         # half-spacing between the two bars in a group

    wo   = data["w/o Stage-1"]
    plus = data["+ Stage-1"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    bars_wo = ax.bar(x - offset, wo,   bar_w, color=COLOR_WO,   label="w/o Stage-1")
    bars_p  = ax.bar(x + offset, plus, bar_w, color=COLOR_PLUS, label="+ Stage-1")

    # Value labels: smaller font + nudge outward (left bar text slightly left,
    # right bar text slightly right) to avoid overlap between the two within a group.
    for b in bars_wo:
        h = b.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(-2, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)
    for b in bars_p:
        h = b.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(2, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS)
    ax.set_xlabel("Generation metric")
    ax.set_ylabel("Score ↑")
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8,
              handlelength=1.2, labelspacing=0.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"[plot] wrote {out_pdf}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out/plots",
                    help="Directory for the two output PDFs (relative or absolute).")
    args = ap.parse_args()

    setup_inter()

    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_single("500k", DATA["500k"], out_dir / "stage1_effect_500k.pdf")
    plot_single("951k", DATA["951k"], out_dir / "stage1_effect_951k.pdf")


if __name__ == "__main__":
    main()
