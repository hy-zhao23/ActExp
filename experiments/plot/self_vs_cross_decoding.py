"""Self-decoding vs Qwen3-4B-decoding — one figure per category.

For each of {Classification, Fact, Gist, Overall} produce one figure:
  2x2 grid of metric subplots (Token F1, ROUGE-L, chrF, BERTScore)
  x-axis = donor model (Llama-3.1-8B, Gemma-3-4B, Gemma-3-12B)
  2 paired bars per donor: Self-decoding | Qwen3-4B cross-decoding
  values annotated on top of bars with xytext offsets so left/right labels
  in a group don't overlap (technique borrowed from stage1_effect_plot.py).
  no std bars; y-axis tight to data range.

Style matches the rest of `experiments/plot/`: Inter font, pdf.fonttype=42,
top/right spines hidden, compact panel sizes.

Usage:
    python experiments/plot/self_vs_cross_decoding.py
        [--eval-root out/eval] [--out-dir out/plots/self_vs_cross_decoding]
"""
import argparse
import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

CATEGORIES = {
    "Classification": ["ag_news", "sst2", "tweeteval_emotion", "tweeteval_sentiment", "dair_emotion"],
    "Fact":           ["wikipedia_concept", "wikipedia_event", "wikipedia_generic",
                       "wikipedia_organization", "wikipedia_person", "wikipedia_place", "wikipedia_work"],
    "Gist":           ["lmsys_user", "scientific", "latentqa_control"],
}
DS2CAT = {ds: c for c, dss in CATEGORIES.items() for ds in dss}
METRICS = [("token_f1", "Token F1"), ("rougeL", "ROUGE-L"),
           ("chrf", "chrF"), ("bertscore_f1", "BERTScore")]
CAT_ORDER = ["Classification", "Fact", "Gist", "Overall"]

SELF_COLOR  = "#b9d6e9"   # light blue
CROSS_COLOR = "#e0caef"   # light lavender
# Darker variants for value labels (light pastel on white is unreadable).
SELF_TEXT   = "#2A6FA3"
CROSS_TEXT  = "#6E3D8A"

DONORS = [
    ("Llama-3.1-8B",  "ours_self_llama8b",      "ours_v1-2_500k_fullqa"),
    ("Gemma-3-4B",    "ours_self_gemma3_4b",    "ours_donor_gemma3_4b"),
    ("Gemma-3-12B",   "ours_self_gemma3_12b",   "ours_donor_gemma3_12b"),
]

INTER_DIR = Path.home() / ".fonts" / "Inter"


def _setup_style() -> None:
    """Match the project's plot style: Inter font, no top/right spines, size 10."""
    if INTER_DIR.exists():
        for f in INTER_DIR.glob("Inter-*.ttf"):
            fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"]       = "Inter"
    plt.rcParams["pdf.fonttype"]      = 42
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"]         = 10
    plt.rcParams["axes.labelsize"]    = 10
    plt.rcParams["xtick.labelsize"]   = 10
    plt.rcParams["ytick.labelsize"]   = 10


def cat_means(eval_root: Path, tag: str, metric: str) -> dict:
    p = eval_root / tag / "scores.jsonl"
    b = {c: [] for c in CATEGORIES}
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            cat = DS2CAT.get(d["dataset"])
            if not cat: continue
            b[cat].append(d[metric])
    out = {c: sum(v) / len(v) for c, v in b.items()}
    all_v = sum(b.values(), [])
    out["Overall"] = sum(all_v) / len(all_v)
    return out


def plot_category(eval_root: Path, category: str, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(5.6, 3.2))
    bar_w  = 0.22
    offset = 0.14   # > bar_w/2 = 0.11 → leaves ~0.06 gap between paired bars
    x = np.arange(len(DONORS))

    for ax, (mk, mlabel) in zip(axes.flat, METRICS):
        self_vals  = [cat_means(eval_root, st, mk)[category] for _, st, _  in DONORS]
        cross_vals = [cat_means(eval_root, ct, mk)[category] for _, _,  ct in DONORS]
        bars_s = ax.bar(x - offset, self_vals,  bar_w, color=SELF_COLOR,
                        alpha=0.92, label="Self-decoding")
        bars_c = ax.bar(x + offset, cross_vals, bar_w, color=CROSS_COLOR,
                        alpha=0.92, label="Qwen3-4B (cross)")

        # Value labels: nudge left bar's label slightly LEFT and right bar's
        # label slightly RIGHT so they don't overlap when the two bars are
        # close to the same height (stage1_effect_plot.py technique).
        for b, v in zip(bars_s, self_vals):
            ax.annotate(f"{v:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(-2, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                        color=SELF_TEXT, fontweight="bold")
        for b, v in zip(bars_c, cross_vals):
            ax.annotate(f"{v:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        xytext=(2, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                        color=CROSS_TEXT, fontweight="bold")

        all_v = self_vals + cross_vals
        lo, hi = min(all_v), max(all_v)
        top_pad = max(0.012, (hi - lo) * 0.18)
        bot_pad = max(0.004, (hi - lo) * 0.05)
        lower = lo - bot_pad if lo > 0 else lo - bot_pad
        ax.set_ylim(lower, hi + top_pad)
        ax.set_xticks(x); ax.set_xticklabels([d[0] for d in DONORS], fontsize=8)
        ax.set_title(mlabel, fontsize=10, pad=3)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", which="both", length=2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=SELF_COLOR,  alpha=0.92),
               plt.Rectangle((0, 0), 1, 1, color=CROSS_COLOR, alpha=0.92)]
    fig.legend(handles, ["Self-decoding", "Qwen3-4B (cross)"],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.2, labelspacing=0.3,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.08, 1, 1), pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[self_vs_cross_decoding] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="out/eval")
    ap.add_argument("--out-dir",   default="out/plots/self_vs_cross_decoding")
    args = ap.parse_args()
    _setup_style()
    eval_root = Path(args.eval_root).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in CAT_ORDER:
        slug = cat.lower()
        plot_category(eval_root, cat, out_dir / f"self_vs_cross_{slug}.pdf")


if __name__ == "__main__":
    main()
