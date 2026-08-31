"""
Decoder size scaling × task-level (classification / gist / fact) plots.

For each of {token_f1, rougeL, chrf, bertscore_f1}, writes one PDF:
    out/plots/decoder_size/decoder_size_task_{metric}.pdf

x-axis = decoder size (Qwen3 0.6B / 4B / 8B / 14B, all w/ Llama-3.1-8B l27 donor).
Three curves per metric (one per task group), pooled-sample mean per point.

Usage:
    python experiments/plot/decoder_size_task_plot.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# (label, eval-dir)
SIZES = [
    ("0.6B", "ours_qwen3_0_6b_v2"),
    ("4B",   "ours_v1-2_500k_fullqa"),
    ("8B",   "ours_qwen3_8b_v2"),
    ("14B",  "ours_qwen3_14b"),
]
METRICS = ["token_f1", "rougeL", "chrf", "bertscore_f1"]
METRIC_LABEL = {
    "token_f1":     "Token-F1",
    "rougeL":       "RougeL",
    "chrf":         "chrF++",
    "bertscore_f1": "BERTScore",
}

CLASSIFICATION_SRC = {"ag_news", "sst2", "dair_emotion",
                      "tweeteval_emotion", "tweeteval_sentiment"}
GIST_SRC           = {"latentqa_control", "lmsys_user", "scientific"}
FACT_SRC           = {f"wikipedia_{s}" for s in
                      ["concept", "event", "generic",
                       "organization", "person", "place", "work"]}

GROUPS = [
    ("Classification", CLASSIFICATION_SRC, "#5B8DB8"),
    ("Gist",           GIST_SRC,           "#6BA368"),
    ("Fact",           FACT_SRC,           "#B65A5A"),
]

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


def _bucket(src: str) -> str | None:
    for name, members, _ in GROUPS:
        if src in members:
            return name
    return None


def load_scores(eval_root: Path):
    """Returns {size_label: {group_name: {metric: [values...]}}}."""
    per = {sz: defaultdict(lambda: defaultdict(list)) for sz, _ in SIZES}
    for sz, sub in SIZES:
        p = eval_root / sub / "scores.jsonl"
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            g = _bucket(r["dataset"])
            if g is None:
                continue
            for m in METRICS:
                if m in r:
                    per[sz][g][m].append(r[m])
    return per


def plot_metric(per, metric: str, out_path: Path):
    sizes = [s[0] for s in SIZES]
    xs    = np.arange(len(SIZES), dtype=float)

    fig, ax = plt.subplots(figsize=(3.0, 2.2))

    for gname, _, color in GROUPS:
        ys = np.array([np.mean(per[sz][gname][metric]) for sz in sizes])
        ax.plot(xs, ys, color=color, linewidth=1.2, marker="o",
                markersize=5, label=gname, zorder=3)

    ax.set_xlabel("Decoder size")
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_xticks(xs)
    ax.set_xticklabels(sizes)
    ax.legend(loc="best", frameon=False, fontsize=8,
              handlelength=1.4, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="out/eval")
    ap.add_argument("--out-dir",   default="out/plots/decoder_size")
    args = ap.parse_args()

    setup_inter()
    eval_root = Path(args.eval_root).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per = load_scores(eval_root)
    for m in METRICS:
        out = out_dir / f"decoder_size_task_{m}.pdf"
        plot_metric(per, m, out)
        print(f"[saved] {out}")


if __name__ == "__main__":
    main()
