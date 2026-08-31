"""
Data-scale ablation plots (v1-2, student=Qwen3-4B, donor=Llama-3.1-8B).

Writes one PDF per metric to out/plots/data_scale/ with Inter font:
    data_scale_curve_token_f1.pdf
    data_scale_curve_rougeL.pdf
    data_scale_curve_chrf.pdf
    data_scale_curve_bertscore_f1.pdf

Usage:
    python experiments/plot/data_scale_plot.py [--eval-root out/eval] [--out-dir out/plots/data_scale]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.lines import Line2D

# (label, tick_label, eval_dir)
SIZES = [
    ("200k",   "200",    "ours_v1-2_200k"),
    ("500k",   "500",    "ours_v1-2_500k"),
    ("750k",   "750",    "ours_v1-2_750k"),
    ("fullqa", "fullqa", "ours_v1-2_500k_fullqa"),
]
METRICS = ["token_f1", "rougeL", "chrf", "bertscore_f1"]
METRIC_LABEL = {
    "token_f1":     "Token-F1",
    "rougeL":       "RougeL",
    "chrf":         "chrF++",
    "bertscore_f1": "BERTScore",
}
SIZE_COLOR = {
    "200k":   "#A6CEE3",
    "500k":   "#FDBF6F",
    "750k":   "#B2DF8A",
    "fullqa": "#FB9A99",
}
METRIC_COLOR = {
    "token_f1":     "#5B8DB8",
    "rougeL":       "#E6A157",
    "chrf":         "#6BA368",
    "bertscore_f1": "#B65A5A",
}

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


def load_scores(eval_root: Path):
    per_sample = {label: defaultdict(list) for label, _, _ in SIZES}
    for label, _, sub in SIZES:
        p = eval_root / sub / "scores.jsonl"
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            for m in METRICS:
                if m in r:
                    per_sample[label][m].append(r[m])
    return per_sample


def plot_metric(per_sample, metric: str, out_path: Path):
    labels = [s[0] for s in SIZES]
    slots  = [s[1] for s in SIZES]
    xs     = np.arange(len(SIZES), dtype=float)
    data   = [np.asarray(per_sample[L][metric]) for L in labels]
    means  = np.array([d.mean() for d in data])
    best_i = int(np.argmax(means))
    best_x = xs[best_i]

    fig, ax = plt.subplots(figsize=(2.85, 2.1))

    rng = np.random.default_rng(0)
    for L, x, vals in zip(labels, xs, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(x + jitter, vals, s=4, color=SIZE_COLOR[L],
                   alpha=0.12, linewidth=0, zorder=2)

    ax.plot(xs, means, color="black", linewidth=0.9, zorder=4)
    non_best = [i for i in range(len(xs)) if i != best_i]
    ax.scatter(xs[non_best], means[non_best],
               marker="D", facecolor="none", edgecolor="black",
               linewidth=0.9, s=20, zorder=5)
    ax.scatter([best_x], [means[best_i]], marker="D",
               facecolors="black", edgecolors="black",
               linewidths=0.9, s=20, alpha=0.8, zorder=6)
    ax.annotate("best", (best_x, means[best_i]),
                xytext=(0, 8), textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
                color="black", weight="bold", zorder=7)

    ax.set_xlabel("QA slots (k)")
    ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_xticks(xs)
    ax.set_xticklabels(slots)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def plot_combined(per_sample, out_path: Path):
    labels = [s[0] for s in SIZES]
    slots  = [s[1] for s in SIZES]
    xs     = np.arange(len(SIZES), dtype=float)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    all_means = []
    for m in METRICS:
        means = np.array([np.mean(per_sample[L][m]) for L in labels])
        all_means.append(means)
        color = METRIC_COLOR[m]
        ax.plot(xs, means, color=color, linewidth=1.2, marker="o",
                markersize=4, label=METRIC_LABEL[m], zorder=3)
        best_i = int(np.argmax(means))
        ax.scatter([xs[best_i]], [means[best_i]], marker="D",
                   facecolors=color, edgecolors="black",
                   linewidths=0.8, s=36, zorder=5)

    all_means = np.concatenate(all_means)
    lo, hi = all_means.min(), all_means.max()
    span = hi - lo
    ax.set_ylim(lo - 0.08 * span, hi + 0.12 * span)

    ax.set_xlabel("QA slots (k)")
    ax.set_ylabel("Score")
    ax.set_xticks(xs)
    ax.set_xticklabels(slots)
    metric_handles = [
        Line2D([0], [0], color=METRIC_COLOR[m], linewidth=1.2,
               marker="o", markersize=4, label=METRIC_LABEL[m])
        for m in METRICS
    ]
    best_handle = Line2D([0], [0], color="#444444", linewidth=0,
                         marker="D", markersize=4,
                         markeredgecolor="black", markeredgewidth=0.8,
                         label="best")
    ax.legend(handles=metric_handles + [best_handle],
              loc="center right", bbox_to_anchor=(0.99, 0.5),
              frameon=False, fontsize=8,
              handlelength=1.2, labelspacing=0.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="out/eval")
    ap.add_argument("--out-dir",   default="out/plots/data_scale")
    args = ap.parse_args()

    setup_inter()
    eval_root = Path(args.eval_root).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample = load_scores(eval_root)
    for m in METRICS:
        out_path = out_dir / f"data_scale_curve_{m}.pdf"
        plot_metric(per_sample, m, out_path)
        print(f"[data_scale_plot] wrote {out_path}")

    combined_path = out_dir / "data_scale_curve_all.pdf"
    plot_combined(per_sample, combined_path)
    print(f"[data_scale_plot] wrote {combined_path}")


if __name__ == "__main__":
    main()
