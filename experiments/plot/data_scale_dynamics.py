"""
Stage-2 finetune training dynamics for v1-2 data-scale ablation.

For each size ∈ {200k, 500k, 750k, fullqa}, reads
  checkpoints/data_scale/oracle_ft_v1-2_<...>/metrics.jsonl
and plots train_loss (dashed) + val_loss (solid) vs. step, one color per size.

Writes a single PDF to out/plots/data_scale/data_scale_dynamics_ft.pdf.

Usage:
    python experiments/plot/data_scale_dynamics.py [--ckpt-root checkpoints] [--out-dir out/plots/data_scale]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

# (label, ckpt_subdir)
SIZES = [
    ("200k",   "oracle_ft_v1-2_200k_diverse"),
    ("500k",   "oracle_ft_v1-2_500k_diverse"),
    ("750k",   "oracle_ft_v1-2_750k_diverse"),
    ("fullqa", "oracle_ft_v1-2_500k_diverse_fullqa"),
]
SIZE_COLOR = {
    "200k":   "#A6CEE3",
    "500k":   "#FDBF6F",
    "750k":   "#B2DF8A",
    "fullqa": "#FB9A99",
}

INTER_DIR = Path.home() / ".fonts" / "Inter"


def setup_inter():
    for w in ("Regular", "Medium", "Bold"):
        f = INTER_DIR / f"Inter-{w}.ttf"
        if f.exists():
            fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"]      = "Inter"
    plt.rcParams["pdf.fonttype"]     = 42
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"]        = 10
    plt.rcParams["axes.labelsize"]   = 10
    plt.rcParams["xtick.labelsize"]  = 10
    plt.rcParams["ytick.labelsize"]  = 10


def load_curves(ckpt_root: Path):
    curves = {}
    for label, sub in SIZES:
        p = ckpt_root / "data_scale" / sub / "metrics.jsonl"
        steps, tr, va = [], [], []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            steps.append(r["step"])
            tr.append(r["train_loss"])
            va.append(r["val_loss"])
        curves[label] = (steps, tr, va)
    return curves


def plot_dynamics(curves, out_path: Path):
    fig, ax = plt.subplots(figsize=(2.85, 2.1))

    for label, _ in SIZES:
        steps, tr, va = curves[label]
        color = SIZE_COLOR[label]
        ax.plot(steps, tr, color=color, linewidth=0.9, linestyle="--", alpha=0.85, zorder=2)
        ax.plot(steps, va, color=color, linewidth=1.2, linestyle="-",  alpha=0.95, zorder=3,
                label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    size_handles = [
        Line2D([0], [0], color=SIZE_COLOR[label], linestyle="-", linewidth=1.2, label=label)
        for label, _ in SIZES
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.2, label="val"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="train"),
    ]
    ax.legend(handles=size_handles + style_handles,
              loc="upper right", frameon=False, ncol=2, fontsize=7,
              handlelength=1.2, columnspacing=0.8, labelspacing=0.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--out-dir",   default="out/plots/data_scale")
    args = ap.parse_args()

    setup_inter()
    ckpt_root = Path(args.ckpt_root).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(ckpt_root)
    out_path = out_dir / "data_scale_dynamics_ft.pdf"
    plot_dynamics(curves, out_path)
    print(f"[data_scale_dynamics] wrote {out_path}")


if __name__ == "__main__":
    main()
