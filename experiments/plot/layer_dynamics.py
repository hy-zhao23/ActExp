"""
Stage-2 finetune training dynamics for Qwen3-4B self-decoding layer ablation.

For each layer L ∈ {3, 9, 15, 21, 27, 33}, reads
  checkpoints/oracle_ft_qwen3_4b_l{L}_diversified/metrics.jsonl
and plots train_loss (dashed) + val_loss (solid) vs. step, one color per layer.

Writes a single PDF to out/plots/layer/layer_dynamics_ft.pdf.

Usage:
    python experiments/plot/layer_dynamics.py [--ckpt-root checkpoints] [--out-dir out/plots/layer]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D

LAYERS = [3, 9, 15, 21, 27, 33]
LAYER_COLOR = {
    3:  "#A6CEE3",  # soft blue
    9:  "#FDBF6F",  # soft orange
    15: "#B2DF8A",  # soft green
    21: "#FB9A99",  # soft red
    27: "#CAB2D6",  # soft purple
    33: "#D7B5A6",  # soft brown
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
    """Returns curves[layer] = (steps, train_loss, val_loss) as parallel lists."""
    curves = {}
    for L in LAYERS:
        p = ckpt_root / f"oracle_ft_qwen3_4b_l{L}_diversified" / "metrics.jsonl"
        steps, tr, va = [], [], []
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            steps.append(r["step"])
            tr.append(r["train_loss"])
            va.append(r["val_loss"])
        curves[L] = (steps, tr, va)
    return curves


def plot_dynamics(curves, out_path: Path):
    fig, ax = plt.subplots(figsize=(2.85, 2.1))

    for L in LAYERS:
        steps, tr, va = curves[L]
        color = LAYER_COLOR[L]
        ax.plot(steps, tr, color=color, linewidth=0.9, linestyle="--", alpha=0.85, zorder=2)
        ax.plot(steps, va, color=color, linewidth=1.2, linestyle="-",  alpha=0.95, zorder=3,
                label=f"L{L}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")

    layer_handles = [
        Line2D([0], [0], color=LAYER_COLOR[L], linestyle="-", linewidth=1.2, label=f"L{L}")
        for L in LAYERS
    ]
    style_handles = [
        Line2D([0], [0], color="black", linestyle="-",  linewidth=1.2, label="val"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=0.9, label="train"),
    ]
    ax.legend(handles=layer_handles + style_handles,
              loc="upper right", frameon=False, ncol=2, fontsize=7,
              handlelength=1.2, columnspacing=0.8, labelspacing=0.2)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--out-dir",   default="out/plots/layer")
    args = ap.parse_args()

    setup_inter()
    ckpt_root = Path(args.ckpt_root).resolve()
    out_dir   = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = load_curves(ckpt_root)
    out_path = out_dir / "layer_dynamics_ft.pdf"
    plot_dynamics(curves, out_path)
    print(f"[layer_dynamics] wrote {out_path}")


if __name__ == "__main__":
    main()
