"""Ablation chart — v1-2 500k_diverse_fullqa (Llama-3.1-8B donor → Qwen3-4B decoder).

Renders ONE PDF per metric (Token F1, ROUGE-L, chrF, BERTScore) — each a
grouped-bar plot:
  x = ablation variants, ordered by how much of the system is enabled
  hue = slot kind (Comp / Fact / Gist)
  Full UAV column shaded for emphasis.

Style: Inter font, pdf.fonttype=42, top/right spines hidden. PDF-only.

Usage:
    python experiments/plot/ablation_v1-2_500k_plot.py
        [--eval-root out/eval]
        [--out-dir   out/plots/ablation_v1-2_500k]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ── variants ────────────────────────────────────────────────────────────────
# (display_label, eval_dir_basename, variant_key_in_scores_jsonl)
# Order is "least system → most system" so Full UAV sits on the right.
VARIANTS_BASE = [
    ("Base",          "ours_v1-2_500k_fullqa_ablation_baseonly",          "base_only"),
    ("No LoRA",       "ours_v1-2_500k_fullqa_ablation_stage1only",        "stage1_only"),
    ("Rand\n+ LoRA",  "_RAND_AVG_",                                       "_RAND_AVG_"),  # special: mean of 3 seeds
    ("FT-LoRA\nonly", "ours_v1-2_500k_fullqa_ablation_nosoft",            "no_soft"),
    ("Full\nUAV",     "ours_v1-2_500k_fullqa",                            "ours_v1-2_500k_fullqa"),
]
RAND_LORA_DIRS = [
    ("ours_v1-2_500k_fullqa_ablation_randadapter_seed0", "rand_adapter_seed0"),
    ("ours_v1-2_500k_fullqa_ablation_randadapter_seed1", "rand_adapter_seed1"),
    ("ours_v1-2_500k_fullqa_ablation_randadapter_seed2", "rand_adapter_seed2"),
]
RAND_BASE_VARIANT = ("Rand.\n+ base", "ours_v1-2_500k_fullqa_ablation_randbase_seed0", "random_base")

METRICS = [("token_f1", "Token F1"), ("rougeL", "ROUGE-L"),
           ("chrf", "chrF"), ("bertscore_f1", "BERTScore")]

SLOTS = [
    ("Comp", "comp", "#d6e4cc"),
    ("Fact", "fact", "#96b4d3"),
    ("Gist", "gist", "#dbc6e0"),
]

INTER_DIR = Path.home() / ".fonts" / "Inter"


def setup_inter():
    if INTER_DIR.exists():
        for w in ("Regular", "Medium", "Bold"):
            f = INTER_DIR / f"Inter-{w}.ttf"
            if f.exists():
                fm.fontManager.addfont(str(f))
    plt.rcParams["font.family"]       = "Inter"
    plt.rcParams["pdf.fonttype"]      = 42
    plt.rcParams["axes.spines.top"]   = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["font.size"]         = 9
    plt.rcParams["axes.labelsize"]    = 9
    plt.rcParams["xtick.labelsize"]   = 8.5
    plt.rcParams["ytick.labelsize"]   = 8.5


def load_slot_means(eval_root: Path, eval_dir: str, variant_key: str) -> dict:
    """Return {(metric, slot): mean} pooled across all samples for this variant."""
    sp = eval_root / eval_dir / "scores.jsonl"
    bucket = defaultdict(list)
    with sp.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("variant") != variant_key:
                continue
            slot = r.get("slot_kind", "?")
            for mk, _ in METRICS:
                if (v := r.get(mk)) is not None:
                    bucket[(mk, slot)].append(v)
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def load_rand_lora_avg(eval_root: Path) -> dict:
    per_seed = [load_slot_means(eval_root, d, k) for d, k in RAND_LORA_DIRS]
    all_keys = set().union(*per_seed)
    return {k: float(np.mean([s[k] for s in per_seed if k in s])) for k in all_keys}


def build_table(eval_root: Path) -> tuple[list, dict]:
    """Returns (variant_labels, data) where data[label][(metric, slot)] = value.

    Always includes the random Q-Former + base model variant ("Rand. + base").
    """
    variants = list(VARIANTS_BASE)
    # Insert "Rand. + base" right after Base (it sits between Base and Stage-1
    # on the "system enabled" axis — adapter is random and there is no LoRA).
    variants.insert(1, RAND_BASE_VARIANT)

    data = {}
    for label, eval_dir, variant_key in variants:
        if variant_key == "_RAND_AVG_":
            data[label] = load_rand_lora_avg(eval_root)
        else:
            data[label] = load_slot_means(eval_root, eval_dir, variant_key)
    return [v[0] for v in variants], data


def plot_one(mk: str, mlabel: str, labels: list, data: dict, out_path: Path):
    n_var  = len(labels)
    n_slot = len(SLOTS)
    bar_w  = 0.18                       # slim bars (was 0.26)
    gap    = 0.02                       # small visible gap between adjacent bars
    step   = bar_w + gap
    x      = np.arange(n_var)
    offsets = np.array([(i - (n_slot - 1) / 2) * step for i in range(n_slot)])

    fig, ax = plt.subplots(figsize=(4.0, 2.7))
    ymin, ymax = float("inf"), float("-inf")

    # Shade Full UAV column for emphasis.
    full_label = "Full\nUAV"
    if full_label in labels:
        idx = labels.index(full_label)
        ax.axvspan(idx - 0.45, idx + 0.45,
                   facecolor="#F4D9B9", alpha=0.28, zorder=0)

    for s_idx, (slot_label, slot_key, color) in enumerate(SLOTS):
        vals = [data[lab].get((mk, slot_key), float("nan")) for lab in labels]
        bars = ax.bar(x + offsets[s_idx], vals, bar_w,
                      color=color, edgecolor="none",
                      label=slot_label, zorder=2)
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ymin = min(ymin, v); ymax = max(ymax, v)
            va = "bottom" if v >= 0 else "top"
            ax.annotate(f"{v:.3f}", xy=(b.get_x() + b.get_width()/2, v),
                        xytext=(0, 1.5 if v >= 0 else -1.5),
                        textcoords="offset points",
                        ha="center", va=va, fontsize=4.5, color="#333333",
                        weight="bold")

    ax.set_title(mlabel, fontsize=11, weight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.tick_params(axis="x", length=0, pad=3)
    pad = (ymax - min(0.0, ymin)) * 0.12
    ax.set_ylim(min(0.0, ymin) - pad, ymax + pad)
    ax.set_ylabel("Score", fontsize=9)
    ax.set_xlabel("Ablation variant", fontsize=9)
    ax.set_axisbelow(True)

    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.88),
              frameon=False, ncol=n_slot,
              fontsize=9, handlelength=1.1, handletextpad=0.5,
              columnspacing=1.1, borderpad=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"saved: {out_path}")
    plt.close(fig)


def plot(eval_root: Path, out_dir: Path):
    setup_inter()
    labels, data = build_table(eval_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mk, mlabel in METRICS:
        plot_one(mk, mlabel, labels, data,
                 out_dir / f"ablation_v1-2_500k_{mk}.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="out/eval", type=Path)
    ap.add_argument("--out-dir",   default="out/plots/ablation_v1-2_500k", type=Path)
    args = ap.parse_args()
    plot(args.eval_root.resolve(), args.out_dir.resolve())


if __name__ == "__main__":
    main()
