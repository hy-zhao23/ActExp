"""
Aggregate per-experiment scores.jsonl files into a single metric.md report.

Walks `out/eval/<tag>/scores.jsonl`, computes per-source means and the
overall (unweighted) mean per variant, and writes a grouped markdown table
to the project root.

Groups follow the experiment families used by the evaluation manifests:
    1. Data scale (v1-2 size sweep)
    2. Training schemes (v0-4 ~ v1-4 fullqa)
    3a. Decoder scaling
    3b. Donor swap
    4. Baselines
    5. Frozen self-donor

Tags that don't match any group are listed under "other".

Usage:
    python experiments/eval/aggregate_metrics.py \\
        [--eval-root out/eval] \\
        [--out metric.md]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

METRIC_KEYS = ["token_f1", "rougeL", "chrf", "bertscore_f1", "bleurt"]
METRIC_LABELS = {
    "token_f1":     "Token-F1",
    "rougeL":       "RougeL",
    "chrf":         "chrF++",
    "bertscore_f1": "BERTScore",
    "bleurt":       "BLEURT",
}

# (group label, list of (tag-pattern, display name)). Order = display order.
GROUPS = [
    ("1. Data scale (v1-2)", [
        ("ours_v1-2_200k$",             "200k"),
        ("ours_v1-2_500k$",             "500k"),
        ("ours_v1-2_500k_fullqa$",      "500k_fullqa"),
        ("ours_v1-2_750k$",             "750k"),
        ("ours_v1-2_500k_nostage1$",    "500k_nostage1"),
        ("ours_v1-2_fullqa_nostage1$",  "fullqa_nostage1"),
    ]),
    ("2. Training schemes", [
        ("ours_v0-4_fullqa$", "v0-4"),
        ("ours_v0-5_fullqa$", "v0-5"),
        ("ours_v0-7_fullqa$", "v0-7"),
        ("ours_v0-8_fullqa$", "v0-8"),
        ("ours_v1-1_fullqa$", "v1-1"),
        ("ours_v1-3_fullqa$", "v1-3"),
        ("ours_v1-4_fullqa$", "v1-4"),
    ]),
    ("3a. Decoder scaling", [
        ("ours_qwen3_0_6b$",  "Qwen3-0.6B"),
        ("ours_qwen3_8b$",    "Qwen3-8B"),
        ("ours_qwen3_32b$",   "Qwen3-32B"),
    ]),
    ("3b. Donor swap", [
        ("ours_donor_gemma3_4b$",  "Gemma3-4B"),
        ("ours_donor_gemma3_12b$", "Gemma3-12B"),
        ("ours_donor_yi15_34b$",   "Yi-1.5-34B"),
    ]),
    ("4. Baselines", [
        ("ao_qwen3-4b_nothink$",   "AO Qwen3-4B (no-think)"),
        ("ao_llama-3.1-8b$",       "AO Llama-3.1-8B"),
        ("latentqa_qwen.*$",       "LatentQA Qwen-4B"),
        ("latentqa_llama.*$",      "LatentQA Llama-8B"),
    ]),
    ("5. Frozen self-donor", [
        ("ours_frozen_self_llama8b$", "frozen self→llama8b"),
    ]),
]


def load_scores(scores_path: Path) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Returns dataset → variant → metric → [values]."""
    out: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for line in scores_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for k in METRIC_KEYS:
            if k in r:
                out[r["dataset"]][r["variant"]][k].append(r[k])
    return out


def overall_means(per_ds: dict) -> dict[str, dict[str, float]]:
    """variant → metric → overall mean (unweighted across datasets)."""
    by_var: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ds, vmap in per_ds.items():
        for v, mmap in vmap.items():
            for m, vals in mmap.items():
                if vals:
                    by_var[v][m].append(sum(vals) / len(vals))
    out: dict[str, dict[str, float]] = {}
    for v, mmap in by_var.items():
        out[v] = {m: (sum(xs)/len(xs) if xs else float("nan")) for m, xs in mmap.items()}
    return out


def find_group(tag: str) -> tuple[int, int, str]:
    """Return (group_idx, in_group_idx, display_name) or (last, last, tag)."""
    for gi, (_, members) in enumerate(GROUPS):
        for mi, (pat, name) in enumerate(members):
            if re.fullmatch(pat, tag):
                return gi, mi, name
    return len(GROUPS), 0, tag  # "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="out/eval",
                    help="root with per-experiment subdirs each containing scores.jsonl")
    ap.add_argument("--scores-name", default="scores.jsonl",
                    help="filename inside each subdir (default: scores.jsonl)")
    ap.add_argument("--out", default="metric.md",
                    help="output markdown path")
    args = ap.parse_args()

    eval_root = Path(args.eval_root).resolve()
    out_path  = Path(args.out).resolve()
    assert eval_root.exists(), f"eval root not found: {eval_root}"

    # rows: list of (group_idx, in_group_idx, display_name, tag, overall_means_per_variant)
    rows: list[tuple] = []
    for sub in sorted(eval_root.iterdir()):
        if not sub.is_dir():
            continue
        sp = sub / args.scores_name
        if not sp.exists():
            continue
        per_ds = load_scores(sp)
        if not per_ds:
            continue
        means = overall_means(per_ds)
        gi, mi, name = find_group(sub.name)
        rows.append((gi, mi, name, sub.name, means))

    rows.sort(key=lambda r: (r[0], r[1], r[3]))

    if not rows:
        print(f"[aggregate] no scores.jsonl found under {eval_root}")
        return

    # Detect which metrics actually have data (e.g. BLEURT may be off everywhere).
    metrics_present: list[str] = []
    for _, _, _, _, means in rows:
        for v, mmap in means.items():
            for k in METRIC_KEYS:
                if k in mmap and k not in metrics_present:
                    metrics_present.append(k)

    lines: list[str] = [
        "# Eval metrics",
        "",
        f"Aggregated from `{eval_root.relative_to(out_path.parent) if eval_root.is_relative_to(out_path.parent) else eval_root}`. "
        "Each row = one experiment; metric = unweighted mean over the sources in that experiment's `scores.jsonl`.",
        "",
        "All metrics ↑ is better. Ranges:",
        "- Token-F1 / RougeL / chrF++ / BERTScore: 0–1",
        "- BLEURT-20: roughly 0–1 (regression target, can slightly under/overshoot)",
        "",
    ]

    current_group = -1
    for gi, mi, name, tag, means in rows:
        if gi != current_group:
            current_group = gi
            label = GROUPS[gi][0] if gi < len(GROUPS) else "Other"
            lines.append(f"## {label}\n")
            header = "| run | " + " | ".join(METRIC_LABELS[m] for m in metrics_present) + " |"
            sep    = "|" + "|".join(["---"] * (len(metrics_present) + 1)) + "|"
            lines.append(header)
            lines.append(sep)
        # If a row has multiple variants in its scores.jsonl, emit one row per variant.
        for v, mmap in sorted(means.items()):
            cells = [f"{name} (`{tag}` / `{v}`)" if v != tag else f"{name} (`{tag}`)"]
            for m in metrics_present:
                if m in mmap:
                    cells.append(f"{mmap[m]:.3f}")
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"[aggregate] {len(rows)} runs → {out_path}")


if __name__ == "__main__":
    main()
