"""Rebuttal: paired significance tests over per-slot eval scores.

Four comparisons (A vs B), each on its designated metric:
  1. UAV Full vs AO            (Qwen3-4B-Instruct, self)   bertscore_f1
  2. UAV Full vs LatentQA      (Qwen3-4B-Instruct, self)   bertscore_f1
  3. Self vs Cross, Llama-8B   (l27; 8B dec vs Qwen3-4B)   rougeL
  4. Self vs Cross, Gemma-3-4B (self vs Qwen3-4B dec)      bertscore_f1

Pairing key = (dataset, idx, slot_kind, question); slot_kind/question are
taken from the per-dataset gen files, whose row order matches scores.jsonl
(verified for all dirs involved). Tests: paired bootstrap CI/p (B=10,000),
Wilcoxon signed-rank, win rate.

Usage:  python rebuttal/significance_tests.py
Output: rebuttal/significance_results.md
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

EVAL_ROOT = Path(__file__).resolve().parent.parent / "out" / "eval"
OUT_MD = Path(__file__).resolve().parent / "significance_results.md"
N_BOOT = 10_000
SEED = 0

METRICS = ["rougeL", "bertscore_f1"]

COMPARISONS = [
    # (label, dir_A, dir_B)  — Δ is reported as A − B; both METRICS computed
    ("1. UAV vs AO (Qwen3-4B-Instr, self)",
     "ours_self_qwen3_4b", "ao_qwen3-4b_nothink"),
    ("2. UAV vs LatentQA (Qwen3-4B-Instr, self)",
     "ours_self_qwen3_4b", "latentqa_qwen_4b"),
    ("3. Self vs Cross (Llama-8B l27)",
     "ours_self_llama8b", "ours_v1-2_500k_fullqa"),
    ("4. Self vs Cross (Gemma-3-4B)",
     "ours_self_gemma3_4b", "ours_donor_gemma3_4b"),
]


def load_scores(variant_dir: str) -> dict:
    """Map (dataset, idx, slot_kind, question) -> score row."""
    d = EVAL_ROOT / variant_dir
    scores = [json.loads(l) for l in (d / "scores.jsonl").open()]
    out = {}
    for ds, grp in itertools.groupby(scores, key=lambda r: r["dataset"]):
        grp = list(grp)
        gen = [json.loads(l) for l in (d / f"{ds}.jsonl").open()]
        assert len(gen) == len(grp), f"{variant_dir}/{ds}: gen/score length mismatch"
        for g, s in zip(gen, grp):
            assert g["idx"] == s["idx"], f"{variant_dir}/{ds}: row order mismatch"
            out[(ds, g["idx"], g["slot_kind"], g["question"])] = s
    return out


def paired_bootstrap(diff: np.ndarray, rng: np.random.Generator):
    n = len(diff)
    boots = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)
    ])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return lo, hi, min(p, 1.0)


def fmt_p(p: float) -> str:
    if p == 0:
        return f"<{1 / N_BOOT:.0e}"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def main():
    rng = np.random.default_rng(SEED)
    lines = [
        "# Rebuttal significance tests",
        "",
        f"Paired per-slot comparison, n=1500 slots (15 datasets × 100 samples, "
        f"multi-slot). Bootstrap B={N_BOOT}, seed={SEED}. Δ = A − B. "
        "Win rate over non-tied pairs; tie share reported separately.",
        "",
        "| # | comparison | metric | mean A | mean B | Δ | 95% CI (boot) | boot p | Wilcoxon p | win rate | ties |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, da, db in COMPARISONS:
        A, B = load_scores(da), load_scores(db)
        keys = sorted(set(A) & set(B))
        assert len(keys) == len(A) == len(B), f"{label}: pairing incomplete"
        for metric in METRICS:
            a = np.array([A[k][metric] for k in keys])
            b = np.array([B[k][metric] for k in keys])
            diff = a - b
            lo, hi, p_boot = paired_bootstrap(diff, rng)
            w = wilcoxon(a, b)  # zero-diff pairs dropped (default zero_method)
            wins, losses = (diff > 0).sum(), (diff < 0).sum()
            ties = (diff == 0).sum()
            win_rate = wins / (wins + losses)
            lines.append(
                f"| {label} | {metric} | {a.mean():.4f} | {b.mean():.4f} "
                f"| {diff.mean():+.4f} | [{lo:+.4f}, {hi:+.4f}] | {fmt_p(p_boot)} "
                f"| {fmt_p(w.pvalue)} | {win_rate:.1%} ({wins}W/{losses}L) "
                f"| {ties / len(diff):.1%} |"
            )
            print(lines[-1])
    lines += [
        "",
        "Dirs: " + "; ".join(f"{label.split('.')[0]}: {da} vs {db}"
                             for label, da, db in COMPARISONS),
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"\nwritten to {OUT_MD}")


if __name__ == "__main__":
    main()
