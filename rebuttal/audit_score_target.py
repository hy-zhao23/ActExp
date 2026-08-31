"""Audit: were stored eval scores computed against gt or input_text?

Samples 100 slots per slot_kind (seed 0) from ours_self_qwen3_4b, recomputes
rougeL and bertscore_f1 against BOTH gt and input_text with the exact
eval_score.py pipeline (strip_question_echo, stemmer, deberta-xlarge-mnli
rescaled), and checks which direction reproduces scores.jsonl.

Run:  python rebuttal/audit_score_target.py
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "experiments" / "eval"))
from eval_score import strip_question_echo  # noqa: E402

VARIANT = "ours_self_qwen3_4b"
EVAL_DIR = PROJ / "out" / "eval" / VARIANT
N_PER_KIND = 100
SEED = 0


def main():
    # Load gen rows joined with stored scores (line order verified aligned).
    scores = [json.loads(l) for l in (EVAL_DIR / "scores.jsonl").open()]
    joined = []
    for ds, grp in itertools.groupby(scores, key=lambda r: r["dataset"]):
        grp = list(grp)
        gen = [json.loads(l) for l in (EVAL_DIR / f"{ds}.jsonl").open()]
        assert len(gen) == len(grp)
        for g, s in zip(gen, grp):
            assert g["idx"] == s["idx"]
            joined.append({**g, "stored_rougeL": s["rougeL"],
                           "stored_bs": s["bertscore_f1"]})

    rng = np.random.default_rng(SEED)
    sample = []
    for kind in ("gist", "comp", "fact"):
        pool = [r for r in joined if r["slot_kind"] == kind]
        idxs = rng.choice(len(pool), min(N_PER_KIND, len(pool)), replace=False)
        sample += [pool[int(i)] for i in sorted(idxs)]
    print(f"sampled {len(sample)} slots "
          f"({', '.join(k + ':' + str(sum(1 for r in sample if r['slot_kind'] == k)) for k in ('gist', 'comp', 'fact'))})")

    preds = [strip_question_echo(r[VARIANT], r["question"]) for r in sample]
    gts = [r["gt"] for r in sample]
    inps = [r["input_text"] for r in sample]

    # ── ROUGE-L (fast, exact-match test) ──────────────────────────────────
    from rouge_score import rouge_scorer
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rl_gt = np.array([rouge.score(g, p)["rougeL"].fmeasure
                      for g, p in zip(gts, preds)])
    rl_in = np.array([rouge.score(t, p)["rougeL"].fmeasure
                      for t, p in zip(inps, preds)])
    stored_rl = np.array([r["stored_rougeL"] for r in sample])
    tol = 1e-9
    print("\n== ROUGE-L ==")
    print(f"stored == vs_gt   : {(np.abs(stored_rl - rl_gt) < tol).mean():.1%}")
    print(f"stored == vs_input: {(np.abs(stored_rl - rl_in) < tol).mean():.1%}")
    for kind in ("gist", "comp", "fact"):
        m = np.array([r["slot_kind"] == kind for r in sample])
        print(f"  {kind:5s} mean: stored={stored_rl[m].mean():.4f} "
              f"vs_gt={rl_gt[m].mean():.4f} vs_input={rl_in[m].mean():.4f}")

    # ── BERTScore (same config as eval_score.py defaults) ─────────────────
    from bert_score import BERTScorer
    bs = BERTScorer(model_type="microsoft/deberta-xlarge-mnli",
                    lang="en", rescale_with_baseline=True, batch_size=32)
    if bs._tokenizer.model_max_length > 4096:
        bs._tokenizer.model_max_length = 512
    cands = [p if p.strip() else "." for p in preds]
    _, _, f_gt = bs.score(cands, [g if g.strip() else "." for g in gts])
    _, _, f_in = bs.score(cands, [t if t.strip() else "." for t in inps])
    f_gt, f_in = f_gt.numpy(), f_in.numpy()
    stored_bs = np.array([r["stored_bs"] for r in sample])
    print("\n== BERTScore F1 ==")
    print(f"stored ≈ vs_gt   (|Δ|<1e-4): {(np.abs(stored_bs - f_gt) < 1e-4).mean():.1%}"
          f"  corr={np.corrcoef(stored_bs, f_gt)[0, 1]:.4f}")
    print(f"stored ≈ vs_input(|Δ|<1e-4): {(np.abs(stored_bs - f_in) < 1e-4).mean():.1%}"
          f"  corr={np.corrcoef(stored_bs, f_in)[0, 1]:.4f}")
    for kind in ("gist", "comp", "fact"):
        m = np.array([r["slot_kind"] == kind for r in sample])
        print(f"  {kind:5s} mean: stored={stored_bs[m].mean():.4f} "
              f"vs_gt={f_gt[m].mean():.4f} vs_input={f_in[m].mean():.4f}")


if __name__ == "__main__":
    main()
