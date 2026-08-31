"""R2 gen-eval 汇总：三个 ft 变体 × {非wiki, wiki-fact, 全部} 的 R-L / BERTScore。

读 out/eval/rebuttal_r2_qatypes_g2f/scores.jsonl（eval_score.py 产物），
pool 全部 sample 算 mean ± std（不分 macro/micro）。
"""

import json
import statistics
from pathlib import Path

SCORES = Path("out/eval/rebuttal_r2_qatypes_g2f/scores.jsonl")
VARIANTS = ["r2_gistcls", "r2_factonly", "r2_mix500k", "r2_g2f"]
METRICS = ["rougeL", "bertscore_f1"]


def slice_of(dataset: str) -> str:
    return "wiki" if dataset.startswith("wikipedia") else "nonwiki"


def main():
    rows = [json.loads(l) for l in SCORES.read_text().splitlines() if l.strip()]
    buckets: dict[tuple, dict[str, list]] = {}
    for r in rows:
        if r["variant"] not in VARIANTS:
            continue
        for sl in (slice_of(r["dataset"]), "all"):
            b = buckets.setdefault((r["variant"], sl), {m: [] for m in METRICS})
            for m in METRICS:
                b[m].append(r[m])

    print(f"{'variant':14s} {'slice':8s} {'n':>5s}  " +
          "  ".join(f"{m:>22s}" for m in METRICS))
    for v in VARIANTS:
        for sl in ("nonwiki", "wiki", "all"):
            b = buckets.get((v, sl))
            if not b:
                continue
            n = len(b[METRICS[0]])
            cells = [f"{statistics.mean(b[m]):.4f} ± {statistics.stdev(b[m]):.4f}"
                     for m in METRICS]
            print(f"{v:14s} {sl:8s} {n:5d}  " + "  ".join(f"{c:>22s}" for c in cells))


if __name__ == "__main__":
    main()
