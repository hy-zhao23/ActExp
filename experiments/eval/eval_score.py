"""
Score the JSONL outputs of eval_gen_*.py.

Metrics per (dataset, idx, variant) row:
    token_f1     SQuAD-style bag-of-tokens F1 (token-level)
    rougeL       Rouge-L F-measure              (token-level)
    chrf         chrF++ via sacrebleu           (char n-gram)
    bertscore_f1 BERTScore F1                   (semantic, embedding)
    bleurt       BLEURT-20                      (semantic, learned regression)

Only datasets in DEFAULT_INCLUDE are scored unless --include overrides it.
Variants are auto-detected from each JSONL row (any string field that is not
in the reserved set is treated as a variant name).

Usage:
    python experiments/eval/eval_score.py \\
        --eval-dir out/eval/ours_v1-2_500k_fullqa \\
        [--bertscore-model microsoft/deberta-xlarge-mnli] \\
        [--bleurt-model    lucadiliello/BLEURT-20] \\
        [--no-bleurt]      # skip BLEURT (e.g. for CPU runs)

You can pass --eval-dir multiple times to score several runs in one shot; each
gets its own `scores.jsonl` + `metrics_summary.md` written next to the inputs.

Resume:
    Re-running on the same --eval-dir reads the existing scores.jsonl and skips
    any (dataset, idx, variant) triple that's already scored.

Outputs (per --eval-dir):
    scores.jsonl         one row per (dataset, idx, variant) with all 5 metrics
    metrics_summary.md   markdown: one table per metric, dataset × variant
"""

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import torch
import sacrebleu
from bert_score import BERTScorer
from rouge_score import rouge_scorer

# Patch transformers 5.x compat before importing bleurt_pytorch.
from experiments.eval import _bleurt_compat  # noqa: F401

RESERVED_KEYS = {"dataset", "source", "idx", "slot_kind", "question", "gt", "input_text"}
# Default scoring set — all 15 open-gen sources from the diversified mix.
# Note: eval_score writes one row per source, so this is implicit (we score
# whatever jsonl files we find). Kept here for documentation only.

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WS_RE    = re.compile(r"\s+")


def normalize(text: str) -> str:
    """SQuAD-style: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def token_f1(pred: str, ref: str) -> float:
    """SQuAD bag-of-tokens F1 on normalized strings."""
    p_toks = normalize(pred).split()
    r_toks = normalize(ref).split()
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0
    common = Counter(p_toks) & Counter(r_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    prec = n_same / len(p_toks)
    rec  = n_same / len(r_toks)
    return 2 * prec * rec / (prec + rec)


_STOPWORDS: set[str] | None = None
_KEEP_FROM_STOPWORDS = {
    "not", "no", "nor", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "don", "don't", "hadn", "hadn't",
    "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't",
    "won", "won't", "wouldn", "wouldn't",
    "can", "should", "should've", "will",
    "what", "when", "where", "which", "who", "whom", "why", "how",
}


def _stopwords() -> set[str]:
    global _STOPWORDS
    if _STOPWORDS is None:
        from nltk.corpus import stopwords  # noqa: WPS433
        _STOPWORDS = set(stopwords.words("english")) - _KEEP_FROM_STOPWORDS
    return _STOPWORDS


def strip_stopwords(text: str) -> str:
    sw = _stopwords()
    toks = [t for t in normalize(text).split() if t and t not in sw]
    return " ".join(toks)


def strip_question_echo(pred: str, question: str) -> str:
    p, q = pred.lstrip(), question.strip()
    if p.startswith(q):
        p = p[len(q):].lstrip(" :\n-")
    return p


def discover_variants(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        for k, v in r.items():
            if k in RESERVED_KEYS or not isinstance(v, str):
                continue
            if k not in seen:
                seen.append(k)
    return seen


def _load_existing(scores_path: Path) -> dict[tuple[str, int, str, str], dict]:
    """Read scores.jsonl (if any) → keyed by (dataset, idx, slot_kind, variant).

    slot_kind is part of the key because a single text idx typically has
    multiple slots (gist/comp/fact); without it, re-running the scorer would
    collapse N rows-per-idx down to 1 (last-write-wins on the dict).
    Old rows missing slot_kind are treated as slot_kind="" (back-compat).
    """
    if not scores_path.exists():
        return {}
    out: dict[tuple[str, int, str, str], dict] = {}
    for line in scores_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[(r["dataset"], r["idx"], r.get("slot_kind", ""), r["variant"])] = r
    return out


class BleurtRunner:
    """Thin batched wrapper around BleurtForSequenceClassification."""

    def __init__(self, model_id: str, device: torch.device, batch_size: int = 32):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
        print(f"[bleurt] loading {model_id} on {device}…", flush=True)
        self.tok   = BleurtTokenizer.from_pretrained(model_id)
        self.model = BleurtForSequenceClassification.from_pretrained(model_id).eval().to(device)
        self.device = device
        self.batch_size = batch_size

    @torch.inference_mode()
    def score(self, references: list[str], candidates: list[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(candidates), self.batch_size):
            batch_refs  = references[i : i + self.batch_size]
            batch_cands = candidates[i : i + self.batch_size]
            inputs = self.tok(
                batch_refs, batch_cands,
                padding=True, truncation=True, max_length=512, return_tensors="pt",
            ).to(self.device)
            out = self.model(**inputs)
            scores.extend(out.logits.flatten().tolist())
        return scores


def score_dir(eval_dir: Path, bertscorer: BERTScorer, rouge: rouge_scorer.RougeScorer,
              bleurt: BleurtRunner | None, bs_batch: int,
              target: str = "gt", nostop: bool = False) -> None:
    jsonl_files = sorted(p for p in eval_dir.glob("*.jsonl")
                         if not p.name.startswith("scores"))
    assert jsonl_files, f"no per-source *.jsonl found in {eval_dir}"

    bleurt_tag = "off" if bleurt is None else "on"
    print(f"\n{'='*70}\n[score] {eval_dir}  ({len(jsonl_files)} datasets)  "
          f"target={target}  nostop={nostop}  bleurt={bleurt_tag}\n{'='*70}")

    suffix = "" if target == "gt" else f"_vs_{target}"
    if nostop:
        suffix += "_nostop"
    scores_path  = eval_dir / f"scores{suffix}.jsonl"
    summary_path = eval_dir / f"metrics_summary{suffix}.md"

    # Resume: read existing rows so we can skip work and re-emit them.
    existing = _load_existing(scores_path)
    if existing:
        print(f"[resume] existing scores.jsonl has {len(existing)} rows — will skip those")

    metric_keys = ["token_f1", "rougeL", "chrf", "bertscore_f1"]
    if bleurt is not None:
        metric_keys.append("bleurt")

    # aggregate[dataset][variant][metric] -> list[float]
    aggregate: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Open scores.jsonl for *rewrite* — we'll emit resumed rows first, then new.
    rows_out: list[dict] = list(existing.values())

    for jf in jsonl_files:
        ds = jf.stem
        rows = [json.loads(l) for l in jf.read_text().splitlines() if l.strip()]
        variants = discover_variants(rows)
        print(f"\n[{ds}] n={len(rows)}  variants={variants}")

        # Buffer for (cand, ref) per (row_idx, variant) that needs scoring.
        cand_pool: list[str] = []
        ref_pool:  list[str] = []
        ptrs:      list[tuple[int, str]] = []
        # Buffer of CPU-side metrics, indexed by (row_idx, variant).
        cpu_metrics: dict[tuple[int, str], dict[str, float]] = {}

        for ri, r in enumerate(rows):
            idx = r["idx"]
            slot = r.get("slot_kind", "")
            ref = r[target]
            ref_for_text = strip_stopwords(ref) if nostop else ref
            for v in variants:
                # Already scored? Pull cached values into aggregate, skip.
                cached = existing.get((ds, idx, slot, v))
                if cached is not None and all(k in cached for k in metric_keys):
                    for k in metric_keys:
                        aggregate[ds][v][k].append(cached[k])
                    continue
                pred = strip_question_echo(r[v], r["question"])
                pred_for_text = strip_stopwords(pred) if nostop else pred
                tf1 = token_f1(pred_for_text, ref_for_text)
                rL  = rouge.score(ref_for_text, pred_for_text)["rougeL"].fmeasure
                # chrF++ on raw (case + punctuation matter at character level).
                chrf = sacrebleu.sentence_chrf(pred, [ref], word_order=2).score / 100.0
                cpu_metrics[(ri, v)] = {"token_f1": tf1, "rougeL": rL, "chrf": chrf}
                cand_pool.append(pred if pred.strip() else ".")
                ref_pool.append(ref  if ref.strip()  else ".")
                ptrs.append((ri, v))

        # ── Batched GPU metrics (BERTScore + BLEURT) over the pending pool ─
        bs_f: list[float] = []
        bl_s: list[float] = []
        if cand_pool:
            P, R, F = bertscorer.score(cand_pool, ref_pool, batch_size=bs_batch)
            bs_f = F.tolist()
            if bleurt is not None:
                bl_s = bleurt.score(ref_pool, cand_pool)

        for k, (ri, v) in enumerate(ptrs):
            m = cpu_metrics[(ri, v)]
            m["bertscore_f1"] = float(bs_f[k])
            if bleurt is not None:
                m["bleurt"] = float(bl_s[k])
            for kk in metric_keys:
                aggregate[ds][v][kk].append(m[kk])
            rows_out.append({
                "dataset": ds, "idx": rows[ri]["idx"],
                "slot_kind": rows[ri].get("slot_kind", ""),
                "variant": v, **m,
            })

        # Quick per-variant means for this dataset.
        for v in variants:
            line = f"  {v:20s}"
            for kk in metric_keys:
                vals = aggregate[ds][v][kk]
                if vals:
                    line += f"  {kk}={sum(vals)/len(vals):.3f}"
            print(line)

    # Rewrite scores.jsonl (cached + new) atomically.
    tmp = scores_path.with_suffix(".jsonl.tmp")
    with tmp.open("w") as fout:
        for r in rows_out:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(scores_path)

    # ── markdown summary ─────────────────────────────────────────────────────
    all_variants: list[str] = []
    for ds in aggregate:
        for v in aggregate[ds]:
            if v not in all_variants:
                all_variants.append(v)

    lines = [
        f"# Eval scores — `{eval_dir}`",
        f"\nBERTScore model: `{bertscorer.model_type}` "
        f"(rescale_with_baseline={bertscorer.rescale_with_baseline})",
    ]
    if bleurt is not None:
        lines.append(f"BLEURT model: `{bleurt.model.config.name_or_path if hasattr(bleurt.model.config, 'name_or_path') else 'BLEURT-20'}`")
    lines.append("")
    for metric in metric_keys:
        lines.append(f"## {metric}\n")
        header = "| dataset | " + " | ".join(all_variants) + " |"
        sep    = "|" + "|".join(["---"] * (len(all_variants) + 1)) + "|"
        lines.append(header)
        lines.append(sep)
        means_per_variant: dict[str, list[float]] = defaultdict(list)
        for ds in sorted(aggregate.keys()):
            cells = [ds]
            for v in all_variants:
                vals = aggregate[ds].get(v, {}).get(metric, [])
                if vals:
                    mean = sum(vals) / len(vals)
                    cells.append(f"{mean:.3f}")
                    means_per_variant[v].append(mean)
                else:
                    cells.append("—")
            lines.append("| " + " | ".join(cells) + " |")
        cells = ["**overall**"]
        for v in all_variants:
            ms = means_per_variant[v]
            cells.append(f"**{sum(ms)/len(ms):.3f}**" if ms else "—")
        lines.append("| " + " | ".join(cells) + " |\n")

    summary_path.write_text("\n".join(lines))
    print(f"\n[score] ✓ {scores_path}")
    print(f"[score] ✓ {summary_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", action="append", required=True,
                    help="directory containing latentqa_*.jsonl from eval_gen_*.py "
                         "(repeatable)")
    ap.add_argument("--bertscore-model", default="microsoft/deberta-xlarge-mnli",
                    help="HF model id for BERTScore")
    ap.add_argument("--bertscore-batch", type=int, default=64)
    ap.add_argument("--no-rescale", action="store_true",
                    help="disable BERTScore baseline rescaling")
    ap.add_argument("--bleurt-model", default="lucadiliello/BLEURT-20",
                    help="HF model id for BLEURT (default: BLEURT-20, the best model)")
    ap.add_argument("--bleurt-batch", type=int, default=32)
    ap.add_argument("--no-bleurt", action="store_true",
                    help="skip BLEURT scoring (e.g. for CPU-only runs)")
    ap.add_argument("--target", default="gt", choices=["gt", "input_text"],
                    help="reference field. 'gt'=ground-truth (default); "
                         "'input_text'=source text (sanity check for input-paraphrasing)")
    ap.add_argument("--strip-stopwords", action="store_true",
                    help="filter English stopwords before computing token-F1/RougeL. "
                         "chrF/BERTScore/BLEURT always see raw text.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[init] device={device}  bertscore_model={args.bertscore_model}  "
          f"bleurt={'off' if args.no_bleurt else args.bleurt_model}")

    bertscorer = BERTScorer(
        model_type=args.bertscore_model,
        lang="en",
        rescale_with_baseline=not args.no_rescale,
        device=device,
    )
    if bertscorer._tokenizer.model_max_length > 4096:
        bertscorer._tokenizer.model_max_length = 512
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    bleurt = None
    if not args.no_bleurt:
        bleurt = BleurtRunner(args.bleurt_model, torch.device(device), args.bleurt_batch)

    for d in args.eval_dir:
        score_dir(Path(d), bertscorer, rouge, bleurt, args.bertscore_batch,
                  target=args.target, nostop=args.strip_stopwords)


if __name__ == "__main__":
    main()
