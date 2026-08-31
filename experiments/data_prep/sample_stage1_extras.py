"""
Sample non-overlapping Stage 1 supplements from 4 sources where Stage 2
diversified didn't exhaust the available pool.

Targets (~175k total):
  ag_news_extra            50,000  ← from finetune_old/ag_news rows NOT picked by seed=42
  tweeteval_sentiment_extra 25,299 ← all remaining unused rows (≤ 25k)
  lmsys_user_extra         50,000  ← stream LMSYS-Chat-1M, skip text hashes already in
                                     finetune_diversified/lmsys_user.jsonl
  dair_emotion_extra       50,000  ← load dair-ai/emotion 'unsplit' config (416k),
                                     skip texts already in default split (19,912 rows)

Output:
  data/raw/finetune_diversified/{name}_extra.jsonl

Ratio safety: Stage 2 ignores these (no QA file generated), so they're seen
ONLY by Stage 1 (oracle_dataset.ActivationDataset).
"""

import json
import os
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parents[2]
OUT_DIR = PROJ / "data/raw/finetune_diversified"

SEED = 42
EVAL_PER_SOURCE = 250


# ── helpers shared with extract_diversified ────────────────────────────────

def select_indices(n_total: int, cap: int, eval_per_source: int = EVAL_PER_SOURCE,
                   seed: int = SEED) -> tuple[list[int], list[int]]:
    n_eval  = min(eval_per_source, n_total // 5)
    eval_lo = max(0, n_total - n_eval)
    train_pool = eval_lo
    if cap >= train_pool:
        train_gis = list(range(train_pool))
    else:
        rng = np.random.default_rng(seed)
        train_gis = sorted(rng.choice(train_pool, cap, replace=False).tolist())
    eval_gis = list(range(eval_lo, n_total))
    return train_gis, eval_gis


def count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def write_extras(out_path: Path, name: str, texts: list[str],
                 source: str = "extra", subtype: str = "extra"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in texts:
            f.write(json.dumps({
                "text":    t,
                "source":  source,
                "subtype": subtype,
                "meta":    {"split": "extra"},
            }, ensure_ascii=False) + "\n")
    print(f"[{name}] wrote {len(texts):,} rows → {out_path}")


# ── ag_news / tweeteval_sentiment: take seed=42 complement ──────────────────

def extract_complement(name: str, cap_used_in_diversified: int,
                       max_take: int | None = None):
    src_path = PROJ / "data/raw/finetune_old" / f"{name}.jsonl"
    n_total = count_lines(src_path)
    train_used, eval_used = select_indices(n_total, cap_used_in_diversified)
    used = set(train_used) | set(eval_used)

    candidates = [i for i in range(n_total) if i not in used]
    print(f"[{name}] total={n_total:,} used={len(used):,} candidates={len(candidates):,}")

    if max_take is not None:
        rng = np.random.default_rng(SEED + 1)   # different stream than diversified
        candidates = sorted(rng.choice(candidates, min(max_take, len(candidates)),
                                       replace=False).tolist())

    keep = set(candidates)
    texts: list[str] = []
    with src_path.open() as f:
        for i, line in enumerate(f):
            if i in keep:
                texts.append(json.loads(line)["text"])

    write_extras(OUT_DIR / f"{name}_extra.jsonl", name, texts,
                 source=name, subtype="extra")


# ── lmsys_user: stream LMSYS-Chat-1M, skip already-included text hashes ────

def extract_lmsys_extra(target: int = 50_000):
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))
    from datasets import load_dataset

    # build seen-set from the already-sampled 20k
    existing = PROJ / "data/raw/finetune_diversified/lmsys_user.jsonl"
    seen: set[int] = set()
    with existing.open() as f:
        for line in f:
            seen.add(hash(json.loads(line)["text"]))
    print(f"[lmsys_user_extra] {len(seen)} existing texts loaded into skip set")

    out_path = OUT_DIR / "lmsys_user_extra.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = n_scanned = n_lang = n_flag = n_len = n_dup = n_pii = 0
    MIN_LEN, MAX_LEN = 30, 400

    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    with out_path.open("w") as fp:
        for sample in ds:
            n_scanned += 1
            if sample["language"] != "English":
                n_lang += 1; continue
            if sample["openai_moderation"][0]["flagged"]:
                n_flag += 1; continue
            text = sample["conversation"][0]["content"].strip()
            if not (MIN_LEN <= len(text) <= MAX_LEN):
                n_len += 1; continue
            if text.startswith("NAME_"):
                n_pii += 1; continue
            h = hash(text)
            if h in seen:
                n_dup += 1; continue
            seen.add(h)

            fp.write(json.dumps({
                "text":    text,
                "source":  "lmsys",
                "subtype": "user_extra",
                "meta": {
                    "model":     sample["model"],
                    "language":  sample["language"],
                    "conv_id":   sample["conversation_id"],
                    "turn_idx":  0,
                    "redacted":  sample["redacted"],
                },
            }, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept % 5000 == 0:
                print(f"  [lmsys_user_extra] kept={n_kept:,}/{target:,} scanned={n_scanned:,} dup={n_dup}")
            if n_kept >= target:
                break

    print(f"[lmsys_user_extra] wrote {n_kept:,} rows → {out_path}")
    print(f"  scanned={n_scanned:,} lang={n_lang} flag={n_flag} len={n_len} dup={n_dup} pii={n_pii}")


# ── dair_emotion: load unsplit (416k), exclude default-split texts ─────────

def extract_dair_extra(target: int = 50_000):
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))
    from datasets import load_dataset

    LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    MIN_LEN, MAX_LEN = 15, 600

    existing = PROJ / "data/raw/finetune_diversified/dair_emotion.jsonl"
    seen: set[int] = set()
    with existing.open() as f:
        for line in f:
            seen.add(hash(json.loads(line)["text"]))
    print(f"[dair_emotion_extra] {len(seen):,} existing texts loaded into skip set")

    print("[dair_emotion_extra] loading dair-ai/emotion (unsplit) ...")
    ds = load_dataset("dair-ai/emotion", "unsplit", trust_remote_code=True)["train"]
    print(f"[dair_emotion_extra] unsplit total = {len(ds):,}")

    out_path = OUT_DIR / "dair_emotion_extra.jsonl"
    n_kept = n_dup = n_len = 0
    with out_path.open("w") as fp:
        for row in ds:
            text = row["text"].strip()
            if not (MIN_LEN <= len(text) <= MAX_LEN):
                n_len += 1; continue
            h = hash(text)
            if h in seen:
                n_dup += 1; continue
            seen.add(h)
            emotion = LABEL_NAMES[row["label"]]
            fp.write(json.dumps({
                "text":    text,
                "source":  "dair_emotion",
                "subtype": "emotion_extra",
                "meta":    {"emotion": emotion},
            }, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept >= target:
                break
    print(f"[dair_emotion_extra] wrote {n_kept:,} rows → {out_path}")
    print(f"  filtered: dup={n_dup} len={n_len}")


# ── main ───────────────────────────────────────────────────────────────────

SOURCES = {
    "ag_news":             lambda: extract_complement("ag_news",
                                                       cap_used_in_diversified=50000,
                                                       max_take=50_000),
    "tweeteval_sentiment": lambda: extract_complement("tweeteval_sentiment",
                                                       cap_used_in_diversified=20000,
                                                       max_take=None),
    "lmsys_user":          lambda: extract_lmsys_extra(target=50_000),
    "dair_emotion":        lambda: extract_dair_extra(target=50_000),
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES.keys()) + ["all"], default="all",
                    help="which source to extract; 'all' = run all 4 sequentially")
    ap.add_argument("--task-id", type=int, default=None,
                    help="SLURM array task id 0..3 → maps to a source")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.task_id is not None:
        names = list(SOURCES.keys())
        if not (0 <= args.task_id < len(names)):
            raise SystemExit(f"--task-id must be in [0, {len(names)})")
        target = names[args.task_id]
        print(f"[main] task-id={args.task_id} → source={target}")
        SOURCES[target]()
        return

    if args.source == "all":
        for name, fn in SOURCES.items():
            fn()
    else:
        SOURCES[args.source]()


if __name__ == "__main__":
    main()
