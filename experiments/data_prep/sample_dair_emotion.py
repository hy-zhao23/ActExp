"""
Sample dair-ai/emotion (default config) into finetune source format.

Source: HF dataset `dair-ai/emotion`, default config = 16k train + 2k val + 2k test.
We take all 20k as training data; FinetuneDataset re-cuts eval from the tail.

Filters:
  1. dedup by exact text
  2. drop empty / overly short (< 15 char) or overly long (> 600 char) lines

Output:
  data/raw/finetune/dair_emotion.jsonl  with one record per kept tweet:
    {"text": ..., "source": "dair_emotion", "subtype": "emotion",
     "meta": {"emotion": <one of 6 labels>, "split": <train|val|test>}}

Usage:
  python experiments/data_prep/sample_dair_emotion.py
"""

import json
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT_PATH = PROJ / "data/raw/finetune_v2/dair_emotion.jsonl"
MIN_LEN, MAX_LEN = 15, 600

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def main():
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))
    from datasets import load_dataset

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[info] loading dair-ai/emotion (default config) ...")
    ds = load_dataset("dair-ai/emotion", trust_remote_code=True)

    seen: set[int] = set()
    n_kept = n_dup = n_len = 0
    counts = {k: 0 for k in LABEL_NAMES}

    with open(OUT_PATH, "w") as fp:
        for split_name in ("train", "validation", "test"):
            split = ds[split_name]
            for row in split:
                text = row["text"].strip()
                if not (MIN_LEN <= len(text) <= MAX_LEN):
                    n_len += 1
                    continue
                h = hash(text)
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)

                emotion = LABEL_NAMES[row["label"]]
                rec = {
                    "text":    text,
                    "source":  "dair_emotion",
                    "subtype": "emotion",
                    "meta": {
                        "emotion": emotion,
                        "split":   split_name,
                    },
                }
                fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_kept += 1
                counts[emotion] += 1

    print(f"[done] wrote {n_kept} rows → {OUT_PATH}")
    print(f"  filtered: len={n_len} dup={n_dup}")
    print(f"  per-label: {counts}")


if __name__ == "__main__":
    main()
