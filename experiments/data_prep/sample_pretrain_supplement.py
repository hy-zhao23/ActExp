"""
Sample pretrain supplements to bring Stage 1 corpus up to ~500k.

Reads:
    data/raw/pretrain/wikipedia.jsonl    (1.2M long-form)
    data/raw/pretrain/scientific.jsonl   (800k abstracts)

Writes to finetune_diversified/ with normalized {text, source, subtype, meta}
schema so they look like 18th/19th sources to oracle_dataset / FinetuneDataset:
    data/raw/finetune_diversified/pretrain_wiki.jsonl   (100k)
    data/raw/finetune_diversified/pretrain_sci.jsonl    (100k)

Sampling: deterministic head-N (pretrain dump is already shuffled at ingest),
no dedup against finetune subtypes (~negligible overlap, doesn't hurt Stage 1).

Stage 2 won't pick these up because there's no QA file.
"""

import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
PRETRAIN_DIR = PROJ / "data/raw/pretrain"
OUT_DIR = PROJ / "data/raw/finetune_old"   # park as backup, not part of diversified mix

PER_SOURCE = 100_000


def sample(stem: str, out_name: str, source: str, subtype: str):
    src = PRETRAIN_DIR / f"{stem}.jsonl"
    dst = OUT_DIR / f"{out_name}.jsonl"
    n_kept = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            j = json.loads(line)
            text = j.get("text", "").strip()
            if not text:
                continue
            rec = {
                "text":    text,
                "source":  source,
                "subtype": subtype,
                "meta":    {},
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1
            if n_kept >= PER_SOURCE:
                break
    print(f"[{out_name}] wrote {n_kept} rows → {dst}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sample("wikipedia",  "pretrain_wiki", source="pretrain", subtype="wikipedia")
    sample("scientific", "pretrain_sci",  source="pretrain", subtype="scientific")


if __name__ == "__main__":
    main()
