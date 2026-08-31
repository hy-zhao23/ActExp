"""
Build the shard manifest for QA generation.

Scans data/raw/finetune/*.jsonl, splits each file into SHARD_SIZE-line chunks,
writes data/raw/finetune_qa_raw/shards.jsonl (one JSON per line).

Each shard: {"shard_id": int, "subtype": str, "start": int, "end": int}

Run once before submitting the array:
    python -m scripts.build_shards
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune"
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune_qa_raw"
MANIFEST     = OUT_DIR / "shards.jsonl"

SHARD_SIZE = 20_000


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shards = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        n = sum(1 for _ in path.open())
        for start in range(0, n, SHARD_SIZE):
            shards.append({
                "shard_id": len(shards),
                "subtype":  path.stem,
                "start":    start,
                "end":      min(start + SHARD_SIZE, n),
            })

    with MANIFEST.open("w") as f:
        for s in shards:
            f.write(json.dumps(s) + "\n")

    print(f"[build_shards] wrote {MANIFEST}")
    print(f"[build_shards] total shards = {len(shards)}  (SHARD_SIZE={SHARD_SIZE})")

    # quick per-subtype breakdown
    by = {}
    for s in shards:
        by.setdefault(s["subtype"], []).append(s)
    for stem in sorted(by):
        n_last = by[stem][-1]["end"]
        print(f"  {stem:30s} {len(by[stem]):3d} shards  (N={n_last:,})")


if __name__ == "__main__":
    main()
