"""
Update metadata.json for finetune_diversified_qwen3_l27/ to include all
sources currently present (rank shards exist for them).

Run after cache_reps_new.sh + cache_reps_extras.sh complete.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REP_DIR = PROJECT_ROOT / "data/representations/finetune_diversified_qwen3_l27"
RAW_DIR = PROJECT_ROOT / "data/raw/finetune_diversified"
SHARD_RE = re.compile(r"^(.+)_rank(\d+)_of_(\d+)\.npy$")


def main():
    # discover sources by present shards
    discovered: dict[str, dict] = defaultdict(lambda: {"world_size": None, "ranks": set()})
    for p in REP_DIR.iterdir():
        m = SHARD_RE.match(p.name)
        if m:
            name = m.group(1)
            r = int(m.group(2))
            ws = int(m.group(3))
            discovered[name]["world_size"] = ws
            discovered[name]["ranks"].add(r)

    # cross-check completeness
    sources: dict[str, int] = {}
    for name, info in discovered.items():
        ws = info["world_size"]
        if info["ranks"] != set(range(ws)):
            print(f"[warn] {name}: incomplete ranks {sorted(info['ranks'])}/{ws}, skipping")
            continue
        # row count = lines in jsonl
        jsonl = RAW_DIR / f"{name}.jsonl"
        if not jsonl.exists():
            print(f"[warn] {name}: no jsonl in raw dir, skipping")
            continue
        with jsonl.open("rb") as f:
            n = sum(1 for _ in f)
        sources[name] = n

    # load existing metadata to preserve model/layer/dtype
    meta_path = REP_DIR / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}

    meta.setdefault("model", "Qwen/Qwen3-4B")
    meta.setdefault("layer", 27)
    meta.setdefault("hidden_dim", 2560)
    meta.setdefault("dtype", "bfloat16 as uint16")
    meta.setdefault("max_len", 64)
    meta.setdefault("from", str(REP_DIR))
    meta["sources"] = dict(sorted(sources.items()))

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[done] wrote {meta_path}")
    print(f"  sources ({len(sources)}):")
    for name, n in sources.items():
        print(f"    {name:<30} {n:>7,}")


if __name__ == "__main__":
    main()
