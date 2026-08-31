"""Truncate each finetune source (JSONL + sharded activations) to target size.

Usage:
    python -m scripts.truncate_finetune_data \
        --subdir finetune \
        --dry-run              # preview without writing

Targets are hard-coded in TARGETS below. Each source is truncated to the first
N lines (simplest); the corresponding shard rows are kept.

Sharding invariant: JSONL line i → rank = i % W, local = i // W.
For target K lines: rank r keeps ceil((K - r) / W) rows if r < K else 0.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   # oracle root
from utils.reps_io import open_reps_shard  # noqa: E402
DATA_DIR     = PROJECT_ROOT / "data"

# Only wiki subtypes and scientific need truncation (prepare_data didn't cap them).
# All other sources are kept in full.
#
# Wiki 340k split balanced across 7 subtypes; event capped at what's available.
# Scientific 260k.
TARGETS = {
    "wikipedia_person":        50_000,
    "wikipedia_place":         50_000,
    "wikipedia_concept":       50_000,
    "wikipedia_organization":  50_000,
    "wikipedia_work":          50_000,
    "wikipedia_event":         40_000,
    "wikipedia_generic":       50_000,
    "scientific":             260_000,
}


_SHARD_RE = re.compile(r"^(.+)_rank(\d+)_of_(\d+)\.npy$")


def _shard_rows(n_total: int, world_size: int, rank: int) -> int:
    if rank >= n_total:
        return 0
    return (n_total - rank + world_size - 1) // world_size


def truncate_jsonl(jsonl: Path, target_n: int, dry: bool) -> tuple[int, int]:
    """Return (original_count, kept_count)."""
    # count lines first
    orig = sum(1 for _ in jsonl.open())
    if orig <= target_n:
        return orig, orig
    if dry:
        return orig, target_n

    # stream first target_n lines to temp, then swap
    tmp = jsonl.with_suffix(".jsonl.tmp")
    with jsonl.open() as fin, tmp.open("w") as fout:
        for i, line in enumerate(fin):
            if i >= target_n:
                break
            fout.write(line)
    tmp.replace(jsonl)
    return orig, target_n


def truncate_shards(rep_dir: Path, name: str, target_n: int, dry: bool) -> list[tuple]:
    """Truncate all {name}_rank*_of_*.npy shards. Return list of (path, old_rows, new_rows)."""
    changes = []
    shards = []
    for p in rep_dir.iterdir():
        m = _SHARD_RE.match(p.name)
        if m and m.group(1) == name:
            shards.append((int(m.group(2)), int(m.group(3)), p))
    if not shards:
        return changes
    shards.sort(key=lambda t: t[0])
    world_size = shards[0][1]
    # read hidden_dim from metadata
    meta = json.loads((rep_dir / "metadata.json").read_text())
    hidden_dim = meta["hidden_dim"]

    for r, _, p in shards:
        old_rows = p.stat().st_size // (hidden_dim * 2)     # uint16 = 2 bytes
        new_rows = _shard_rows(target_n, world_size, r)
        if new_rows >= old_rows:
            changes.append((p, old_rows, old_rows))
            continue
        changes.append((p, old_rows, new_rows))
        if dry:
            continue
        # read first new_rows, overwrite file
        old = open_reps_shard(p, old_rows, hidden_dim)
        arr = np.array(old[:new_rows], dtype=np.uint16, copy=True)
        del old
        # truncate file: simplest is write new then replace
        tmp = p.with_suffix(".npy.tmp")
        np.save(tmp, arr, allow_pickle=False)
        tmp.replace(p)
    return changes


def update_metadata(rep_dir: Path, new_counts: dict[str, int], dry: bool) -> None:
    meta_path = rep_dir / "metadata.json"
    meta = json.loads(meta_path.read_text())
    sources = meta.get("sources", {})
    for k, v in new_counts.items():
        if k in sources:
            sources[k] = v
    meta["sources"] = sources
    if not dry:
        meta_path.write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir",   default="finetune",
                    help="representations subdir (finetune / finetune_qwen3)")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    raw_dir = DATA_DIR / "raw" / "finetune"
    rep_dir = DATA_DIR / "representations" / args.subdir

    print(f"[truncate] raw={raw_dir}")
    print(f"[truncate] rep={rep_dir}  dry_run={args.dry_run}")
    print()

    new_counts: dict[str, int] = {}
    for name, target_n in TARGETS.items():
        jsonl = raw_dir / f"{name}.jsonl"
        if not jsonl.exists():
            print(f"[skip] {name}: no jsonl")
            continue

        orig, kept = truncate_jsonl(jsonl, target_n, args.dry_run)
        shard_changes = truncate_shards(rep_dir, name, kept, args.dry_run)
        new_counts[name] = kept

        mark = "DRY" if args.dry_run else "DONE"
        n_shards_changed = sum(1 for _, old, new in shard_changes if old != new)
        print(f"[{mark}] {name:32s}  jsonl {orig:>7,} → {kept:>7,}   shards modified: {n_shards_changed}/{len(shard_changes)}")

    print()
    update_metadata(rep_dir, new_counts, args.dry_run)
    print(f"[{'DRY' if args.dry_run else 'DONE'}] metadata.json updated")
    print(f"total train-side items after truncation: {sum(new_counts.values()):,}")


if __name__ == "__main__":
    main()
