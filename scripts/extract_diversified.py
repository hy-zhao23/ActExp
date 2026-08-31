"""
Physically extract a self-contained diversified subset (jsonl + QA + reps)
from old/new source folders, based on mix_caps in a yaml config.

After running this, training can read ONLY from the diversified folders;
data/raw/finetune_old/ and its rep cache become deletable.

Per-source layout (output):
    data/raw/finetune_diversified/{name}.jsonl
    data/raw/finetune_qa_diversified/{name}.jsonl
    <out-rep-dir>/{name}_rank0_of_1.npy        (single-shard, dense)
    <out-rep-dir>/metadata.json

Eval-split convention preserved: the last min(eval_per_source, 20%) rows of
the original source are kept verbatim at the tail of the new file. Train
rows are deterministically subsampled (seeded) to fit `mix_caps[name]`.

Subset selection per source
---------------------------
  cap = mix_caps[name]
  eval_lo = n_total - min(eval_per_source, n_total // 5)
  eval_gis  = range(eval_lo, n_total)             # always kept
  train_pool = range(0, eval_lo)
  if cap >= len(train_pool):
      train_gis = list(train_pool)
  else:
      train_gis = sorted(rng.choice(eval_lo, cap, replace=False))
  selected = train_gis + list(eval_gis)
  # → new_gi 0..cap-1 are train, cap..cap+n_eval-1 are eval

Usage
-----
    python -m scripts.extract_diversified \\
        --mix-config experiments/training/configs/mix_diversified.yaml \\
        --rep-dir   data/representations/finetune_qwen3_l27 \\
        --out-tag   diversified

    # rerun for another model's rep cache (jsonl + qa skip if already done)
    python -m scripts.extract_diversified \\
        --mix-config experiments/training/configs/mix_diversified.yaml \\
        --rep-dir   data/representations/finetune_qwen3 \\
        --out-tag   diversified \\
        --reps-only

If a source's reps or QA aren't ready yet (e.g. lmsys_user / dair_emotion
before the vLLM job finishes), the script prints a warning and skips that
piece. Re-run after the missing pieces become available.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))   # oracle root
from utils.reps_io import open_reps_shard  # noqa: E402

DEFAULT_JSONL_DIRS = [
    PROJECT_ROOT / "data/raw/finetune_old",
    PROJECT_ROOT / "data/raw/finetune_new",
]
DEFAULT_QA_DIRS = [
    PROJECT_ROOT / "data/raw/finetune_qa_old",
    PROJECT_ROOT / "data/raw/finetune_qa_new",
]

_SHARD_RE = re.compile(r"^(.+)_rank(\d+)_of_(\d+)\.npy$")


# ── helpers ────────────────────────────────────────────────────────────────

def find_in_dirs(name: str, dirs: list[Path], suffix: str = ".jsonl") -> Path | None:
    for d in dirs:
        p = d / f"{name}{suffix}"
        if p.exists():
            return p
    return None


def count_lines(p: Path) -> int:
    n = 0
    with p.open("rb") as f:
        for _ in f:
            n += 1
    return n


def select_indices(n_total: int, cap: int, eval_per_source: int, seed: int) -> tuple[list[int], list[int]]:
    """Return (train_gis, eval_gis), both sorted in original order."""
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


def extract_jsonl(src: Path, dst: Path, gis_in_order: list[int]) -> None:
    """Copy lines at gis_in_order into dst, preserving order."""
    keep = set(gis_in_order)
    rank = {gi: new_idx for new_idx, gi in enumerate(gis_in_order)}
    buf  = [None] * len(gis_in_order)
    with src.open() as f:
        for i, line in enumerate(f):
            if i in keep:
                buf[rank[i]] = line
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as f:
        for line in buf:
            f.write(line)


def extract_qa(src: Path, dst: Path, gis_in_order: list[int]) -> None:
    """Read QA bundles by old text_idx, rewrite with new text_idx (= position in gis_in_order)."""
    bundles: dict[int, dict] = {}
    with src.open() as f:
        for line in f:
            j = json.loads(line)
            bundles[j["text_idx"]] = j
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with dst.open("w") as f:
        for new_idx, old_gi in enumerate(gis_in_order):
            b = bundles.get(old_gi)
            if b is None:
                continue
            b = dict(b)
            b["text_idx"] = new_idx
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
            written += 1
    return written


def open_rep_shards(rep_dir: Path, name: str, n_total: int, hidden_dim: int):
    """Return (shards, world_size) for source. None if not present."""
    shards = []
    for p in rep_dir.iterdir():
        m = _SHARD_RE.match(p.name)
        if m and m.group(1) == name:
            shards.append((int(m.group(2)), int(m.group(3)), p))
    if not shards:
        return None, None
    shards.sort()
    world_size = shards[0][1]
    assert all(ws == world_size for _, ws, _ in shards), f"world_size mismatch for {name}"
    mmaps = []
    for r, _, p in shards:
        cnt = (n_total - r + world_size - 1) // world_size
        mmaps.append(open_reps_shard(p, cnt, hidden_dim))
    return mmaps, world_size


def extract_reps(rep_dir: Path, out_dir: Path, name: str, n_total: int,
                 hidden_dim: int, gis_in_order: list[int]) -> bool:
    mmaps, world_size = open_rep_shards(rep_dir, name, n_total, hidden_dim)
    if mmaps is None:
        return False
    out_arr = np.empty((len(gis_in_order), hidden_dim), dtype=np.uint16)
    for new_idx, gi in enumerate(gis_in_order):
        r = gi % world_size
        j = gi // world_size
        out_arr[new_idx] = mmaps[r][j]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}_rank0_of_1.npy"
    # 统一存 npy（自描述 dtype/shape）；读取一律走 utils.reps_io.open_reps_shard。
    np.save(out_path, out_arr, allow_pickle=False)
    return True


# ── main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-config", type=Path, required=True)
    ap.add_argument("--rep-dir",    type=Path, required=True,
                    help="source rep cache, e.g. data/representations/finetune_qwen3_l27")
    ap.add_argument("--jsonl-dirs", type=Path, nargs="*", default=DEFAULT_JSONL_DIRS)
    ap.add_argument("--qa-dirs",    type=Path, nargs="*", default=DEFAULT_QA_DIRS)
    ap.add_argument("--out-tag",    type=str, default="diversified",
                    help="output suffix; produces finetune_<tag> / finetune_qa_<tag> / finetune_<tag>_<repdir-stem>")
    ap.add_argument("--reps-only",  action="store_true",
                    help="skip jsonl/qa extraction (assume already done by a prior run)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.mix_config.read_text())
    mix_caps        = cfg["mix_caps"]
    eval_per_source = cfg.get("eval_per_source", 250)
    seed            = cfg.get("seed", 42)

    # output dirs
    out_jsonl_dir = PROJECT_ROOT / "data/raw" / f"finetune_{args.out_tag}"
    out_qa_dir    = PROJECT_ROOT / "data/raw" / f"finetune_qa_{args.out_tag}"
    rep_stem      = args.rep_dir.name.replace("finetune_", "", 1) if args.rep_dir.name.startswith("finetune_") else args.rep_dir.name
    out_rep_dir   = PROJECT_ROOT / "data/representations" / f"finetune_{args.out_tag}_{rep_stem}"

    print(f"[extract] mix_config = {args.mix_config}")
    print(f"[extract] rep-dir    = {args.rep_dir}")
    print(f"[extract] out: jsonl = {out_jsonl_dir}")
    print(f"          out: qa    = {out_qa_dir}")
    print(f"          out: reps  = {out_rep_dir}")
    print()

    # rep dir metadata for hidden_dim
    src_meta_path = args.rep_dir / "metadata.json"
    src_meta = json.loads(src_meta_path.read_text())
    hidden_dim = src_meta["hidden_dim"]

    new_meta = {
        "model":      src_meta["model"],
        "layer":      src_meta["layer"],
        "hidden_dim": hidden_dim,
        "dtype":      src_meta["dtype"],
        "max_len":    src_meta.get("max_len"),
        "world_size": 1,
        "from":       str(args.rep_dir),
        "mix_config": str(args.mix_config),
        "sources":    {},
    }

    summary = []
    for name, cap in mix_caps.items():
        if cap == 0:
            print(f"[{name}] cap=0  →  SKIP")
            continue

        jsonl_src = find_in_dirs(name, args.jsonl_dirs)
        if jsonl_src is None:
            print(f"[{name}] !! jsonl not found in {args.jsonl_dirs} → SKIP", file=sys.stderr)
            continue

        n_total = count_lines(jsonl_src)
        train_gis, eval_gis = select_indices(n_total, cap, eval_per_source, seed)
        all_gis = train_gis + eval_gis
        new_total = len(all_gis)

        # 1) jsonl
        if not args.reps_only:
            extract_jsonl(jsonl_src, out_jsonl_dir / f"{name}.jsonl", all_gis)

        # 2) qa
        n_qa = -1
        if not args.reps_only:
            qa_src = find_in_dirs(name, args.qa_dirs)
            if qa_src is None:
                print(f"  [{name}] WARN: QA not found, skipping QA extraction")
            else:
                n_qa = extract_qa(qa_src, out_qa_dir / f"{name}.jsonl", all_gis)

        # 3) reps
        ok = extract_reps(args.rep_dir, out_rep_dir, name,
                          n_total, hidden_dim, all_gis)
        rep_state = "OK" if ok else "MISSING"
        if ok:
            new_meta["sources"][name] = new_total

        summary.append((name, n_total, len(train_gis), len(eval_gis), new_total, n_qa, rep_state))
        print(f"[{name}] orig={n_total} train_keep={len(train_gis)} eval={len(eval_gis)} "
              f"new_total={new_total} qa={n_qa} reps={rep_state}")

    # write out rep metadata
    out_rep_dir.mkdir(parents=True, exist_ok=True)
    (out_rep_dir / "metadata.json").write_text(json.dumps(new_meta, indent=2))

    print()
    print("─" * 90)
    print(f"{'source':<24} {'orig':>8} {'train':>7} {'eval':>5} {'total':>7} {'qa':>7} reps")
    for r in summary:
        n, o, tr, ev, t, q, rs = r
        print(f"{n:<24} {o:>8} {tr:>7} {ev:>5} {t:>7} {q:>7} {rs}")
    print()
    print(f"[done] wrote rep metadata → {out_rep_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
