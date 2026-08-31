"""
Offline parser: merge all shard raw files into per-subtype structured JSONL.

Input  : data/raw/finetune_qa_raw/{subtype}.s{start}-{end}.jsonl
Output : data/raw/finetune_qa/{subtype}.jsonl

Output per line:
    factual mode       -> {"text_idx": i, "factual":       [{"q","a"}, ...]}
    comprehension mode -> {"text_idx": i, "gist": "...", "comprehension": [{"q","a"}, ...]}

Records that fail to parse are logged to stderr and skipped.

Run:
    python -m scripts.parse_qa                    # all subtypes
    python -m scripts.parse_qa --subtype ag_news  # one subtype
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune_qa_raw"
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune_qa"

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _clean_pairs(arr) -> list:
    if not isinstance(arr, list):
        return []
    return [p for p in arr if isinstance(p, dict) and p.get("q") and p.get("a")]


def extract(raw: str, mode: str):
    m = _JSON_RE.search(raw)
    if m is None:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    if mode == "factual":
        pairs = _clean_pairs(obj.get("factual", []))
        if not pairs:
            return None
        return {"factual": pairs}

    gist = obj.get("gist")
    if not (isinstance(gist, str) and gist.strip()):
        return None
    return {"gist": gist.strip(), "comprehension": _clean_pairs(obj.get("comprehension", []))}


def parse_subtype(subtype: str) -> tuple[int, int]:
    shards = sorted(RAW_DIR.glob(f"{subtype}.s*.jsonl"))
    if not shards:
        print(f"[{subtype}] no shards yet", file=sys.stderr)
        return 0, 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{subtype}.jsonl"

    seen = set()
    rows  = []
    for shard in shards:
        with shard.open() as f:
            for line in f:
                rec = json.loads(line)
                idx = rec["text_idx"]
                if idx in seen:
                    continue
                seen.add(idx)
                parsed = extract(rec["raw"], rec["mode"])
                if parsed is None:
                    continue
                rows.append({"text_idx": idx, **parsed})

    rows.sort(key=lambda r: r["text_idx"])
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_raw = len(seen)
    kept      = len(rows)
    print(f"[{subtype}] shards={len(shards)} raw={total_raw:,} parsed={kept:,} "
          f"drop={total_raw-kept:,} ({(total_raw-kept)/max(total_raw,1)*100:.1f}%)")
    return total_raw, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subtype", default=None)
    args = ap.parse_args()

    if args.subtype:
        parse_subtype(args.subtype)
        return

    subtypes = sorted({p.name.split(".s")[0] for p in RAW_DIR.glob("*.s*.jsonl")})
    tot_raw, tot_kept = 0, 0
    for st in subtypes:
        r, k = parse_subtype(st)
        tot_raw += r
        tot_kept += k
    print(f"[all] raw={tot_raw:,} parsed={tot_kept:,} drop={tot_raw-tot_kept:,}")


if __name__ == "__main__":
    main()
