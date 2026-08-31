"""
Build physical train / val / test QA splits for diversified data.

Per-source pipeline:
  1. Load raw text from data/raw/finetune_diversified/{source}.jsonl
  2. Normalize text (whitespace + lowercase) → keep first text_idx per unique text;
     mark later occurrences as duplicates (excluded from ALL splits)
  3. From unique pool (with non-empty QA bundle):
       test : walk highest→lowest text_idx, accumulate records until slot ≥ TEST_SLOTS
       val  : from remaining (idx < min(test_idx)), seeded shuffle, take until slot ≥ VAL_SLOTS
       train: leftover

Outputs three sibling dirs:
  data/raw/finetune_qa_diversified/        ← train QA (OVERWRITES canonical)
  data/raw/finetune_qa_diversified_val/    ← val QA
  data/raw/finetune_qa_diversified_test/   ← test QA

The pre-split canonical is backed up to data/raw/finetune_qa_diversified_full/
on first run (idempotent: skipped if backup already exists).

Run:
  python -m scripts.build_splits
  python -m scripts.build_splits --test-slots 100 --val-slots 1000
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).resolve().parents[1]
RAW_TEXT   = ROOT / "data/raw/finetune_diversified"
QA_DIR     = ROOT / "data/raw/finetune_qa_diversified"
BACKUP_DIR = ROOT / "data/raw/finetune_qa_diversified_full"
VAL_DIR    = ROOT / "data/raw/finetune_qa_diversified_val"
TEST_DIR   = ROOT / "data/raw/finetune_qa_diversified_test"

VAL_SEED = 1337


def norm_text(s: str) -> str:
    return " ".join(s.lower().split())


def slot_count(bundle: dict) -> int:
    n = 0
    if bundle.get("gist"):
        n += 1
    for k in ("factual", "comprehension"):
        for slot in bundle.get(k) or []:
            if slot and slot.get("q") and slot.get("a"):
                n += 1
    return n


def split_one(name: str, text_path: Path, qa_path: Path,
              test_slots_target: int, val_slots_target: int) -> dict:
    texts = []
    with text_path.open() as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    n_total = len(texts)

    first_seen: dict[str, int] = {}
    dup_idx: set[int] = set()
    for i, t in enumerate(texts):
        key = norm_text(t)
        if key in first_seen:
            dup_idx.add(i)
        else:
            first_seen[key] = i

    bundles: dict[int, dict] = {}
    with qa_path.open() as f:
        for line in f:
            b = json.loads(line)
            bundles[b["text_idx"]] = b
    slots = {i: slot_count(bundles.get(i, {})) for i in range(n_total)}

    unique_with_qa = [i for i in range(n_total) if i not in dup_idx and slots[i] > 0]
    unique_set = set(unique_with_qa)

    # test: walk from end backward
    test_idx: list[int] = []
    acc = 0
    for i in reversed(unique_with_qa):
        test_idx.append(i)
        acc += slots[i]
        if acc >= test_slots_target:
            break
    test_set = set(test_idx)
    test_lo = min(test_idx) if test_idx else n_total

    # val: random pool from unique with idx < test_lo (lower segment), seeded
    val_pool = [i for i in unique_with_qa if i not in test_set and i < test_lo]
    rng = np.random.default_rng(VAL_SEED)
    rng.shuffle(val_pool)
    val_idx: list[int] = []
    acc = 0
    for i in val_pool:
        val_idx.append(i)
        acc += slots[i]
        if acc >= val_slots_target:
            break
    val_set = set(val_idx)

    train_set = unique_set - test_set - val_set

    return {
        "name":       name,
        "n_total":    n_total,
        "n_dup":      len(dup_idx),
        "n_unique":   len(unique_with_qa),
        "n_zero_qa":  n_total - len(unique_with_qa) - len(dup_idx),
        "bundles":    bundles,
        "slots":      slots,
        "train_idx":  sorted(train_set),
        "val_idx":    sorted(val_set),
        "test_idx":   sorted(test_set),
    }


def write_split(out_dir: Path, name: str, bundles: dict, idx_list: list[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    with out_path.open("w") as f:
        for i in idx_list:
            b = bundles[i]
            rec = {"text_idx": i, **{k: v for k, v in b.items() if k != "text_idx"}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-slots", type=int, default=100)
    ap.add_argument("--val-slots",  type=int, default=1000)
    args = ap.parse_args()

    sources = sorted(p.stem for p in QA_DIR.glob("*.jsonl"))
    assert sources, f"no QA files in {QA_DIR}"

    if not BACKUP_DIR.exists():
        print(f"[backup] {QA_DIR}  ->  {BACKUP_DIR}")
        shutil.copytree(QA_DIR, BACKUP_DIR)
    else:
        print(f"[backup] {BACKUP_DIR} already exists, skipping")

    print(f"\nTarget: test ≥ {args.test_slots} slots / source,  val ≥ {args.val_slots} slots / source")
    print(f"{'source':<22} {'n_total':>8} {'n_dup':>6} {'n_uniq':>7} "
          f"{'train (rec/slot)':>18} {'val (rec/slot)':>16} {'test (rec/slot)':>17}")
    print("-" * 100)

    grand = dict.fromkeys(
        ["train_rec","train_slot","val_rec","val_slot","test_rec","test_slot","n_dup"], 0)
    for src in sources:
        text_path = RAW_TEXT / f"{src}.jsonl"
        qa_path   = QA_DIR   / f"{src}.jsonl"
        if not text_path.exists():
            print(f"[skip] {src}: raw text missing")
            continue
        s = split_one(src, text_path, qa_path, args.test_slots, args.val_slots)

        # write
        write_split(QA_DIR,  src, s["bundles"], s["train_idx"])
        write_split(VAL_DIR, src, s["bundles"], s["val_idx"])
        write_split(TEST_DIR, src, s["bundles"], s["test_idx"])

        tr_s = sum(s["slots"][i] for i in s["train_idx"])
        v_s  = sum(s["slots"][i] for i in s["val_idx"])
        te_s = sum(s["slots"][i] for i in s["test_idx"])

        print(f"{src:<22} {s['n_total']:>8,} {s['n_dup']:>6,} {s['n_unique']:>7,} "
              f"{len(s['train_idx']):>7,}/{tr_s:>7,}    "
              f"{len(s['val_idx']):>5,}/{v_s:>5,}    "
              f"{len(s['test_idx']):>5,}/{te_s:>5,}")

        grand["train_rec"]  += len(s["train_idx"]);  grand["train_slot"] += tr_s
        grand["val_rec"]    += len(s["val_idx"]);    grand["val_slot"]   += v_s
        grand["test_rec"]   += len(s["test_idx"]);   grand["test_slot"]  += te_s
        grand["n_dup"]      += s["n_dup"]

    print("-" * 100)
    print(f"{'TOTAL':<22} {'':>8} {grand['n_dup']:>6,} {'':>7} "
          f"{grand['train_rec']:>7,}/{grand['train_slot']:>7,}    "
          f"{grand['val_rec']:>5,}/{grand['val_slot']:>5,}    "
          f"{grand['test_rec']:>5,}/{grand['test_slot']:>5,}")
    print(f"\nWrote:\n  {QA_DIR}/   (train)\n  {VAL_DIR}/\n  {TEST_DIR}/")


if __name__ == "__main__":
    main()
