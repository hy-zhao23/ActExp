"""
Sample from FEVER (v1.0 train split):
  train: 10,000 SUPPORTS + 10,000 REFUTES  (20,000 rows)
  test:    500 SUPPORTS +   500 REFUTES   ( 1,000 rows)

Filters:
  1. label in {SUPPORTS, REFUTES}
  2. claim not in geometry_of_truth statement set (exact match)
  3. tokenized claim length <= 64 tokens (Qwen/Qwen3-4B)

Output:
  data/raw/finetune/fever_true_false_train.csv
  data/raw/finetune/fever_true_false_test.csv
"""

import csv
import glob
import os
import random
import sys

random.seed(42)

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
GMT_DIR = os.path.join(PROJ, "experiments/baseline/activation_oracles/datasets/classification_datasets/gmt")
OUT_DIR = os.path.join(PROJ, "data/raw/finetune")
MAX_TOKENS = 64
N_TRAIN = 10_000   # per label
N_TEST = 500       # per label
MODEL_NAME = "Qwen/Qwen3-4B"


def load_gmt_statements(gmt_dir: str) -> set[str]:
    statements: set[str] = set()
    for f in glob.glob(f"{gmt_dir}/**/*.csv", recursive=True) + glob.glob(f"{gmt_dir}/*.csv"):
        with open(f) as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                if "statement" in row:
                    statements.add(row["statement"].strip())
    return statements


def write_csv(path: str, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["statement", "label"])
        writer.writerows(rows)


def main() -> None:
    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "tmp/huggingface"))

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading GMT statements for dedup...")
    gmt_statements = load_gmt_statements(GMT_DIR)
    print(f"  GMT statements: {len(gmt_statements)}")

    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading FEVER train split...")
    fever = load_dataset("fever", "v1.0", split="train", trust_remote_code=True)
    print(f"  Total rows: {len(fever)}")

    supported: list[str] = []
    refuted: list[str] = []

    for row in fever:
        label = row["label"]
        if label not in ("SUPPORTS", "REFUTES"):
            continue
        claim = row["claim"].strip()
        if claim in gmt_statements:
            continue
        if len(tokenizer(claim, add_special_tokens=False)["input_ids"]) > MAX_TOKENS:
            continue
        if label == "SUPPORTS":
            supported.append(claim)
        else:
            refuted.append(claim)

    print(f"  After filtering — SUPPORTS: {len(supported)}, REFUTES: {len(refuted)}")

    n_needed = N_TRAIN + N_TEST
    for name, pool in [("SUPPORTS", supported), ("REFUTES", refuted)]:
        if len(pool) < n_needed:
            print(f"WARNING: only {len(pool)} {name} claims, need {n_needed}", file=sys.stderr)

    sampled_pos = random.sample(supported, min(n_needed, len(supported)))
    sampled_neg = random.sample(refuted, min(n_needed, len(refuted)))

    train_rows = (
        [(s, 1) for s in sampled_pos[:N_TRAIN]]
        + [(s, 0) for s in sampled_neg[:N_TRAIN]]
    )
    test_rows = (
        [(s, 1) for s in sampled_pos[N_TRAIN:N_TRAIN + N_TEST]]
        + [(s, 0) for s in sampled_neg[N_TRAIN:N_TRAIN + N_TEST]]
    )
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    write_csv(os.path.join(OUT_DIR, "fever_true_false_train.csv"), train_rows)
    write_csv(os.path.join(OUT_DIR, "fever_true_false_test.csv"), test_rows)

    print(f"\ntrain: {len(train_rows)} rows  (pos={sum(l for _,l in train_rows)}, neg={sum(1-l for _,l in train_rows)})")
    print(f"test:  {len(test_rows)} rows  (pos={sum(l for _,l in test_rows)}, neg={sum(1-l for _,l in test_rows)})")


if __name__ == "__main__":
    main()
