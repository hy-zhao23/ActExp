"""
Sample from geometry_of_truth and relations datasets with train/test splits.

geometry_of_truth:
  train: 3,000 true + 3,000 false  (6,000 rows)
  test:    250 true +   250 false  (  500 rows)

relations:
  train: 3,000 true + 3,000 false  (6,000 rows)
  test:    250 true +   250 false  (  500 rows)

Filters: tokenized statement length <= 64 tokens (Qwen/Qwen3-4B).

Output:
  data/raw/finetune/geometry_of_truth_true_false_train.csv
  data/raw/finetune/geometry_of_truth_true_false_test.csv
  data/raw/finetune/relations_true_false_train.csv
  data/raw/finetune/relations_true_false_test.csv
"""

import csv
import json
import os
import random
import sys

random.seed(42)

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(PROJ, "experiments/baseline/activation_oracles/datasets/classification_datasets")
GMT_DIR = os.path.join(DATA_DIR, "gmt")
RELATIONS_DIR = os.path.join(DATA_DIR, "relations")
OUT_DIR = os.path.join(PROJ, "data/raw/finetune")
MAX_TOKENS = 64
N_TRAIN = 3000   # per label
N_TEST = 250     # per label
MODEL_NAME = "Qwen/Qwen3-4B"

GMT_DATASETS = [
    "sp_en_trans",
    "cities",
    "smaller_than",
    "larger_than",
    "companies_true_false",
]


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def load_gmt(tokenizer) -> tuple[list[str], list[str]]:
    true_stmts: list[str] = []
    false_stmts: list[str] = []
    for name in GMT_DATASETS:
        path = os.path.join(GMT_DIR, name + ".csv")
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                stmt = row["statement"].strip()
                if token_len(tokenizer, stmt) > MAX_TOKENS:
                    continue
                if row["label"] == "1":
                    true_stmts.append(stmt)
                else:
                    false_stmts.append(stmt)
    return true_stmts, false_stmts


def load_relations(tokenizer) -> tuple[list[str], list[str]]:
    true_stmts: list[str] = []
    false_stmts: list[str] = []

    for rel_type in sorted(os.listdir(RELATIONS_DIR)):
        type_dir = os.path.join(RELATIONS_DIR, rel_type)
        if not os.path.isdir(type_dir):
            continue
        for fname in sorted(os.listdir(type_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(type_dir, fname)) as f:
                data = json.load(f)

            templates = data["prompt_templates"]
            samples = data["samples"]
            objects = list({s["object"] for s in samples})

            for sample in samples:
                template = templates[0] + " {}."
                subj, obj = sample["subject"], sample["object"]

                true_stmt = template.format(subj, obj)
                if token_len(tokenizer, true_stmt) <= MAX_TOKENS:
                    true_stmts.append(true_stmt)

                other_objs = [o for o in objects if o != obj]
                if other_objs:
                    false_obj = random.choice(other_objs)
                    false_stmt = template.format(subj, false_obj)
                    if token_len(tokenizer, false_stmt) <= MAX_TOKENS:
                        false_stmts.append(false_stmt)

    return true_stmts, false_stmts


def write_csv(path: str, rows: list[tuple[str, int]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label"])
        writer.writerows(rows)


def split_and_save(true_stmts: list[str], false_stmts: list[str], prefix: str, name: str) -> None:
    n_needed = N_TRAIN + N_TEST
    for label_name, pool in [("true", true_stmts), ("false", false_stmts)]:
        if len(pool) < n_needed:
            print(f"WARNING [{name}]: only {len(pool)} {label_name} stmts, need {n_needed}", file=sys.stderr)

    sampled_pos = random.sample(true_stmts, min(n_needed, len(true_stmts)))
    sampled_neg = random.sample(false_stmts, min(n_needed, len(false_stmts)))

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

    write_csv(f"{prefix}_train.csv", train_rows)
    write_csv(f"{prefix}_test.csv", test_rows)

    print(f"[{name}] train: {len(train_rows)} rows  (pos={sum(l for _,l in train_rows)}, neg={sum(1-l for _,l in train_rows)})")
    print(f"[{name}] test:  {len(test_rows)} rows  (pos={sum(l for _,l in test_rows)}, neg={sum(1-l for _,l in test_rows)})")


def main():
    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "tmp/huggingface"))
    tokenizer = load_tokenizer()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading geometry_of_truth...")
    gmt_true, gmt_false = load_gmt(tokenizer)
    print(f"  true={len(gmt_true)}, false={len(gmt_false)}")
    split_and_save(gmt_true, gmt_false, os.path.join(OUT_DIR, "geometry_of_truth_true_false"), "geometry_of_truth")

    print("Loading relations...")
    rel_true, rel_false = load_relations(tokenizer)
    print(f"  true={len(rel_true)}, false={len(rel_false)}")
    split_and_save(rel_true, rel_false, os.path.join(OUT_DIR, "relations_true_false"), "relations")


if __name__ == "__main__":
    main()
