"""
Sample sst2, md_gender, snli, ner into (statement, label) CSVs with train/test splits.

Per dataset:
  train: 3,000 pos + 3,000 neg  (6,000 rows)
  test:    250 pos +   250 neg  (  500 rows)

Statement construction:
  sst2      : sentence text
  md_gender : "{text}\n\nThis text is about {entity}."  (mirrors baseline context)
  snli      : "{premise} {hypothesis}"  (self-contained entailment pair)
  ner       : "{sentence}\nNamed entity: {entity}"  (self-contained NE check)

Labels:
  sst2      : 1=positive, 0=negative
  md_gender : 1=female,   0=male
  snli      : 1=entailment (label==0 in HF), 0=contradiction (label==2)
  ner       : 1=entity IS in sentence, 0=entity is NOT in sentence

Output: data/raw/finetune/{dataset}_true_false_{train,test}.csv
"""

import csv
import os
import random

random.seed(42)

PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
NER_CSV = os.path.join(
    PROJ,
    "experiments/baseline/activation_oracles/datasets/classification_datasets/ner/ner.csv",
)
OUT_DIR = os.path.join(PROJ, "data/raw/finetune")
N_TRAIN = 15000  # per label
N_TEST = 250     # per label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_split(prefix: str, true_pool: list[str], false_pool: list[str], name: str) -> None:
    # Reserve test first, then use remainder for train
    sampled_pos = random.sample(true_pool, min(N_TRAIN + N_TEST, len(true_pool)))
    sampled_neg = random.sample(false_pool, min(N_TRAIN + N_TEST, len(false_pool)))

    test_pos = sampled_pos[:N_TEST]
    test_neg = sampled_neg[:N_TEST]
    train_pos = sampled_pos[N_TEST : N_TEST + N_TRAIN]
    train_neg = sampled_neg[N_TEST : N_TEST + N_TRAIN]

    train_rows = [(s, 1) for s in train_pos] + [(s, 0) for s in train_neg]
    test_rows = [(s, 1) for s in test_pos] + [(s, 0) for s in test_neg]
    random.shuffle(train_rows)
    random.shuffle(test_rows)

    for split, rows in [("train", train_rows), ("test", test_rows)]:
        path = f"{prefix}_{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["statement", "label"])
            writer.writerows(rows)
        n1 = sum(l for _, l in rows)
        n0 = len(rows) - n1
        print(f"[{name}] {split}: {len(rows)} rows  (pos={n1}, neg={n0})  → {path}")


# ---------------------------------------------------------------------------
# sst2
# ---------------------------------------------------------------------------

def process_sst2():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    pos, neg = [], []
    for split in ("train", "validation"):
        for row in ds[split]:
            stmt = row["sentence"].strip()
            if row["label"] == 1:
                pos.append(stmt)
            else:
                neg.append(stmt)
    print(f"sst2: pos={len(pos)}, neg={len(neg)}")
    write_split(os.path.join(OUT_DIR, "sst2_true_false"), pos, neg, "sst2")


# ---------------------------------------------------------------------------
# md_gender  (funpedia + image_chat + wizard combined)
# ---------------------------------------------------------------------------

def process_md_gender():
    import re
    from datasets import load_dataset

    female, male = [], []

    # funpedia: gender 1=female, 2=male
    ds = load_dataset("facebook/md_gender_bias", name="funpedia", trust_remote_code=True)
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            if row["gender"] == 0:
                continue
            stmt = f"{row['text']}\n\nThis text is about {row['title']}."
            if row["gender"] == 1:
                female.append(stmt)
            else:
                male.append(stmt)

    # image_chat: caption text with <start>/<eos> markers stripped; female/male bool fields
    ds = load_dataset("facebook/md_gender_bias", name="image_chat", trust_remote_code=True)
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            caption = re.sub(r"<[^>]+>", "", row["caption"]).strip()
            if not caption:
                continue
            if row["female"] and not row["male"]:
                female.append(caption)
            elif row["male"] and not row["female"]:
                male.append(caption)

    # wizard: gender 0=female, 2=male, 1=neutral (skip)
    ds = load_dataset("facebook/md_gender_bias", name="wizard", trust_remote_code=True)
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            if row["gender"] == 1:
                continue
            if row["gender"] == 0:
                female.append(row["text"])
            else:
                male.append(row["text"])

    print(f"md_gender (combined): female={len(female)}, male={len(male)}")
    write_split(os.path.join(OUT_DIR, "md_gender_true_false"), female, male, "md_gender")


# ---------------------------------------------------------------------------
# snli
# ---------------------------------------------------------------------------

def process_snli():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli")["train"]
    entailment, contradiction = [], []
    for row in ds:
        if row["label"] == 0:       # entailment → pos
            stmt = f"{row['premise']} {row['hypothesis']}"
            entailment.append(stmt)
        elif row["label"] == 2:     # contradiction → neg
            stmt = f"{row['premise']} {row['hypothesis']}"
            contradiction.append(stmt)
    print(f"snli: entailment={len(entailment)}, contradiction={len(contradiction)}")
    write_split(os.path.join(OUT_DIR, "snli_true_false"), entailment, contradiction, "snli")


# ---------------------------------------------------------------------------
# ner
# ---------------------------------------------------------------------------

def parse_ner_csv(path: str) -> list[tuple[list[str], list[str]]]:
    """Parse NER CSV into list of (words, entities) per sentence."""
    sentences = []
    current_words: list[str] = []
    current_entities: list[str] = []
    current_entity_tokens: list[str] = []

    with open(path, encoding="unicode_escape") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent_marker = row["Sentence #"].strip()
            word = row["Word"]
            tag = row["Tag"]

            if sent_marker and current_words:
                if current_entity_tokens:
                    current_entities.append(" ".join(current_entity_tokens))
                    current_entity_tokens = []
                sentences.append((current_words, current_entities))
                current_words = []
                current_entities = []

            current_words.append(word)
            if tag.startswith("B-"):
                if current_entity_tokens:
                    current_entities.append(" ".join(current_entity_tokens))
                current_entity_tokens = [word]
            elif tag.startswith("I-"):
                current_entity_tokens.append(word)
            else:  # O
                if current_entity_tokens:
                    current_entities.append(" ".join(current_entity_tokens))
                    current_entity_tokens = []

    if current_words:
        if current_entity_tokens:
            current_entities.append(" ".join(current_entity_tokens))
        sentences.append((current_words, current_entities))

    return sentences


def process_ner():
    print("Parsing NER CSV (large file, may take a moment)...")
    sentences = parse_ner_csv(NER_CSV)
    print(f"  Total sentences: {len(sentences)}")

    # Build global entity pool for false examples
    all_entities: set[str] = set()
    for _, ents in sentences:
        all_entities.update(ents)
    all_entities_list = list(all_entities)

    pos, neg = [], []
    for words, ents in sentences:
        if not ents:
            continue
        sentence_text = " ".join(words)
        ents_set = set(ents)

        # positive: sentence + one of its true entities
        entity = random.choice(ents)
        pos.append(f"{sentence_text}\nNamed entity: {entity}")

        # negative: sentence + an entity NOT in this sentence
        false_entity = random.choice(all_entities_list)
        attempts = 0
        while false_entity in ents_set and attempts < 10:
            false_entity = random.choice(all_entities_list)
            attempts += 1
        if false_entity not in ents_set:
            neg.append(f"{sentence_text}\nNamed entity: {false_entity}")

    print(f"ner: pos={len(pos)}, neg={len(neg)}")
    write_split(os.path.join(OUT_DIR, "ner_true_false"), pos, neg, "ner")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", os.path.join(PROJ, "tmp/huggingface"))
    os.makedirs(OUT_DIR, exist_ok=True)

    process_sst2()
    process_md_gender()
    process_snli()
    process_ner()
