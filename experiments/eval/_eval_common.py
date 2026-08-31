"""Shared bits for eval gen scripts (ours / baseline_*) — source list, sampling
from physical val/test QA folders.

After scripts/build_splits.py:
    data/raw/{qa_subdir}_val/   ← val QA per source
    data/raw/{qa_subdir}_test/  ← test QA per source

All baselines + ours read the SAME physical files → samples are deterministic
and identical across methods without locked seeds.
"""

import json
from pathlib import Path

import numpy as np

from utils.finetune_dataset import (
    _Source,
    _enumerate_qa_slots,
    _load_qa_bundles,
    _resolve_qa,
)

# All 15 diversified-mix sources. v2 regen converted every source (including
# ag_news / sst2 / tweeteval_* / dair_emotion / lmsys_user) to free-form
# QA (gist + comprehension), so we score them all with the same open-gen
# metrics. md_gender / ner were dropped from v2 entirely.
OPEN_GEN_SOURCES = [
    "latentqa_control",
    "scientific",
    "wikipedia_concept",
    "wikipedia_event",
    "wikipedia_generic",
    "wikipedia_organization",
    "wikipedia_person",
    "wikipedia_place",
    "wikipedia_work",
    "ag_news",
    "sst2",
    "tweeteval_emotion",
    "tweeteval_sentiment",
    "dair_emotion",
    "lmsys_user",
]

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

# Default data variant: diversified mix (current production).
DEFAULT_DATA_VARIANT = "finetune_diversified"

# Kept for backward compat with old caller signatures (ignored — physical files
# are deterministic; no seed needed for the test/val draws).
TEST_SEED = 42
VAL_SEED  = 1337


def _data_dirs(data_variant: str, qa_split: str) -> tuple[Path, Path]:
    """Returns (raw_dir, qa_dir) for the chosen split.
    qa_split ∈ {"val", "test"}.
    """
    suffix = {"val": "_val", "test": "_test"}[qa_split]
    raw_dir = DATA_ROOT / "raw" / data_variant
    qa_dir  = DATA_ROOT / "raw" / f"{data_variant.replace('finetune', 'finetune_qa', 1)}{suffix}"
    assert raw_dir.exists(), f"raw_dir missing: {raw_dir}"
    assert qa_dir.exists(),  f"qa_dir missing: {qa_dir} (run scripts/build_splits.py)"
    return raw_dir, qa_dir


def _build_samples(
    source: str, qa_split: str, n_samples: int | None,
    reps_subdir: str, data_variant: str, seed: int,
) -> list[dict]:
    """Internal: load all (text_idx, slot) pairs from physical split dir,
    optionally cap to n_samples (seeded shuffle for determinism).
    """
    raw_dir, qa_dir = _data_dirs(data_variant, qa_split)
    rep_dir = DATA_ROOT / "representations" / reps_subdir
    hidden_dim = json.loads((rep_dir / "metadata.json").read_text())["hidden_dim"]

    src     = _Source(raw_dir / f"{source}.jsonl", rep_dir, hidden_dim)
    bundles = _load_qa_bundles(qa_dir / f"{source}.jsonl")

    eligible = sorted(gi for gi in bundles if _enumerate_qa_slots(bundles[gi]))

    samples: list[dict] = []
    for gi in eligible:
        rec = src.get(gi)
        for slot in _enumerate_qa_slots(bundles[gi]):
            q, a = _resolve_qa(bundles[gi], slot, gi)
            samples.append({
                "idx":        gi,
                "slot_kind":  slot[0],
                "vector":     rec["vector"],
                "question":   q,
                "gt":         a,
                "input_text": rec["text"],
            })

    if n_samples is not None and n_samples < len(samples):
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(samples), size=n_samples, replace=False)
        samples = [samples[i] for i in sorted(picks.tolist())]

    return samples


def build_source_samples(
    source: str, n_samples: int | None = None, seed: int = TEST_SEED,
    reps_subdir: str = "finetune_diversified_qwen3_l27",
    data_variant: str = DEFAULT_DATA_VARIANT,
    eval_per_source: int | None = None,   # deprecated, ignored
) -> list[dict]:
    """TEST samples — loaded from physical {qa_subdir}_test/.

    With the build_splits.py split, this returns ~100 (text_idx, slot) samples
    per source. `n_samples` caps the total if provided (seeded shuffle).
    """
    return _build_samples(source, "test", n_samples, reps_subdir, data_variant, seed)


def build_val_samples(
    source: str, n_samples: int | None = None, seed: int = VAL_SEED,
    reps_subdir: str = "finetune_diversified_qwen3_l27",
    data_variant: str = DEFAULT_DATA_VARIANT,
    eval_per_source: int | None = None,   # deprecated, ignored
) -> list[dict]:
    """VAL samples — loaded from physical {qa_subdir}_val/.

    Returns ~1000 (text_idx, slot) samples per source. Use ONLY for zero-shot
    baseline hyperparam tuning; NEVER report numbers on val.
    """
    return _build_samples(source, "val", n_samples, reps_subdir, data_variant, seed)


def assert_val_test_disjoint(
    val_samples: list[dict], test_samples: list[dict], source_name: str = "?",
) -> None:
    """Runtime check — val and test should have ZERO text_idx overlap.

    With physical split this is guaranteed by construction (scripts/build_splits.py
    partitions text_idx into disjoint train/val/test sets). Kept as a cheap
    runtime assertion to catch upstream contract violations.
    """
    val_idx  = {s["idx"] for s in val_samples}
    test_idx = {s["idx"] for s in test_samples}
    overlap  = val_idx & test_idx
    assert not overlap, (
        f"[CONTAMINATION] {len(overlap)} overlapping idx in {source_name}: "
        f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
    )


def hidden_dim_for(reps_subdir: str) -> int:
    return json.loads(
        (DATA_ROOT / "representations" / reps_subdir / "metadata.json").read_text()
    )["hidden_dim"]
