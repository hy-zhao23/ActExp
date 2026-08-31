"""
Stage 2 finetune dataset — JSONL + sharded activation memmaps.

Layout
------
data/raw/{raw_subdir}/{source_subtype}.jsonl
    {"text": str, "source": str, "subtype": str, "meta": dict}

data/representations/{subdir}/{source_subtype}_rank{r}_of_{W}.npy
    shape = (n_rows_for_rank, hidden_dim)  dtype = uint16 (bfloat16-bitcast)
data/representations/{subdir}/metadata.json
    {"model": ..., "layer": ..., "hidden_dim": ..., "sources": {...}}

Sharding rule (matches cache_finetune_reps_v2.py):
    JSONL line i  →  rank = i % W,  local = i // W

Train / val / test split
------------------------
Splits are physical QA-file partitions (see scripts/build_splits.py):
    data/raw/{qa_subdir}/        ← train QA only
    data/raw/{qa_subdir}_val/    ← val QA (held out, used for early-stop)
    data/raw/{qa_subdir}_test/   ← test QA (held out, never read during training)

FinetuneDataset(split="train"|"val"|"test") routes qa_dir to the right
physical folder. Raw text + reps are shared across splits (one set of memmaps);
which text_idx contributes a training/val/test sample is determined solely by
QA bundle presence in the routed qa_dir.

QA bundle attachment
--------------------
Records with no QA bundle in the routed qa_dir are silently dropped — that's
how train/val/test physical separation is realized. Each (activation, slot)
pair is a distinct sample, so every QA item is visited exactly once per epoch.

Fields produced by __getitem__
------------------------------
    vector    : torch.Tensor (hidden_dim,) bfloat16
    text      : str
    source    : str           ("wikipedia", "sst2", ...)
    subtype   : str
    meta      : dict
    question  : str           (sampled Q)
    label     : str           (sampled A)
"""

import json
import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.reps_io import open_reps_shard


def _uint16_to_bf16(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)


def _shard_count(n_total: int, world_size: int, rank: int) -> int:
    """Number of rows assigned to `rank` under i % W == rank sharding."""
    if rank >= n_total:
        return 0
    return (n_total - rank + world_size - 1) // world_size


class _Source:
    """One (jsonl + sharded npy) source pair."""

    _SHARD_RE = re.compile(r"^(.+)_rank(\d+)_of_(\d+)\.npy$")

    def __init__(self, jsonl: Path, rep_dir: Path, hidden_dim: int):
        self.name  = jsonl.stem
        self.jsonl = jsonl

        # scan JSONL: keep byte offsets + cheap parsed fields per record
        self._records: list[dict] = []
        self._offsets: list[int]  = []
        with jsonl.open("rb") as f:
            while True:
                off  = f.tell()
                line = f.readline()
                if not line:
                    break
                self._offsets.append(off)
                j = json.loads(line)
                self._records.append({
                    "source":  j["source"],
                    "subtype": j["subtype"],
                    "meta":    j.get("meta", {}),
                })
        self.n_total = len(self._records)

        # discover shards for this source
        shards = []
        for p in rep_dir.iterdir():
            m = self._SHARD_RE.match(p.name)
            if m and m.group(1) == self.name:
                shards.append((int(m.group(2)), int(m.group(3)), p))
        assert shards, f"no shards for {self.name!r} in {rep_dir}"
        shards.sort(key=lambda t: t[0])
        world_size = shards[0][1]
        assert all(ws == world_size for _, ws, _ in shards), f"world_size mismatch for {self.name}"
        self.world_size = world_size

        self._shards = [
            open_reps_shard(p, _shard_count(self.n_total, world_size, r), hidden_dim)
            for r, _, p in shards
        ]

    def get(self, global_idx: int) -> dict:
        rec = self._records[global_idx]
        r   = global_idx % self.world_size
        j   = global_idx // self.world_size
        vec = _uint16_to_bf16(self._shards[r][j].copy())

        with self.jsonl.open("rb") as f:
            f.seek(self._offsets[global_idx])
            text = json.loads(f.readline())["text"]

        return {
            "vector":    vec,
            "text":      text,
            "source":    rec["source"],
            "subtype":   rec["subtype"],
            "meta":      rec["meta"],
            "_text_idx": global_idx,      # for looking up generated QA bundle
        }


# ── QA bundle loading ────────────────────────────────────────────────────────

def _load_qa_bundles(qa_path: Path) -> dict:
    """Load LLM-generated QA file into {text_idx: bundle}."""
    if not qa_path.exists():
        return {}
    bundles: dict = {}
    with qa_path.open() as f:
        for line in f:
            j = json.loads(line)
            bundles[j["text_idx"]] = {
                k: v for k, v in j.items() if k != "text_idx"
            }
    return bundles


def _enumerate_qa_slots(bundle: dict) -> list:
    """Enumerate every (question, answer) item in a bundle as a separate sample.

    Returns a list of slot descriptors: [(kind, payload), ...]
      ("gist",   None)            — LLM gist answer; Q sampled from GIST_POOL deterministically
      ("fact",   k)               — bundle["factual"][k]
      ("comp",   k)               — bundle["comprehension"][k]

    Empty list if the bundle has nothing usable.
    """
    slots: list = []
    if bundle.get("gist"):
        slots.append(("gist", None))
    for k, p in enumerate(bundle.get("factual", [])):
        if p.get("q") and p.get("a"):
            slots.append(("fact", k))
    for k, p in enumerate(bundle.get("comprehension", [])):
        if p.get("q") and p.get("a"):
            slots.append(("comp", k))
    return slots


def _resolve_qa(bundle: dict, slot: tuple, text_idx: int):
    """Given a slot descriptor, return the concrete (question, answer)."""
    from scripts.qa_schemas import GIST_POOL

    kind, k = slot
    if kind == "gist":
        # deterministic gist-phrasing rotation by text_idx
        q = GIST_POOL[text_idx % len(GIST_POOL)]
        return q, bundle["gist"]
    if kind == "fact":
        p = bundle["factual"][k]
        return p["q"], p["a"]
    if kind == "comp":
        p = bundle["comprehension"][k]
        return p["q"], p["a"]
    return None, None


class FinetuneDataset(Dataset):
    """Flat dataset across all {source_subtype}.jsonl files.

    Split is determined by which physical QA folder we route to (see
    scripts/build_splits.py):
        split="train" → data/raw/{qa_subdir}/
        split="val"   → data/raw/{qa_subdir}_val/
        split="test"  → data/raw/{qa_subdir}_test/
        split="eval"  → ALIAS for "test" (deprecated)

    Raw text + memmapped reps are shared across splits — only the QA bundle
    presence in the routed folder decides which (text_idx, slot) pairs become
    samples in this split. Records without QA in the routed folder are
    silently dropped.

    Args:
        split           : "train" | "val" | "test" (alias "eval" → "test")
        data_dir        : project data root (defaults to ../data relative to this file)
        subdir          : which representations subdir to use
        raw_subdir      : raw text subdir under data/raw/
        qa_subdir       : QA base subdir; val/test get `_val` / `_test` suffix
        sources_filter  : if set, only include sources whose .jsonl stem is in this list
        max_per_source  : per-source slot cap (train-only)
        default_max     : default cap when source not in max_per_source (train-only)
        sample_seed     : seed for train cap sampling
        train_subsample_ratio : if set, train cap = round(n_records * ratio) per source
    """

    def __init__(
        self,
        split:           Literal["train", "val", "test", "eval"] = "train",
        data_dir:        Path | None = None,
        subdir:          str  = "finetune",
        raw_subdir:      str  = "finetune",
        qa_subdir:       str | None = None,
        sources_filter:  list | None = None,
        max_per_source:  dict | None = None,
        default_max:     int  | None = None,
        sample_seed:     int           = 42,
        train_subsample_ratio: float | None = None,
        # deprecated, kept to swallow legacy kwarg from existing yaml/code:
        eval_per_source: int  | None = None,
    ):
        if split == "eval":  # legacy alias: "eval" was the held-out used for early-stop
            split = "val"
        assert split in ("train", "val", "test"), f"unknown split {split!r}"

        data_dir = Path(data_dir) if data_dir else (Path(__file__).resolve().parents[1] / "data")
        raw_dir  = data_dir / "raw"           / raw_subdir
        rep_dir  = data_dir / "representations" / subdir

        # qa_subdir base; physical split routes to suffix folders
        qa_base = qa_subdir if qa_subdir is not None else f"{raw_subdir}_qa"
        suffix  = {"train": "", "val": "_val", "test": "_test"}[split]
        qa_dir  = data_dir / "raw" / f"{qa_base}{suffix}"
        assert qa_dir.exists(), f"split={split} expects QA dir {qa_dir} (run scripts/build_splits.py first)"

        meta = json.loads((rep_dir / "metadata.json").read_text())
        self.hidden_dim: int = meta["hidden_dim"]
        self.source_model    = meta.get("model", "unknown")

        # If sources_filter not given, auto-derive from reps metadata so that
        # raw .jsonl files without matching reps shards (e.g. extras / archives)
        # are silently skipped instead of asserting in _Source.
        if sources_filter is None:
            sources_filter = list(meta.get("sources", {}).keys()) or None

        self._sources: list[_Source] = []
        self._qa_by_subtype: dict[str, dict] = {}
        for jsonl in sorted(raw_dir.glob("*.jsonl")):
            if sources_filter and jsonl.stem not in sources_filter:
                continue
            src = _Source(jsonl, rep_dir, self.hidden_dim)
            self._sources.append(src)
            self._qa_by_subtype[src.name] = _load_qa_bundles(qa_dir / jsonl.name)

        # train-side sampling: cap per source (deterministic seeded subset).
        # val/test never get sub-sampled (we want all held-out items).
        max_per_source = max_per_source or {}
        rng = np.random.default_rng(sample_seed)

        # Enumerate every (activation, qa_slot) as a distinct sample so every
        # QA item is seen exactly once per epoch. Records without usable slots
        # in the routed split's QA dir are dropped outright.
        self._index: list[tuple] = []
        self.per_source_counts: dict[str, int] = {}
        self.per_source_sample_counts: dict[str, int] = {}
        for si, src in enumerate(self._sources):
            qa_map = self._qa_by_subtype.get(src.name, {})
            # all text_idx with a usable QA bundle in this split's folder
            eligible = [gi for gi in qa_map.keys() if _enumerate_qa_slots(qa_map[gi])]
            eligible.sort()

            if split == "train":
                cap = max_per_source.get(src.name, default_max)
                if train_subsample_ratio is not None and src.name not in max_per_source:
                    cap = int(round(len(eligible) * train_subsample_ratio))
                if cap is not None and cap < len(eligible):
                    picks = rng.choice(len(eligible), size=cap, replace=False)
                    gis = sorted(eligible[i] for i in picks)
                else:
                    gis = eligible
            else:
                gis = eligible

            self.per_source_counts[src.name] = len(gis)
            sample_count = 0
            for gi in gis:
                for slot in _enumerate_qa_slots(qa_map[gi]):
                    self._index.append((si, gi, slot))
                    sample_count += 1
            self.per_source_sample_counts[src.name] = sample_count

        self.split = split

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, i: int) -> dict:
        si, gi, slot = self._index[i]
        rec = self._sources[si].get(gi)
        bundle = self._qa_by_subtype[self._sources[si].name][gi]
        rec["question"], rec["label"] = _resolve_qa(bundle, slot, gi)
        return rec

    def summary(self) -> str:
        lines = [
            f"[FinetuneDataset split={self.split}  model={self.source_model}]",
            f"  activations: {sum(self.per_source_counts.values()):,}",
            f"  QA samples : {len(self):,}  (each (activation, QA item) is one sample)",
        ]
        for src in self._sources:
            n_act = self.per_source_counts[src.name]
            n_qa  = self.per_source_sample_counts[src.name]
            avg   = (n_qa / n_act) if n_act else 0.0
            lines.append(
                f"  {src.name:28s}  act={n_act:>7,}  qa={n_qa:>8,}  "
                f"qa/act={avg:.2f}"
            )
        return "\n".join(lines)


if __name__ == "__main__":
    for split in ("train", "eval"):
        ds = FinetuneDataset(split=split)
        print(ds.summary())
        if len(ds):
            s = ds[0]
            print(f"  sample[0]: vec={tuple(s['vector'].shape)} {s['vector'].dtype}"
                  f"  {s['source']}/{s['subtype']}  meta={s['meta']}"
                  f"  text={s['text'][:60]!r}")
        print()
