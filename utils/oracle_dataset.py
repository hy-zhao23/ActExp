"""
Oracle dataset loaders.

ActivationDataset
    vector    : last-token residual stream at a cached layer (sharded .npy files)
    label     : the original input text
    label_ids : tokenizer.encode(label)  (lazy, per __getitem__)

Supports a seeded, reproducible train/test split (default 9:1).

Tokenizer
---------
Pass tokenizer= to the constructor.
Uses tok.encode(text, add_special_tokens=False) so the caller controls
chat-template / prefix wrapping in their collator.
"""

import json
import re
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizerBase

from utils.reps_io import open_reps_shard

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# ── helpers ───────────────────────────────────────────────────────────────────

def _uint16_to_bf16(arr: np.ndarray) -> torch.Tensor:
    """uint16 numpy array → bfloat16 tensor (same 16-bit layout, zero-copy view)."""
    return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)


def _scan_offsets(path: Path) -> list:
    """Return byte offset of every line in a file (fast sequential scan)."""
    offsets = []
    with path.open("rb") as f:
        while True:
            offsets.append(f.tell())
            if not f.readline():
                offsets.pop()
                break
    return offsets


def _split_indices(
    n: int,
    split: Literal["train", "test"],
    seed: int,
    test_ratio: float,
    n_samples: Optional[int],
) -> np.ndarray:
    """
    Return an array of indices for the requested split.

    1. If n_samples is given, subsample n_samples from n (seeded).
    2. Shuffle with seed.
    3. First (1 - test_ratio) fraction → train; remainder → test.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(n)
    if n_samples is not None:
        pool = rng.choice(pool, size=n_samples, replace=False)
    rng.shuffle(pool)
    split_point = int(len(pool) * (1.0 - test_ratio))
    if split == "train":
        return pool[:split_point]
    return pool[split_point:]


# ── ActivationDataset ─────────────────────────────────────────────────────────

def _find_act_shards(rep_dir: Path, source: str) -> list:
    """Return shard paths sorted by rank."""
    pattern = re.compile(rf"^{re.escape(source)}_rank(\d+)_of_(\d+)\.npy$")
    matched = [(p, pattern.match(p.name)) for p in rep_dir.iterdir()]
    shards = sorted(
        [(p, m) for p, m in matched if m is not None],
        key=lambda pm: int(pm[1].group(1)),
    )
    assert shards, f"no shards found for source={source!r} in {rep_dir}"
    return [p for p, _ in shards]


class ActivationDataset(Dataset):
    """
    Residual-stream activations paired with their source texts.

    Shard layout
    ------------
    {source}_rank{r}_of_{W}.npy  stores activations for JSONL lines r::W.
    Global JSONL index i  →  rank = i % W,  local_offset = i // W.

    Texts are loaded lazily via pre-scanned byte offsets (no strings in RAM).
    label_ids are tokenized lazily in __getitem__ when tokenizer is set.

    Args:
        source      : "wikipedia" or "scientific"
        split       : "train" or "test"
        seed        : RNG seed for the train/test split
        test_ratio  : fraction held out for test  (default 0.1)
        n_samples   : total samples to draw before splitting; None = use all
        data_dir    : project data root
        tokenizer   : decoder tokenizer; if set, __getitem__ adds "label_ids"
    """

    def __init__(
        self,
        source: str,
        split: Literal["train", "test"] = "train",
        seed: int = 42,
        test_ratio: float = 0.1,
        n_samples: Optional[int] = None,
        data_dir: Path = DATA_DIR,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        subdir: str = "pretrain",
        reps_subdir: Optional[str] = None,
    ):
        rep_subdir_eff = reps_subdir if reps_subdir is not None else subdir
        rep_dir  = data_dir / "representations" / rep_subdir_eff
        raw_path = data_dir / "raw" / subdir / f"{source}.jsonl"

        # ── metadata: hidden_dim ───────────────────────────────────────────────
        meta = json.loads((rep_dir / "metadata.json").read_text())
        self.hidden_dim: int = meta["hidden_dim"]

        # ── shard discovery ────────────────────────────────────────────────────
        shard_paths = _find_act_shards(rep_dir, source)
        world_size = len(shard_paths)

        # ── lazy text: byte offsets only ───────────────────────────────────────
        self._text_path = raw_path
        self._offsets = _scan_offsets(raw_path)
        n_total = len(self._offsets)

        # ── memmaps (one per shard) ────────────────────────────────────────────
        self._shards = [
            open_reps_shard(p, len(range(r, n_total, world_size)), self.hidden_dim)
            for r, p in enumerate(shard_paths)
        ]
        self._world_size = world_size

        # ── seeded split ───────────────────────────────────────────────────────
        self._indices: np.ndarray = _split_indices(
            n_total, split, seed, test_ratio, n_samples
        )

        assert tokenizer is None or tokenizer.padding_side == "right", (
            f"tokenizer.padding_side must be 'right' for training, got {tokenizer.padding_side!r}"
        )
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> dict:
        global_idx = int(self._indices[i])
        r = global_idx % self._world_size
        j = global_idx // self._world_size

        vec = _uint16_to_bf16(self._shards[r][j].copy())

        with self._text_path.open("rb") as f:
            f.seek(self._offsets[global_idx])
            text = json.loads(f.readline())["text"]

        item = {"vector": vec, "label": text}
        if self.tokenizer is not None:
            item["label_ids"] = self.tokenizer.encode(text, add_special_tokens=False)
        return item


def make_activation_dataset(
    source: Union[str, list[str], None],
    split: Literal["train", "test"] = "train",
    seed: int = 42,
    test_ratio: float = 0.1,
    n_samples: Optional[int] = None,
    data_dir: Path = DATA_DIR,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    subdir: str = "finetune_diversified",
    reps_subdir: Optional[str] = None,
) -> Dataset:
    """Single source → ActivationDataset. List → ConcatDataset. None → auto-discover.

    All splits are seeded per-source (same seed applied independently) so each
    source's train/test boundary is deterministic across runs — essential for
    resume safety.

    `subdir` selects raw text dir (data/raw/<subdir>/{source}.jsonl);
    `reps_subdir` selects rep cache (data/representations/<reps_subdir>/);
    if reps_subdir is None it defaults to subdir.

    When `source is None`, all *.jsonl stems under data/raw/<subdir>/ are used.
    """
    if source is None:
        raw_dir = data_dir / "raw" / subdir
        source = sorted(p.stem for p in raw_dir.glob("*.jsonl"))
        assert source, f"no *.jsonl found in {raw_dir} for auto-discovery"

    if isinstance(source, str):
        return ActivationDataset(
            source, split=split, seed=seed, test_ratio=test_ratio,
            n_samples=n_samples, data_dir=data_dir, tokenizer=tokenizer,
            subdir=subdir, reps_subdir=reps_subdir,
        )

    parts = [
        ActivationDataset(
            s, split=split, seed=seed, test_ratio=test_ratio,
            n_samples=n_samples, data_dir=data_dir, tokenizer=tokenizer,
            subdir=subdir, reps_subdir=reps_subdir,
        )
        for s in source
    ]
    hidden_dim = parts[0].hidden_dim
    assert all(p.hidden_dim == hidden_dim for p in parts), \
        f"hidden_dim mismatch across sources: {[p.hidden_dim for p in parts]}"

    combined = ConcatDataset(parts)
    combined.hidden_dim = hidden_dim    # expose for adapter init
    combined.sources = list(source)
    return combined


# ── smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("=== ActivationDataset ===")
    for split in ("train", "test"):
        ds = ActivationDataset("wikipedia", split=split, seed=42, n_samples=1000, tokenizer=tok)
        print(f"  {split:5s}  len={len(ds):,}")
        s = ds[0]
        print(f"         vector    {s['vector'].shape} {s['vector'].dtype}")
        print(f"         label     {s['label'][:80]!r}")
        print(f"         label_ids {s['label_ids'][:8]} ...")

    tr = ActivationDataset("wikipedia", split="train", seed=42, n_samples=1000)
    te = ActivationDataset("wikipedia", split="test",  seed=42, n_samples=1000)
    assert set(tr._indices.tolist()).isdisjoint(set(te._indices.tolist())), "split overlap!"
    print("  train/test disjoint: OK")

    print()
    print("=== DataLoader (batch_size=4) ===")
    loader = DataLoader(tr, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"  vectors {batch['vector'].shape}")
    print(f"  labels  {batch['label']}")
