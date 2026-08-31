"""
Cache layer-27 residual stream (last token) from Llama-3.1-8B-Instruct.

Architecture: hidden_dim=4096, 32 layers, SwiGLU FFN (intermediate=14336)
GPU budget  : 80G A100  →  model 16 GB + activations peak ~34 GB = 50 GB (63%)
Batch size  : 8192
Seq len     : 64 tokens (truncate if longer, left-pad if shorter)
Dtype       : bfloat16 stored as uint16 memmap (same bit layout, numpy-compatible)

Early stopping: hook on model.model.layers[LAYER] raises _Stop to abort the
forward pass immediately — layers LAYER+1 … 31 never execute.

Storage layout:
  data/representations/pretrain/
    wikipedia_rank{r}_of_{n}.npy    shape=[1_200_000 // n, 4096]  uint16 (bfloat16)
    scientific_rank{r}_of_{n}.npy   shape=[  800_000 // n, 4096]  uint16 (bfloat16)
    metadata.json

Checkpoint: .progress JSON written every CHUNK_SIZE texts; on restart the shard
resumes from the last flushed position.

Usage:
  python cache_representations.py --rank $RANK --world-size $N
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # oracle root
from utils.reps_io import finalize_building, open_building_shard  # noqa: E402

HIDDEN_DIM  = 4096
BATCH_SIZE  = 3500
MAX_LEN     = 64
LAYER       = 27    # 0-indexed; model.model.layers[27] = 28th transformer block
CHUNK_SIZE  = 10_000


# ── early-stop hook ──────────────────────────────────────────────────────────

class _Stop(Exception):
    pass


def _make_hook(buf: dict):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out  # bare tensor or tuple
        buf["h"] = h.detach().cpu()                    # [B, L, D]  offload immediately; frees GPU before _Stop unwinds
        raise _Stop
    return hook


# ── bfloat16 ↔ uint16 helpers ────────────────────────────────────────────────

def to_uint16(t: torch.Tensor) -> np.ndarray:
    """bfloat16 tensor → uint16 numpy (same 16-bit layout, no data loss)."""
    return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def from_uint16(arr: np.ndarray) -> torch.Tensor:
    """uint16 numpy → bfloat16 tensor."""
    return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, tokenizer, texts: list[str], device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Return last-token residual stream at LAYER, shape [N, HIDDEN_DIM], uint16.
    Forward pass aborts after LAYER via _Stop exception (layers LAYER+1… never run).
    """
    buf: dict[str, torch.Tensor] = {}
    handle = model.model.layers[LAYER].register_forward_hook(_make_hook(buf))
    results = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(device)
        try:
            model(**enc)
        except _Stop:
            pass
        # left-padding → last real token always at position -1
        # buf["h"] is already on CPU (offloaded in hook)
        rep = buf["h"][:, -1, :]   # [B, D] bfloat16
        results.append(to_uint16(rep))
        del buf["h"], enc

    handle.remove()
    return np.concatenate(results, axis=0)  # [N, D] uint16


# ── shard writer ─────────────────────────────────────────────────────────────

def load_shard_texts(jsonl_path: str, rank: int, world_size: int) -> list[str]:
    with open(jsonl_path) as f:
        lines = f.readlines()
    return [json.loads(l)["text"] for l in lines[rank::world_size]]


def cache_shard(
    jsonl_path: str,
    out_path: str,
    rank: int,
    world_size: int,
    model,
    tokenizer,
    device,
    batch_size: int = BATCH_SIZE,
) -> None:
    progress_path = out_path + ".progress"

    # done = file exists with no progress file (progress deleted on completion)
    if os.path.exists(out_path) and not os.path.exists(progress_path):
        print(f"[skip] {os.path.basename(out_path)}")
        return

    texts   = load_shard_texts(jsonl_path, rank, world_size)
    n_total = len(texts)

    n_done = 0
    if os.path.exists(progress_path):
        # align down to last complete CHUNK_SIZE boundary (guards against partial flush)
        raw = json.loads(open(progress_path).read())["n_done"]
        n_done = (raw // CHUNK_SIZE) * CHUNK_SIZE
        print(f"[resume] {os.path.basename(out_path)}: {n_done:,}/{n_total:,}")
    else:
        # write progress BEFORE creating the memmap file so a crash during
        # pre-allocation never produces a zero-filled file mistaken for "done"
        with open(progress_path, "w") as f:
            json.dump({"n_done": 0}, f)

    # 分块期间裸写 <out>.building（memmap 可增量写、支持续传），全部算完再定稿为 npy
    mmap = open_building_shard(out_path, n_total, HIDDEN_DIM)

    for start in tqdm(range(n_done, n_total, CHUNK_SIZE), desc=os.path.basename(out_path)):
        end  = min(start + CHUNK_SIZE, n_total)
        reps = extract(model, tokenizer, texts[start:end], device, batch_size)
        mmap[start:end] = reps
        mmap.flush()
        with open(progress_path, "w") as f:
            json.dump({"n_done": end}, f)

    nbytes = mmap.nbytes
    shape = finalize_building(mmap, out_path)
    os.remove(progress_path)
    print(f"[done] {out_path}  shape={shape}  {nbytes / 1e9:.2f} GB")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",       type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--layer",      type=int, default=LAYER,
                        help="0-indexed transformer layer to extract (default: 25)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Forward-pass batch size. 40G A100→4096, 80G A100→8192")
    parser.add_argument("--data-dir",   default="data/raw")
    parser.add_argument("--out-dir",    default="data/representations/pretrain")
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    print(f"[rank {args.rank}/{args.world_size}] Loading {args.model_name}  layer={args.layer}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()

    suffix = f"rank{args.rank}_of_{args.world_size}"
    for name in ("wikipedia", "scientific"):
        cache_shard(
            jsonl_path=os.path.join(args.data_dir, f"{name}.jsonl"),
            out_path=os.path.join(args.out_dir, f"{name}_{suffix}.npy"),
            rank=args.rank,
            world_size=args.world_size,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )

    if args.rank == 0:
        meta = {
            "model":      args.model_name,
            "layer":      args.layer,
            "hidden_dim": HIDDEN_DIM,
            "dtype":      "bfloat16 as uint16",
            "max_len":    MAX_LEN,
            "batch_size": args.batch_size,
            "world_size": args.world_size,
            "sources":    {"wikipedia": 1_200_000, "scientific": 800_000},
        }
        with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[meta] {args.out_dir}/metadata.json written")


if __name__ == "__main__":
    main()
