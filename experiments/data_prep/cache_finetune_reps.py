"""
Cache layer-27 last-token residual stream (Llama-3.1-8B-Instruct) for Stage 2
finetune data.

Input  : data/raw/finetune/{source_subtype}.jsonl   (from prepare_data.py)
Output : data/representations/finetune/{source_subtype}_rank{r}_of_{W}.npy

Each JSONL file is processed independently. Row index in the shard = global
line index of the text in the source JSONL (matches FinetuneDataset lookup).

Model / hook / chunked-resume semantics copied from cache_representations.py
so behavior is consistent.

Usage
-----
  # SLURM array over (rank, source-file) — one source per job by default:
  python cache_finetune_reps_v2.py --rank 0 --world-size 1 \
      --jsonl data/raw/finetune/ag_news.jsonl \
      --out-dir data/representations/finetune

  # Or pass --all to loop through all *.jsonl in the raw-finetune dir
  python cache_finetune_reps_v2.py --rank 0 --world-size 1 --all
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))   # oracle root
from utils.reps_io import finalize_building, open_building_shard  # noqa: E402

BATCH_SIZE = 3500
MAX_LEN    = 64
LAYER      = 27
CHUNK_SIZE = 10_000


class _Stop(Exception):
    pass


def _make_hook(buf):
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        buf["h"] = h.detach().cpu()
        raise _Stop
    return hook


def to_uint16(t):
    return t.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


@torch.no_grad()
def extract(model, tokenizer, texts, device, batch_size, layer):
    buf = {}
    handle = model.model.layers[layer].register_forward_hook(_make_hook(buf))
    results = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)
            try:
                model(**enc, use_cache=False)
            except _Stop:
                pass
            # Last non-pad token per row, independent of padding side.
            h = buf["h"]                                 # [B, L, D], cpu bf16
            attn = enc["attention_mask"].cpu()           # [B, L]
            last = attn.shape[1] - 1 - attn.flip(dims=(1,)).argmax(dim=1)
            rows = torch.arange(h.shape[0])
            last_h = h[rows, last]                       # [B, D]
            results.append(to_uint16(last_h))
    finally:
        handle.remove()
    return np.concatenate(results, axis=0)


def load_shard_texts(jsonl_path, rank, world_size):
    """Return texts assigned to `rank` (every `world_size`-th line)."""
    texts = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i % world_size == rank:
                texts.append(json.loads(line)["text"])
    return texts


def cache_shard(jsonl_path, out_path, rank, world_size, model, tokenizer, device, batch_size, hidden_dim, layer):
    progress_path = out_path + ".progress"
    if os.path.exists(out_path) and not os.path.exists(progress_path):
        print(f"[skip] {os.path.basename(out_path)}")
        return

    texts   = load_shard_texts(jsonl_path, rank, world_size)
    n_total = len(texts)
    if n_total == 0:
        print(f"[empty] {os.path.basename(jsonl_path)} rank={rank} → no texts")
        return

    n_done = 0
    if os.path.exists(progress_path):
        raw = json.loads(open(progress_path).read())["n_done"]
        n_done = (raw // CHUNK_SIZE) * CHUNK_SIZE
        print(f"[resume] {os.path.basename(out_path)}: {n_done:,}/{n_total:,}")
    else:
        with open(progress_path, "w") as f:
            json.dump({"n_done": 0}, f)

    # 分块期间裸写 <out>.building（memmap 可增量写、支持续传），全部算完再定稿为 npy
    mmap = open_building_shard(out_path, n_total, hidden_dim)

    for start in tqdm(range(n_done, n_total, CHUNK_SIZE), desc=os.path.basename(out_path)):
        end  = min(start + CHUNK_SIZE, n_total)
        reps = extract(model, tokenizer, texts[start:end], device, batch_size, layer)
        mmap[start:end] = reps
        mmap.flush()
        with open(progress_path, "w") as f:
            json.dump({"n_done": end}, f)

    nbytes = mmap.nbytes
    shape = finalize_building(mmap, out_path)
    os.remove(progress_path)
    print(f"[done] {out_path}  shape={shape}  {nbytes / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank",       type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--jsonl",      default=None, help="single JSONL path")
    ap.add_argument("--all",        action="store_true", help="process every .jsonl in --raw-dir")
    ap.add_argument("--raw-dir",    default=PROJECT_ROOT / "data/raw/finetune")
    ap.add_argument("--out-dir",    default=PROJECT_ROOT / "data/representations/finetune")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--layer",      type=int, default=LAYER)
    ap.add_argument("--model-name", default="meta-llama/Llama-3.1-8B-Instruct")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    if args.all:
        jsonl_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl")))
    elif args.jsonl:
        jsonl_paths = [args.jsonl]
    else:
        raise SystemExit("must specify --jsonl or --all")

    print(f"[rank {args.rank}/{args.world_size}]  Loading {args.model_name}  layer={args.layer}")
    print(f"[rank {args.rank}/{args.world_size}]  {len(jsonl_paths)} JSONL file(s) to process")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()
    hidden_dim = model.config.hidden_size

    suffix = f"rank{args.rank}_of_{args.world_size}"
    for jsonl in jsonl_paths:
        name = os.path.splitext(os.path.basename(jsonl))[0]
        out_path = os.path.join(args.out_dir, f"{name}_{suffix}.npy")
        print(f"\n[rank {args.rank}] {name}")
        cache_shard(jsonl, out_path, args.rank, args.world_size,
                    model, tokenizer, device, args.batch_size, hidden_dim, args.layer)

    # rank 0 writes metadata
    if args.rank == 0:
        counts = {}
        for jsonl in jsonl_paths:
            name = os.path.splitext(os.path.basename(jsonl))[0]
            with open(jsonl) as f:
                counts[name] = sum(1 for _ in f)
        # 往已有 metadata 里**合并**本次处理的源，不要整体覆盖：out-dir 通常是多次
        # 运行、多个源共用的目录，直接覆盖会把其它源的 n_total 全抹掉，导致所有
        # 加载器读不到数据（曾发生过，需从别处副本手工恢复）。
        meta_path = os.path.join(args.out_dir, "metadata.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        sources = {**meta.get("sources", {}), **counts}
        meta.update({
            "model":      args.model_name,
            "layer":      args.layer,
            "hidden_dim": hidden_dim,
            "dtype":      "bfloat16 as uint16",
            "max_len":    MAX_LEN,
            "batch_size": args.batch_size,
            "world_size": args.world_size,
            "sources":    sources,
        })
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[meta] {meta_path} written  (sources: {len(sources)}, +{list(counts)})")


if __name__ == "__main__":
    main()
