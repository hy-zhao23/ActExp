"""
Cache layer-<L> last-token residual stream from a *VLM-wrapped* donor
(Gemma-4 E4B, Ministral-3 8B, etc.) — variant of cache_finetune_reps.py.

Why a separate script:
- VLMs expose config.hidden_size under config.text_config.hidden_size
- Some (e.g. Mistral3Config) are NOT in AutoModelForCausalLM mapping —
  must load via AutoModelForImageTextToText
- Text decoder layers live under different attribute paths depending on
  wrapper (model.model.layers vs model.language_model.model.layers ...)

Input / output layout and per-shard semantics are identical to the plain
cache_finetune_reps.py — same JSONL input, same .npy shard naming, same
CHUNK_SIZE resume protocol, same metadata.json schema.

Usage
-----
  python experiments/data_prep/cache_finetune_reps_vlm.py \
      --rank 0 --world-size 8 --all \
      --model-name google/gemma-4-E4B-it \
      --layer 35 \
      --out-dir data/representations/finetune_gemma4_e4b \
      [--device-map auto]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))   # oracle root
from utils.reps_io import finalize_building, open_building_shard  # noqa: E402

BATCH_SIZE = 1500
MAX_LEN    = 64
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


# ── VLM-aware helpers ─────────────────────────────────────────────────────────

_LAYER_PATHS = (
    "model.layers",                   # plain decoder-only (Llama/Qwen/Mistral)
    "language_model.model.layers",    # VLM outer: has .language_model
    "model.language_model.layers",    # alt nesting
    "language_model.layers",          # some VLMs expose directly
)


def resolve_text_decoder(model) -> tuple[torch.nn.Module, int]:
    """Return (layers_list, hidden_size) for the text decoder inside model.

    Handles both plain causal LMs and VLM wrappers. Raises loudly if neither
    known path matches — so a new architecture fails fast rather than silently.
    """
    cfg = model.config
    hidden_dim = getattr(cfg, "hidden_size", None)
    if hidden_dim is None:
        text_cfg = getattr(cfg, "text_config", None)
        assert text_cfg is not None, (
            f"config {type(cfg).__name__} has neither hidden_size nor text_config"
        )
        hidden_dim = text_cfg.hidden_size

    for path in _LAYER_PATHS:
        obj = model
        ok = True
        for attr in path.split("."):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok:
            return obj, hidden_dim

    raise AttributeError(
        f"cannot locate text-decoder layers in {type(model).__name__}; "
        f"tried paths: {_LAYER_PATHS}"
    )


def load_model(name: str, device_map, dtype=torch.bfloat16):
    """Load donor model, supporting VLMs that aren't in AutoModelForCausalLM.

    Tries CausalLM first (works for Gemma-4 which registers Gemma4Config
    there), falls back to AutoModelForImageTextToText (Mistral-3 path).
    """
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, device_map=device_map,
        )
    except ValueError as e:
        if "Unrecognized configuration" not in str(e):
            raise
        print(f"[load_model] AutoModelForCausalLM rejected {name}; "
              f"falling back to AutoModelForImageTextToText")
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(
        name, dtype=dtype, device_map=device_map,
    )


def load_tokenizer(name: str):
    """Mistral tokenizers in transformers 5.x emit a regex-fix warning; opt
    in to the corrected version explicitly when available."""
    try:
        tok = AutoTokenizer.from_pretrained(
            name, padding_side="left", fix_mistral_regex=True,
        )
    except TypeError:
        # older/other tokenizers don't accept the flag
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── extraction (input device inferred from model's embedding weight) ─────────

@torch.no_grad()
def extract(model, tokenizer, texts, layer_module, batch_size):
    buf = {}
    handle = layer_module.register_forward_hook(_make_hook(buf))
    results = []
    # inputs must live on the same device as the embedding (entry-point shard)
    in_device = model.get_input_embeddings().weight.device
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt",
            ).to(in_device)
            # snapshot mask BEFORE forward — some models (Yi w/ device_map=auto)
            # mutate enc["attention_mask"] in-place during attention prep, which
            # corrupts the post-forward sum() and produces huge bogus indices.
            attn = enc["attention_mask"].detach().cpu().clone()
            try:
                model(**enc, use_cache=False)
            except _Stop:
                pass
            h = buf["h"]                                 # [B, L, D], cpu bf16
            # Padding-side agnostic last non-pad position.
            last = attn.shape[1] - 1 - attn.flip(dims=(1,)).argmax(dim=1)
            rows = torch.arange(h.shape[0])
            last_h = h[rows, last]
            # fail-fast: cache must be clean. Yi-1.5-34B with device_map=auto has
            # been seen to emit NaN on certain rank/shard combinations (1025718,
            # 1026120 produced 11.8M NaNs in rank-1 shards); silently writing NaN
            # caches NaN-poisons downstream training (loss=nan at step 25).
            nan_count = int(torch.isnan(last_h).sum())
            inf_count = int(torch.isinf(last_h).sum())
            if nan_count or inf_count:
                raise RuntimeError(
                    f"NaN/Inf in extracted hidden states: NaN={nan_count} Inf={inf_count} "
                    f"(batch start={start}, size={len(batch)}); refusing to write corrupt cache"
                )
            results.append(to_uint16(last_h))
    finally:
        handle.remove()
    return np.concatenate(results, axis=0)


def load_shard_texts(jsonl_path, rank, world_size):
    texts = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i % world_size == rank:
                texts.append(json.loads(line)["text"])
    return texts


def cache_shard(jsonl_path, out_path, rank, world_size,
                model, tokenizer, layer_module, batch_size, hidden_dim):
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
        reps = extract(model, tokenizer, texts[start:end], layer_module, batch_size)
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
    ap.add_argument("--jsonl",      default=None)
    ap.add_argument("--all",        action="store_true")
    ap.add_argument("--raw-dir",    default=PROJECT_ROOT / "data/raw/finetune")
    ap.add_argument("--out-dir",    required=True)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--layer",      type=int, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--device-map", default="cuda:0",
                    help='"cuda:0" (single GPU) or "auto" (shard across visible GPUs)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        jsonl_paths = sorted(glob.glob(os.path.join(args.raw_dir, "*.jsonl")))
    elif args.jsonl:
        jsonl_paths = [args.jsonl]
    else:
        raise SystemExit("must specify --jsonl or --all")

    # device_map: str shortcut → dict for single GPU
    if args.device_map == "auto":
        dev_map = "auto"
    else:
        dev_map = {"": args.device_map}

    print(f"[rank {args.rank}/{args.world_size}]  Loading {args.model_name}"
          f"  layer={args.layer}  device_map={dev_map}")
    print(f"[rank {args.rank}/{args.world_size}]  {len(jsonl_paths)} JSONL file(s)")

    tokenizer = load_tokenizer(args.model_name)
    model     = load_model(args.model_name, device_map=dev_map)
    model.eval()

    layer_module, hidden_dim = resolve_text_decoder(model)
    try:
        layer_m = layer_module[args.layer]
    except (IndexError, TypeError) as e:
        raise SystemExit(
            f"layer {args.layer} unreachable: {type(layer_module).__name__} "
            f"with {len(layer_module)} entries — {e}"
        )
    print(f"[rank {args.rank}] text decoder: "
          f"{type(layer_module).__name__}[{args.layer}]  hidden_dim={hidden_dim}")

    suffix = f"rank{args.rank}_of_{args.world_size}"
    for jsonl in jsonl_paths:
        name = os.path.splitext(os.path.basename(jsonl))[0]
        out_path = os.path.join(args.out_dir, f"{name}_{suffix}.npy")
        print(f"\n[rank {args.rank}] {name}")
        cache_shard(jsonl, out_path, args.rank, args.world_size,
                    model, tokenizer, layer_m, args.batch_size, hidden_dim)

    if args.rank == 0:
        counts = {}
        for jsonl in jsonl_paths:
            name = os.path.splitext(os.path.basename(jsonl))[0]
            with open(jsonl) as f:
                counts[name] = sum(1 for _ in f)
        meta = {
            "model":      args.model_name,
            "layer":      args.layer,
            "hidden_dim": hidden_dim,
            "dtype":      "bfloat16 as uint16",
            "max_len":    MAX_LEN,
            "batch_size": args.batch_size,
            "world_size": args.world_size,
            "sources":    counts,
        }
        with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[meta] {args.out_dir}/metadata.json written")


if __name__ == "__main__":
    main()
