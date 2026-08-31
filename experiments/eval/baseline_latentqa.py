"""
Open-generation eval — LatentQA baseline (cached-activation variant).

Inference: load decoder LM + LatentQA LoRA adapter (saved by lit/train.py),
build the write sequence `?` + BASE_DIALOG + question (with add_generation_prompt
=True), patch the cached layer-27 last-token vector at the `?` placeholder
position via a forward hook on decoder layer 0, then greedy-generate.

Output schema matches eval_score.py: one JSONL per source containing
{source, idx, slot_kind, question, gt, input_text, <variant_name>}.

Usage:
    python experiments/eval/baseline_latentqa.py \\
        --variants  qwen:experiments/baseline/latentqa_diversified/out/cached_qwen_20260511_122149/000/checkpoints/epoch3-steps29708-2026-05-11_18-27-13 \\
        --output-dir out/eval/latentqa_qwen_4b

LatentQA's Qwen target is Qwen3-4B-Instruct-2507 (the dedicated no-think
instruct variant; its chat template has no <think> section). For safety we
still pass enable_thinking=False on every chat-template call.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ      = Path(__file__).resolve().parents[2]
LATENTQA  = PROJ / "experiments" / "baseline" / "latentqa_diversified"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(LATENTQA))

from lit.utils.dataset_utils import (  # noqa: E402
    BASE_DIALOG,
    DECODER_CHAT_TEMPLATES,
    NUM_WRITE_TOKENS_TO_SHIFT,
    PAD_TOKEN_IDS,
)

from experiments.eval._eval_common import (  # noqa: E402
    OPEN_GEN_SOURCES,
    build_source_samples,
)


# Map base model -> (reps_subdir for vectors, write-layer index).
# Same wiring as training (lit/configs/train_config.py + exp_args.json):
#   Qwen3-4B-Instruct-2507  -> finetune_diversified_qwen3_l27  (Qwen3-4B l27)
#   Llama-3.1-8B-Instruct   -> finetune_diversified            (Llama-8B l27)
REPS_SUBDIR_FOR_BASE = {
    "Qwen/Qwen3-4B-Instruct-2507":   "finetune_diversified_qwen3_l27",
    "meta-llama/Llama-3.1-8B-Instruct": "finetune_diversified",
}
LAYER_TO_WRITE = 0


def base_model_from_adapter_cfg(ckpt: Path) -> str:
    cfg = json.loads((ckpt / "adapter_config.json").read_text())
    return cfg["base_model_name_or_path"]


def load_tokenizer(base_name: str) -> AutoTokenizer:
    """Mirror lit.infra_utils.get_tokenizer but eval-side (padding side, eos)."""
    tok = AutoTokenizer.from_pretrained(
        base_name, padding_side="left", add_eos_token=True
    )
    tok.pad_token_id = PAD_TOKEN_IDS[base_name]
    return tok


def load_decoder(
    base_name: str, tokenizer, lora_ckpts: dict[str, Path], device: torch.device,
):
    """Load base LM + attach each LoRA adapter under its own adapter_name.

    Mirrors training's resize_token_embeddings(len(tokenizer)) BEFORE PEFT load:
    Qwen3-4B-Instruct-2507 ships with embed padded to 151936 rows but the
    tokenizer only has 151669 entries, and lit/train.py calls
    `model.resize_token_embeddings(len(tokenizer))` which shrinks the embed
    to 151669. We must do the same here or PeftModel.from_pretrained errors
    with a size mismatch on embed_tokens / lm_head.
    """
    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    peft_model = None
    for name, ckpt in lora_ckpts.items():
        print(f"[latentqa] loading variant '{name}' from {ckpt}", flush=True)
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(
                model, ckpt, adapter_name=name,
                is_trainable=False, autocast_adapter_dtype=True,
            )
        else:
            peft_model.load_adapter(ckpt, adapter_name=name, is_trainable=False)
    peft_model.eval()
    return peft_model


def resolve_layer0(peft_model):
    """Return decoder layer 0 module (write hook target), robust to PEFT wrap."""
    for path in ("base_model.model.model.layers", "model.model.layers", "model.layers"):
        cur = peft_model
        ok = True
        for attr in path.split("."):
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok:
            return cur[LAYER_TO_WRITE]
    raise RuntimeError("could not locate decoder layer 0")


def build_write_batch(
    tokenizer, base_name: str, question: str, device: torch.device,
):
    """Tokenize one sample's write sequence (`?` + BASE_DIALOG + user-question).

    Returns a BatchEncoding (on device) plus write_lengths (per the LatentQA
    convention: attention_mask.sum() - NUM_WRITE_TOKENS_TO_SHIFT[name]).
    """
    chat_template = DECODER_CHAT_TEMPLATES[base_name]
    messages = (
        [{"role": "user", "content": "? "}]
        + BASE_DIALOG
        + [{"role": "user", "content": question}]
    )
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        chat_template=chat_template,
        enable_thinking=False,
    )
    enc = tokenizer(
        [text], return_tensors="pt", padding=True, add_special_tokens=False,
    ).to(device)
    write_len = enc.attention_mask.sum(dim=1) - NUM_WRITE_TOKENS_TO_SHIFT[base_name]
    return enc, write_len


@torch.inference_mode()
def generate_answer(
    peft_model, tokenizer, base_name: str, layer0,
    vector: torch.Tensor, question: str,
    device: torch.device, max_new_tokens: int, adapter_name: str,
) -> str:
    """Greedy-generate with the cached activation patched at the `?` placeholder.

    KV-cache-compatible reformulation of the upstream LatentQA hook:
      - For batch=1 / no padding, write_seq_len == orig_len and
        write_mask_len == orig_len - SHIFT, so the splice in
        generate_substitute_layer_single() is mathematically equivalent to
        overwriting layer-0 output at position SHIFT with `vector`.
      - Hook only fires on the prompt forward (seq_len > 1); subsequent
        generation steps (seq_len == 1, drawing from KV cache) skip the patch.
        This lets us use `use_cache=True` and avoid the O(L^2) cost.
    """
    peft_model.set_adapter(adapter_name)

    enc, _ = build_write_batch(tokenizer, base_name, question, device)
    patch_pos = NUM_WRITE_TOKENS_TO_SHIFT[base_name]  # batch=1, no padding
    activation = vector.to(device, dtype=torch.bfloat16)  # (d,)

    fired = [0]
    def patch_hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if h.shape[1] > 1:                              # prompt forward
            h[:, patch_pos, :] = activation
            fired[0] += 1
        return (h, *output[1:]) if is_tuple else h

    handle = layer0.register_forward_hook(patch_hook)
    try:
        out_ids = peft_model.generate(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    finally:
        handle.remove()

    assert fired[0] >= 1, "patch hook never fired — sequence packing wrong?"
    num_prompt_tokens = enc.input_ids.shape[1]
    completion = tokenizer.decode(out_ids[0][num_prompt_tokens:], skip_special_tokens=True)
    return completion.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variants", nargs="+", required=True, metavar="NAME:CKPT",
        help="LatentQA LoRA adapter dir (saved by lit/train.py save_model). "
             "Multiple variants must share the same base model.",
    )
    ap.add_argument("--n-samples",      type=int, default=50)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--sources",        nargs="*", default=OPEN_GEN_SOURCES)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[latentqa] device={device}  output={out_dir}", flush=True)

    # Parse variants & verify they all share the same base model.
    variant_ckpts: dict[str, Path] = {}
    base_names: set[str] = set()
    for spec in args.variants:
        name, dir_str = spec.split(":", 1)
        ck = Path(dir_str)
        assert (ck / "adapter_config.json").exists(), f"no adapter_config.json in {ck}"
        variant_ckpts[name] = ck
        base_names.add(base_model_from_adapter_cfg(ck))
    assert len(base_names) == 1, f"variants must share base model, got {base_names}"
    base_name = next(iter(base_names))
    reps_subdir = REPS_SUBDIR_FOR_BASE[base_name]
    print(f"[latentqa] base={base_name}  reps_subdir={reps_subdir}", flush=True)

    tokenizer = load_tokenizer(base_name)
    peft_model = load_decoder(base_name, tokenizer, variant_ckpts, device)
    layer0 = resolve_layer0(peft_model)
    print(f"[latentqa] write hook -> {type(layer0).__name__} at layer {LAYER_TO_WRITE}",
          flush=True)

    # Build per-source eval samples (test split — same physical files as ours).
    datasets: list[tuple[str, list[dict]]] = []
    for src in args.sources:
        samples = build_source_samples(src, args.n_samples, args.seed, reps_subdir)
        datasets.append((src, samples))
        print(f"[data] {src:24s}  {len(samples)} samples")

    summary_lines: list[str] = []
    for src, samples in datasets:
        out_path = out_dir / f"{src}.jsonl"
        if out_path.exists():
            existing = sum(1 for _ in out_path.open()) if out_path.stat().st_size else 0
            if existing >= len(samples):
                print(f"\n[latentqa] {src}  skip (already have {existing}/{len(samples)})")
                summary_lines.append(f"{src}: {existing} → {out_path.name} (resumed)")
                continue
            else:
                print(f"\n[latentqa] {src}  redoing (have {existing}/{len(samples)})")
        print(f"\n{'='*70}\n[latentqa] source={src}  n={len(samples)}")
        with out_path.open("w") as fout:
            for i, s in enumerate(samples):
                record = {
                    "source": src, "idx": s["idx"], "slot_kind": s["slot_kind"],
                    "question": s["question"], "gt": s["gt"], "input_text": s["input_text"],
                }
                for name in variant_ckpts:
                    record[name] = generate_answer(
                        peft_model, tokenizer, base_name, layer0,
                        s["vector"], s["question"], device, args.max_new_tokens,
                        adapter_name=name,
                    )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i < 2:
                    print(f"  [{i+1}] slot={s['slot_kind']}  Q: {s['question'][:90]}")
                    print(f"       GT: {s['gt'][:120]}")
                    for name in variant_ckpts:
                        print(f"       {name:18s}: {record[name][:120]}")
        summary_lines.append(f"{src}: {len(samples)} → {out_path.name}")
        print(f"[latentqa] saved {out_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        f"method: latentqa\n"
        f"base: {base_name}\n"
        f"variants: {', '.join(variant_ckpts.keys())}\n"
        f"n_samples: {args.n_samples}  seed: {args.seed}\n"
        f"reps_subdir: {reps_subdir}\n\n"
        + "\n".join(summary_lines)
    )
    print(f"\n[latentqa] ✓ summary → {summary_path}")


if __name__ == "__main__":
    main()
