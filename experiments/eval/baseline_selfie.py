"""
Open-generation eval — SelfIE baseline (zero-shot self-explanation).

SelfIE = "self-explain-self": the source model interprets its own activations.
We adapt the original `Sure, I'll summarize:` prompt to QA form by replacing
the summarize suffix with the actual question (single-stage SelfIE-QA). No
training, LoRA, or adapter is required.

Mechanism:
    prompt:  <|im_start|>user\n@\n@\n@\n@\n@\n{question}<|im_end|>\n<|im_start|>assistant\n
             ↑  K placeholder tokens patched with the SAME cached activation
    hook  :  post-forward on model.model.layers[k], overwrite K placeholder
             positions with cached_vector at every forward step except KV-cache
             continuation steps (seq_len == 1).
    layer :  source = 27 (cache fixed); inject k = 3 by default (early layer,
             gives the remaining 33 layers room to integrate the patched info).

Usage:
    python experiments/eval/baseline_selfie.py \\
        --inject-layer 3 --num-placeholders 5 \\
        --output-dir   out/eval/selfie_l3_k5
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from experiments.eval._eval_common import (
    OPEN_GEN_SOURCES,
    TEST_SEED,
    VAL_SEED,
    assert_val_test_disjoint,
    build_source_samples,
    build_val_samples,
)

# Default decoder/cache pair (Qwen3-4B). Override with --model-name and
# --reps-subdir to run other "self" models (e.g. Llama-3.1-8B-Instruct paired
# with finetune_diversified, both at layer 27).
DEFAULT_MODEL_NAME  = "Qwen/Qwen3-4B"
DEFAULT_DONOR_LAYER = 27
DEFAULT_REPS_SUBDIR = "finetune_diversified_qwen3_l27"

# Use a unique 8-char marker string that we KNOW won't appear in any natural
# question; render the chat template as a STRING, split on it, then manually
# splice in K placeholder token ids. This bypasses BPE-merging issues that
# happen when the placeholder neighbors `<|im_start|>` etc.
MARKER = "SELFIEPATCH"  # SOH-wrapped, won't survive any normal tokenization

# Placeholder token id (per-tokenizer): pick a single-token char that is
# guaranteed to stay 1 token regardless of context. `unk_token_id` works
# everywhere; otherwise fall back to a rare punctuation.
def _placeholder_id(tokenizer) -> int:
    if tokenizer.unk_token_id is not None:
        return tokenizer.unk_token_id
    # Qwen3 has no unk; use a rare reserved special token if present
    for tok in ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>",
                "<|reserved_special_token_0|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.convert_tokens_to_ids(tokenizer.unk_token or ""):
            return tid
    # Last resort: encode `@` and use its id (we'll just splice it in directly)
    ids = tokenizer.encode("@", add_special_tokens=False)
    assert len(ids) == 1
    return ids[0]


def build_prompt(tokenizer, question: str, num_placeholders: int, placeholder_id: int):
    """Returns (input_ids tensor [1, L], list of K consecutive patch positions).

    Renders chat template with a unique MARKER, splits on it, then splices
    K placeholder token ids at the marker site. Guaranteed K positions.
    """
    user_content = MARKER + question
    prompt_str: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert MARKER in prompt_str, \
        f"MARKER vanished from rendered template — check chat_template behavior"
    left_str, right_str = prompt_str.split(MARKER, 1)

    left_ids  = tokenizer.encode(left_str,  add_special_tokens=False)
    right_ids = tokenizer.encode(right_str, add_special_tokens=False)
    prompt_ids = left_ids + [placeholder_id] * num_placeholders + right_ids
    patch_positions = list(range(len(left_ids), len(left_ids) + num_placeholders))

    return torch.tensor([prompt_ids], dtype=torch.long), patch_positions


def make_patch_hook(positions: list[int], vector: torch.Tensor):
    """Post-forward hook: copy vector into all `positions` of the layer output.

    Skips KV-cache continuation steps (seq_len == 1) — at that point the
    placeholder tokens are already cached and we'd be smashing the *current*
    generated token's hidden state instead.
    """
    def hook(module, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        if out.shape[1] == 1:
            return output
        for pos in positions:
            out[0, pos, :] = vector.to(out.dtype)
        return output
    return hook


@torch.inference_mode()
def generate_answer(
    model, tokenizer, vector: torch.Tensor, question: str,
    inject_layer: int, num_placeholders: int, placeholder_id: int,
    device: torch.device, max_new_tokens: int, no_patch: bool = False,
) -> str:
    input_ids, patch_positions = build_prompt(
        tokenizer, question, num_placeholders, placeholder_id,
    )
    input_ids = input_ids.to(device)
    attn      = torch.ones_like(input_ids, dtype=torch.bool)

    handle = None
    if not no_patch:
        handle = model.model.layers[inject_layer].register_forward_hook(
            make_patch_hook(patch_positions, vector.to(device))
        )
    try:
        out = model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    finally:
        if handle is not None:
            handle.remove()

    return tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject-layer",     type=int, default=3,
                    help="layer index k where we patch (post-hook on model.model.layers[k])")
    ap.add_argument("--num-placeholders", type=int, default=5,
                    help="K — how many copies of the same activation to inject")
    ap.add_argument("--split",            choices=["test", "val"], default="test",
                    help="test = held-out eval split (for main table); "
                         "val = train-tail samples (for hyperparam sweep, "
                         "never reported in main table)")
    ap.add_argument("--n-samples",        type=int, default=111,
                    help="per source; default 111 (≈1000 total over 9 sources)")
    ap.add_argument("--seed",             type=int, default=None,
                    help="default: TEST_SEED=42 for split=test, VAL_SEED=1337 for split=val. "
                         "Overriding test seed will desync from ours/AO — don't.")
    ap.add_argument("--max-new-tokens",   type=int, default=128)
    ap.add_argument("--output-dir",       required=True)
    ap.add_argument("--model-name",       default=DEFAULT_MODEL_NAME,
                    help="HF model id for decoder (must == source model — SelfIE "
                         "is self-interpretation, decoder = source)")
    ap.add_argument("--donor-layer",      type=int, default=DEFAULT_DONOR_LAYER,
                    help="record-only: the layer the cached vectors come from "
                         "(used in summary.txt, must match --reps-subdir)")
    ap.add_argument("--reps-subdir",      default=DEFAULT_REPS_SUBDIR,
                    help="must come from --model-name's layer --donor-layer")
    ap.add_argument("--sources",          nargs="*", default=OPEN_GEN_SOURCES)
    ap.add_argument("--variant-name",     default="selfie",
                    help="key under which the generation is recorded in the JSONL")
    ap.add_argument("--no-patch",         action="store_true",
                    help="control: skip hook registration, generate from prompt only "
                         "(reveals what the model produces from name+question alone)")
    args = ap.parse_args()

    sample_fn = build_val_samples if args.split == "val" else build_source_samples
    seed = args.seed if args.seed is not None else (VAL_SEED if args.split == "val" else TEST_SEED)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[selfie] device={device}  output={out_dir}", flush=True)
    print(f"[selfie] model={args.model_name}  donor_layer={args.donor_layer}  "
          f"inject_layer={args.inject_layer}  K={args.num_placeholders}", flush=True)
    print(f"[selfie] reps_subdir={args.reps_subdir}", flush=True)

    print("[selfie] loading tokenizer + model…", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    # Pick a single-token placeholder id and verify the splice produces K slots.
    placeholder_id = _placeholder_id(tokenizer)
    sanity_ids, sanity_positions = build_prompt(
        tokenizer, "test question", args.num_placeholders, placeholder_id,
    )
    print(f"[selfie] placeholder_id={placeholder_id} ({tokenizer.decode([placeholder_id])!r})  "
          f"slots={len(sanity_positions)} at {sanity_positions}  "
          f"prompt_len={sanity_ids.shape[1]}", flush=True)
    assert len(sanity_positions) == args.num_placeholders

    # Per-source samples — test draws share TEST_SEED with ours / baseline_ao;
    # val draws come from train region with VAL_SEED. Cross-check disjointness.
    datasets: list[tuple[str, list[dict]]] = []
    for src in args.sources:
        samples = sample_fn(src, args.n_samples, seed, args.reps_subdir)
        if args.split == "val":
            # Belt-and-suspenders: also pull the test draw and assert no overlap.
            test_peek = build_source_samples(src, args.n_samples, TEST_SEED, args.reps_subdir)
            assert_val_test_disjoint(samples, test_peek, source_name=src)
        datasets.append((src, samples))
        print(f"[data] {src:24s}  {len(samples)} samples ({args.split}, seed={seed})")

    summary_lines: list[str] = []
    for src, samples in datasets:
        out_path = out_dir / f"{src}.jsonl"
        print(f"\n{'='*70}\n[selfie] source={src}  n={len(samples)}")
        with out_path.open("w") as fout:
            for i, s in enumerate(samples):
                ans = generate_answer(
                    model, tokenizer, s["vector"], s["question"],
                    args.inject_layer, args.num_placeholders, placeholder_id,
                    device, args.max_new_tokens, no_patch=args.no_patch,
                )
                record = {
                    "source": src, "idx": s["idx"], "slot_kind": s["slot_kind"],
                    "question": s["question"], "gt": s["gt"], "input_text": s["input_text"],
                    args.variant_name: ans,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i < 2:
                    print(f"  [{i+1}] slot={s['slot_kind']}  Q: {s['question'][:90]}")
                    print(f"       GT: {s['gt'][:120]}")
                    print(f"       {args.variant_name}: {ans[:120]}")
        summary_lines.append(f"{src}: {len(samples)} → {out_path.name}")
        print(f"[selfie] saved {out_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        f"method: selfie\n"
        f"variant: {args.variant_name}\n"
        f"split: {args.split}\n"
        f"model: {args.model_name}  donor_layer: {args.donor_layer}  "
        f"inject_layer: {args.inject_layer}  K: {args.num_placeholders}\n"
        f"n_samples: {args.n_samples}  seed: {seed}\n"
        f"reps_subdir: {args.reps_subdir}\n\n"
        + "\n".join(summary_lines)
    )
    print(f"\n[selfie] ✓ summary → {summary_path}")


if __name__ == "__main__":
    main()
