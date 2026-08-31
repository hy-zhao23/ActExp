"""
Open-generation eval — AO baseline (steering-vector hook + LoRA, no MLP adapter).

Reuses our cached last-token activation (same `--reps-subdir` as ours), but
inference goes through AO's steering-hook path:
    prompt = "Layer: 27\\n ?...? \\n" + question      (AO's introspection prefix)
    forward hook at oracle layer 1 injects the activation as a steering vector
    model.generate runs greedy

Usage:
    python experiments/eval/baseline_ao.py \\
        --ao-dirs    ao_qa_lr1e-4:checkpoints/ao_stage2_qa_ft_Qwen3-4B_lr1e-4 \\
        --output-dir out/eval/stage2_ao_qa_lr1e-4
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

PROJ    = Path(__file__).resolve().parents[2]
AO_ROOT = PROJ / "experiments" / "baseline" / "activation_oracles"
sys.path.insert(0, str(PROJ))
sys.path.insert(0, str(AO_ROOT))
sys.path.insert(0, str(PROJ / "experiments" / "baseline" / "ao_stage2_qa_ft"))

# Capture caller cwd before AO's import-time chdir so we can still resolve
# relative paths the user passed on the command line.
_ORIG_CWD = Path.cwd()

# AO's classification_dataset_manager does relative os.listdir at import time;
# chdir into AO_ROOT so the side-effect resolves. FinetuneDataset uses absolute paths.
os.chdir(AO_ROOT)

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_tokenizer
from nl_probes.utils.dataset_utils import (
    construct_batch,
    create_training_datapoint,
    get_prompt_tokens_only,
)
from nl_probes.utils.eval import eval_features_batch
import ao_patches

from experiments.eval._eval_common import (
    OPEN_GEN_SOURCES,
    build_source_samples,
)

# Defaults are for the Qwen3-4B AO setup. Pass --model / --reps-subdir to
# evaluate a different donor (e.g. Llama-3.1-8B). HOOK_LAYER = 1 works for both
# the 36-layer Qwen3-4B and the 32-layer Llama-3.1-8B.
MODEL_NAME      = "Qwen/Qwen3-4B"
DONOR_LAYER     = 27
HOOK_LAYER      = 1
REPS_SUBDIR     = "finetune_diversified_qwen3_l27"
NUM_POSITIONS   = 1


def best_ao_checkpoint(ao_dir: Path) -> Path:
    """AO writes best/ on every eval-loss improvement; prefer it, fall back to final/."""
    for sub in ("best", "final", "latest"):
        p = ao_dir / sub
        if (p / "adapter_config.json").exists():
            print(f"[ckpt] {ao_dir.name}  using {sub}/")
            return p
    raise FileNotFoundError(f"no best/final/latest with adapter_config.json under {ao_dir}")


def load_ao_variant(ao_ckpt: Path, base_model, adapter_name: str):
    """Wrap base_model with AO's LoRA adapter. No MLP adapter — AO uses the hook."""
    peft_model = PeftModel.from_pretrained(
        base_model, ao_ckpt, adapter_name=adapter_name,
        is_trainable=False, autocast_adapter_dtype=True,
    )
    peft_model.eval()
    return peft_model


@torch.inference_mode()
def generate_answer(
    model, tokenizer, submodule, vector: torch.Tensor, question: str,
    device: torch.device, max_new_tokens: int, adapter_name: str,
) -> str:
    """One-sample greedy generate via AO's steering hook + construct_batch path."""
    model.set_adapter(adapter_name)

    # AO expects target_response (only used for label masking; we strip it via
    # get_prompt_tokens_only before generation).
    dp = create_training_datapoint(
        datapoint_type   = "oracle_stage2",
        prompt           = question,
        target_response  = "",
        layer            = DONOR_LAYER,
        num_positions    = NUM_POSITIONS,
        tokenizer        = tokenizer,
        acts_BD          = vector.unsqueeze(0).cpu().clone().detach(),
        feature_idx      = -1,
    )
    dp = get_prompt_tokens_only(dp)
    batch = construct_batch([dp], tokenizer, device)

    results = eval_features_batch(
        eval_batch          = batch,
        model               = model,
        submodule           = submodule,
        tokenizer           = tokenizer,
        device              = device,
        dtype               = torch.bfloat16,
        steering_coefficient= 1.0,
        generation_kwargs   = {"do_sample": False, "max_new_tokens": max_new_tokens,
                               "pad_token_id": tokenizer.pad_token_id,
                               "eos_token_id": tokenizer.eos_token_id},
    )
    return results[0].api_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ao-dirs",        nargs="+", required=True,
                    metavar="NAME:DIR",
                    help="AO ckpt dir; best/ subdir is auto-picked")
    ap.add_argument("--n-samples",      type=int, default=50)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output-dir",     required=True)
    ap.add_argument("--model",          default=MODEL_NAME,
                    help="HF model id of the donor (must match the AO ckpt's training model)")
    ap.add_argument("--reps-subdir",    default=REPS_SUBDIR,
                    help="must match the subdir AO trained on")
    ap.add_argument("--sources",        nargs="*", default=OPEN_GEN_SOURCES)
    args = ap.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ao] device={device}  output={out_dir}", flush=True)

    # Resolve ckpts
    variants_paths: dict[str, Path] = {}
    for spec in args.ao_dirs:
        name, dir_str = spec.split(":", 1)
        p = Path(dir_str)
        if not p.is_absolute():
            p = (_ORIG_CWD / p).resolve()
        variants_paths[name] = best_ao_checkpoint(p)

    # Tokenizer (AO patches chat template to return list, not BatchEncoding)
    tokenizer = load_tokenizer(args.model)
    ao_patches.patch_tokenizer_chat_template(tokenizer)

    # Base model — load once, attach all variants as named adapters
    print("[ao] loading base model…", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    base_model.eval()

    peft_model = None
    for name, ckpt in variants_paths.items():
        print(f"[ao] loading variant '{name}' from {ckpt}…", flush=True)
        peft_model = load_ao_variant(ckpt, base_model, adapter_name=name)

    # get_hf_submodule expects an unwrapped CausalLM (model.model.layers[L]); peft
    # wraps as peft_model.base_model.model. Use the inner unwrapped reference.
    submodule = get_hf_submodule(peft_model.base_model.model, HOOK_LAYER)

    # Per-source eval samples (chdir is AO_ROOT; build_source_samples uses absolute paths)
    datasets: list[tuple[str, list[dict]]] = []
    for src in args.sources:
        samples = build_source_samples(src, args.n_samples, args.seed, args.reps_subdir)
        datasets.append((src, samples))
        print(f"[data] {src:24s}  {len(samples)} samples")

    summary_lines: list[str] = []
    for src, samples in datasets:
        out_path = out_dir / f"{src}.jsonl"
        if out_path.exists():
            existing = sum(1 for _ in out_path.open()) if out_path.stat().st_size else 0
            if existing >= len(samples):
                print(f"\n[ao] source={src}  skip (already have {existing}/{len(samples)} rows)")
                summary_lines.append(f"{src}: {existing} → {out_path.name} (resumed)")
                continue
            else:
                print(f"\n[ao] source={src}  redoing (existing={existing}/{len(samples)} incomplete)")
        print(f"\n{'='*70}\n[ao] source={src}  n={len(samples)}")
        with out_path.open("w") as fout:
            for i, s in enumerate(samples):
                record = {
                    "source": src, "idx": s["idx"], "slot_kind": s["slot_kind"],
                    "question": s["question"], "gt": s["gt"], "input_text": s["input_text"],
                }
                for name in variants_paths:
                    record[name] = generate_answer(
                        peft_model, tokenizer, submodule,
                        s["vector"], s["question"], device, args.max_new_tokens,
                        adapter_name=name,
                    )
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                if i < 2:
                    print(f"  [{i+1}] slot={s['slot_kind']}  Q: {s['question'][:90]}")
                    print(f"       GT: {s['gt'][:120]}")
                    for name in variants_paths:
                        print(f"       {name:18s}: {record[name][:120]}")
        summary_lines.append(f"{src}: {len(samples)} → {out_path.name}")
        print(f"[ao] saved {out_path}")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text(
        f"method: ao\n"
        f"variants: {', '.join(variants_paths.keys())}\n"
        f"n_samples: {args.n_samples}  seed: {args.seed}\n"
        f"reps_subdir: {args.reps_subdir}  donor_layer: {DONOR_LAYER}  "
        f"hook_layer: {HOOK_LAYER}\n\n"
        + "\n".join(summary_lines)
    )
    print(f"\n[ao] ✓ summary → {summary_path}")


if __name__ == "__main__":
    main()
