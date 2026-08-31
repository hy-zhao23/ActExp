"""
Build activation cache for classification eval.

For each dataset, collect last-token activations at the 75% layer from the base
model (no target LoRA).  One cache entry per (activation_prompt, qa_pair).

Usage:
    python tests/paranet/build_cls_cache.py
"""

import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.chdir(Path(__file__).resolve().parents[1] / "activation_oracles")

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.dataset_classes.classification import get_classification_datapoints
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer


def collect_last_token_acts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    act_layer: int,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Tokenize each prompt as a single-user chat message, run the base model,
    and return the last-token activation at act_layer for each prompt.

    Returns list of Tensor[D], one per prompt, on CPU.
    """
    submodule = get_hf_submodule(model, act_layer)
    results: list[torch.Tensor] = []

    for start in tqdm(range(0, len(prompts), batch_size), desc="  Collecting activations", leave=False):
        batch = prompts[start : start + batch_size]
        messages = [[{"role": "user", "content": p}] for p in batch]
        rendered = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for m in messages
        ]
        inputs_BL = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(device)

        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules={act_layer: submodule},
            inputs_BL=inputs_BL,
            min_offset=None,
            max_offset=None,
        )
        # left-padding → last real token is always at seq_len-1
        acts_BD = acts_by_layer[act_layer][:, -1, :].detach().cpu()  # [B, D]
        results.extend(acts_BD.unbind(0))

    return results


if __name__ == "__main__":
    # ── configuration ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-4B"
    layer_percent = 75
    batch_size = 64
    num_test_per_dataset = 500  # number of context samples per dataset
    num_qa_per_sample = 1

    DATASETS = [
        "geometry_of_truth",
        "relations",
        "sst2",
        "md_gender",
        "snli",
        "ag_news",
        "ner",
        "tense",
    ]

    cache_dir = Path(__file__).resolve().parent / "cache"
    # ─────────────────────────────────────────────────────────────────────────

    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype)
    model.eval()

    # Dummy adapter for consistent PeftModel API
    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")
    model.enable_adapters()  # use base weights (no non-default adapter set)

    act_layer = layer_percent_to_layer(model_name, layer_percent)
    print(f"Collecting last-token activations at layer {act_layer} ({layer_percent}%)")

    for dataset_name in DATASETS:
        print(f"\n[{dataset_name}]")
        _, test_dps = get_classification_datapoints(
            dataset_name,
            num_qa_per_sample,
            train_examples=0,
            test_examples=num_test_per_dataset,
            random_seed=42,
        )

        # Collect activations for unique activation_prompts in one pass
        unique_prompts = list(dict.fromkeys(dp.activation_prompt for dp in test_dps))
        acts = collect_last_token_acts(
            model=model,
            tokenizer=tokenizer,
            prompts=unique_prompts,
            act_layer=act_layer,
            batch_size=batch_size,
            device=device,
        )
        prompt_to_act = dict(zip(unique_prompts, acts))

        entries = [
            {
                "last_token_act": prompt_to_act[dp.activation_prompt],
                "activation_prompt": dp.activation_prompt,
                "classification_prompt": dp.classification_prompt,
                "target_response": dp.target_response,
                "ds_label": dp.ds_label,
            }
            for dp in test_dps
        ]

        out_file = cache_dir / f"cls_{dataset_name}.pt"
        torch.save(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "act_layer": act_layer,
                "layer_percent": layer_percent,
                "entries": entries,
            },
            out_file,
        )
        print(f"  Saved {len(entries)} entries → {out_file}")

    print("\nDone.")
