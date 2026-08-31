"""
Build activation cache for LatentQA eval.

Uses the eval split.  For each of the first 50 items, apply the chat template
to read_prompt, run the base model (no LoRA), and store the last-token
activation at the 75% layer.

Usage:
    python tests/paranet/build_latentqa_cache.py
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

import nl_probes.dataset_classes.misc.latentqa_loader as latentqa_loader
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer


def build_latentqa_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    items: list[dict],
    act_layer: int,
    batch_size: int,
    device: torch.device,
) -> list[dict]:
    """
    For each latentqa item, render read_prompt through the base model and store
    the last-token activation at act_layer.

    Returns list of dicts:
        {
            last_token_act: Tensor[D],   # on CPU
            question:       str,
            answer:         str,
            label:          str,
            source:         str,
        }
    """
    submodule = get_hf_submodule(model, act_layer)
    cache_entries: list[dict] = []

    for start in tqdm(range(0, len(items), batch_size), desc="Building latentqa cache"):
        batch = items[start : start + batch_size]

        rendered = []
        for item in batch:
            add_gen = item["read_prompt"][-1]["role"] != "assistant"
            rendered.append(
                tokenizer.apply_chat_template(
                    item["read_prompt"],
                    tokenize=False,
                    add_generation_prompt=add_gen,
                    enable_thinking=False,
                )
            )

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
        # left-padding → last real token always at seq_len-1
        acts_BD = acts_by_layer[act_layer][:, -1, :].detach().cpu()  # [B, D]

        for b_idx, item in enumerate(batch):
            cache_entries.append(
                {
                    "last_token_act": acts_BD[b_idx],
                    "question": item["dialog"][0]["content"],
                    "answer": item["dialog"][1]["content"],
                    "label": item["label"],
                    "source": item["source"],
                }
            )

    return cache_entries


if __name__ == "__main__":
    # ── configuration ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-4B"
    layer_percent = 75
    batch_size = 16   # read_prompts can be long
    num_samples = 50

    cache_dir = Path(__file__).resolve().parent / "cache"
    eval_data_dir = "datasets/latentqa_datasets/eval"
    # ─────────────────────────────────────────────────────────────────────────

    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = latentqa_loader.DataPaths(
        system=None,  # gemma-2 chat template does not support system role
        stimulus_completion=f"{eval_data_dir}/stimulus_completion.json",
        stimulus=f"{eval_data_dir}/stimulus.json",
        control=f"{eval_data_dir}/control.json",
        qa=f"{eval_data_dir}/qa.json",
    )
    ds = latentqa_loader.load_latentqa_dataset(paths, filter_prefixes=[], train_percent=1.0, seed=42)
    items = [ds[i] for i in range(min(num_samples, len(ds)))]
    print(f"Loaded {len(items)} latentqa eval items")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype)
    model.eval()

    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")
    model.enable_adapters()

    act_layer = layer_percent_to_layer(model_name, layer_percent)
    print(f"Collecting last-token activations at layer {act_layer} ({layer_percent}%)")

    cache_entries = build_latentqa_cache(
        model=model,
        tokenizer=tokenizer,
        items=items,
        act_layer=act_layer,
        batch_size=batch_size,
        device=device,
    )

    out_file = cache_dir / "latentqa_eval.pt"
    torch.save(
        {
            "model_name": model_name,
            "act_layer": act_layer,
            "layer_percent": layer_percent,
            "num_samples": len(cache_entries),
            "entries": cache_entries,
        },
        out_file,
    )
    print(f"Saved {len(cache_entries)} entries → {out_file}")
