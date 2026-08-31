"""
Build a Qwen3-4B activation cache that exactly mirrors our 100-sample test set
(seed=42, per-source sampling from oracle_vsta eval JSONs).

Sampling logic matches experiments/eval/oracle_eval_gen.py::load_latentqa_test:
  - flatten (item × qa_pair) for each source
  - sample 100 with np.random.default_rng(42)

read_prompt construction matches oracle_vsta latentqa_loader:
  - control:              [user: control_user]
  - stimulus:             [user:ctl, asst:ctl_model, user:stim]
  - stimulus_completion:  [user:ctl, asst:ctl_model, user:stim, asst:stim_model]

Output: cache/latentqa_eval_100_{source}.pt  (one file per source)
Each file:  { model_name, act_layer, layer_percent, num_samples, entries: [...] }
  entry: { last_token_act: Tensor[D], question: str, answer: str,
           label: str, source: str }

Usage:
    cd oracle/tests/ao  (already handled by the os.chdir below)
    python build_latentqa_cache_100.py
"""

import json
import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_AO_DIR = Path(__file__).resolve().parents[1] / "activation_oracles"
os.chdir(_AO_DIR)

import sys
sys.path.insert(0, str(_AO_DIR))

import numpy as np
import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer

EVAL_DIR  = Path("datasets/latentqa_datasets/eval")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SOURCES   = ["control", "stimulus", "stimulus_completion"]
SEED      = 42
N_SAMPLES = 100
BATCH_SIZE = 16
MODEL_NAME  = "Qwen/Qwen3-4B"
LAYER_PCT   = 75


def build_read_prompt(item: dict, source: str) -> list[dict]:
    """Reconstruct oracle_vsta-style read_prompt from a raw JSON item."""
    cu  = item.get("control_user",    "")
    cm  = item.get("control_model",   "")
    su  = item.get("stimulus_user",   "")
    sm  = item.get("stimulus_model",  "")

    if source == "control":
        return [{"role": "user", "content": cu}]
    elif source == "stimulus":
        return [
            {"role": "user",      "content": cu},
            {"role": "assistant", "content": cm},
            {"role": "user",      "content": su},
        ]
    else:  # stimulus_completion
        return [
            {"role": "user",      "content": cu},
            {"role": "assistant", "content": cm},
            {"role": "user",      "content": su},
            {"role": "assistant", "content": sm},
        ]


def sample_entries(source: str, qa_lookup: dict) -> list[dict]:
    """Reproduce load_latentqa_test sampling: flatten → sample 100 w/ seed=42."""
    items = json.loads((EVAL_DIR / f"{source}.json").read_text())

    flat = []
    for i, item in enumerate(items):
        pairs = qa_lookup.get(item["label"])
        if pairs is None:
            continue
        for question, answer in pairs:
            flat.append({
                "item_idx": i,
                "item": item,
                "question": question,
                "answer": answer,
                "label": item["label"],
                "source": source,
            })

    rng     = np.random.default_rng(SEED)
    chosen  = rng.choice(len(flat), size=min(N_SAMPLES, len(flat)), replace=False).tolist()
    return [flat[i] for i in chosen]


@torch.no_grad()
def compute_acts(
    read_prompts: list[list[dict]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    act_layer: int,
    device: torch.device,
) -> torch.Tensor:
    """Return [N, D] bfloat16 last-token activations."""
    submodule = get_hf_submodule(model, act_layer)
    all_acts  = []

    for start in tqdm(range(0, len(read_prompts), BATCH_SIZE), desc="  activations", leave=False):
        batch = read_prompts[start : start + BATCH_SIZE]
        rendered = []
        for rp in batch:
            add_gen = rp[-1]["role"] != "assistant"
            rendered.append(
                tokenizer.apply_chat_template(
                    rp,
                    tokenize=False,
                    add_generation_prompt=add_gen,
                    enable_thinking=False,
                )
            )

        inputs = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        acts_by_layer = collect_activations_multiple_layers(
            model=model,
            submodules={act_layer: submodule},
            inputs_BL=inputs,
            min_offset=None,
            max_offset=None,
        )
        # left-padding → last real token always at seq_len - 1
        acts_BD = acts_by_layer[act_layer][:, -1, :].detach().cpu()
        all_acts.append(acts_BD)

    return torch.cat(all_acts, dim=0)


def build_source_cache(
    source: str,
    qa_lookup: dict,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    act_layer: int,
    device: torch.device,
) -> list[dict]:
    entries = sample_entries(source, qa_lookup)
    print(f"  {source}: {len(entries)} sampled entries, "
          f"{len(set(e['item_idx'] for e in entries))} unique items")

    # compute activation once per unique item
    unique_idxs  = list(dict.fromkeys(e["item_idx"] for e in entries))
    idx_to_pos   = {idx: pos for pos, idx in enumerate(unique_idxs)}
    unique_items = [None] * len(unique_idxs)
    items_json   = json.loads((EVAL_DIR / f"{source}.json").read_text())
    for idx in unique_idxs:
        unique_items[idx_to_pos[idx]] = items_json[idx]

    read_prompts = [build_read_prompt(item, source) for item in unique_items]
    acts_ND      = compute_acts(read_prompts, model, tokenizer, act_layer, device)  # [N_unique, D]

    cache = []
    for entry in entries:
        pos = idx_to_pos[entry["item_idx"]]
        cache.append({
            "last_token_act": acts_ND[pos],
            "question":       entry["question"],
            "answer":         entry["answer"],
            "label":          entry["label"],
            "source":         entry["source"],
        })
    return cache


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"Loading model {MODEL_NAME}…")
    tokenizer = load_tokenizer(MODEL_NAME)
    model     = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()

    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")
    model.enable_adapters()

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PCT)
    print(f"act_layer={act_layer} ({LAYER_PCT}%)")

    qa_lookup = json.loads((EVAL_DIR / "qa.json").read_text())

    for source in SOURCES:
        print(f"\n[{source}]")
        entries = build_source_cache(source, qa_lookup, model, tokenizer, act_layer, device)

        out_file = CACHE_DIR / f"latentqa_eval_100_{source}.pt"
        torch.save(
            {
                "model_name":   MODEL_NAME,
                "act_layer":    act_layer,
                "layer_percent": LAYER_PCT,
                "source":        source,
                "num_samples":   len(entries),
                "entries":       entries,
            },
            out_file,
        )
        print(f"  → {out_file}  ({len(entries)} entries)")

    print("\nDone.")
