"""
Gemma-2-9B version of build_latentqa_cache_100.py.

Same sampling (seed=42, 100/source), but runs Gemma-2-9B base model at layer 75%.
Gemma-2 chat template does not support system role — only user/assistant.

Output: cache/latentqa_eval_100_gemma_{source}.pt
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

EVAL_DIR   = Path("datasets/latentqa_datasets/eval")
CACHE_DIR  = Path(__file__).resolve().parent / "cache"
SOURCES    = ["control", "stimulus", "stimulus_completion"]
SEED       = 42
N_SAMPLES  = 100
BATCH_SIZE = 8   # Gemma-2-9B is bigger → smaller batch
MODEL_NAME = "google/gemma-2-9b-it"
LAYER_PCT  = 75


def build_read_prompt(item: dict, source: str) -> list[dict]:
    cu = item.get("control_user",   "")
    cm = item.get("control_model",  "")
    su = item.get("stimulus_user",  "")
    sm = item.get("stimulus_model", "")

    if source == "control":
        return [{"role": "user", "content": cu}]
    elif source == "stimulus":
        return [
            {"role": "user",      "content": cu},
            {"role": "assistant", "content": cm},
            {"role": "user",      "content": su},
        ]
    else:
        return [
            {"role": "user",      "content": cu},
            {"role": "assistant", "content": cm},
            {"role": "user",      "content": su},
            {"role": "assistant", "content": sm},
        ]


def sample_entries(source: str, qa_lookup: dict) -> list[dict]:
    items = json.loads((EVAL_DIR / f"{source}.json").read_text())
    flat = []
    for i, item in enumerate(items):
        pairs = qa_lookup.get(item["label"])
        if pairs is None:
            continue
        for q, a in pairs:
            flat.append({"item_idx": i, "question": q, "answer": a,
                         "label": item["label"], "source": source})
    rng = np.random.default_rng(SEED)
    idxs = rng.choice(len(flat), size=min(N_SAMPLES, len(flat)), replace=False).tolist()
    return [flat[i] for i in idxs]


@torch.no_grad()
def compute_acts(read_prompts, model, tokenizer, act_layer, device):
    submodule = get_hf_submodule(model, act_layer)
    all_acts = []
    for start in tqdm(range(0, len(read_prompts), BATCH_SIZE), desc="  acts", leave=False):
        batch = read_prompts[start : start + BATCH_SIZE]
        rendered = []
        for rp in batch:
            add_gen = rp[-1]["role"] != "assistant"
            rendered.append(
                tokenizer.apply_chat_template(
                    rp, tokenize=False, add_generation_prompt=add_gen,
                )
            )
        inputs = tokenizer(rendered, return_tensors="pt",
                           add_special_tokens=False, padding=True).to(device)
        acts_by_layer = collect_activations_multiple_layers(
            model=model, submodules={act_layer: submodule},
            inputs_BL=inputs, min_offset=None, max_offset=None,
        )
        acts_BD = acts_by_layer[act_layer][:, -1, :].detach().cpu()
        all_acts.append(acts_BD)
    return torch.cat(all_acts, dim=0)


def build_source_cache(source, qa_lookup, model, tokenizer, act_layer, device):
    entries = sample_entries(source, qa_lookup)
    uniq = list(dict.fromkeys(e["item_idx"] for e in entries))
    pos  = {i: k for k, i in enumerate(uniq)}
    items_json = json.loads((EVAL_DIR / f"{source}.json").read_text())
    unique_items = [items_json[i] for i in uniq]
    prompts = [build_read_prompt(it, source) for it in unique_items]
    acts = compute_acts(prompts, model, tokenizer, act_layer, device)

    out = []
    for e in entries:
        out.append({
            "last_token_act": acts[pos[e["item_idx"]]],
            "question": e["question"], "answer": e["answer"],
            "label": e["label"], "source": e["source"],
        })
    return out


if __name__ == "__main__":
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    print(f"Loading model {MODEL_NAME}…")
    tokenizer = load_tokenizer(MODEL_NAME)
    model     = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()

    dummy = LoraConfig()
    model.add_adapter(dummy, adapter_name="default")
    model.enable_adapters()

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PCT)
    print(f"act_layer={act_layer} ({LAYER_PCT}%)")

    qa_lookup = json.loads((EVAL_DIR / "qa.json").read_text())

    for src in SOURCES:
        print(f"\n[{src}]")
        entries = build_source_cache(src, qa_lookup, model, tokenizer, act_layer, device)
        out_file = CACHE_DIR / f"latentqa_eval_100_gemma_{src}.pt"
        torch.save({
            "model_name": MODEL_NAME,
            "act_layer":  act_layer,
            "layer_percent": LAYER_PCT,
            "source": src,
            "num_samples": len(entries),
            "entries": entries,
        }, out_file)
        print(f"  → {out_file}  ({len(entries)} entries)")

    print("\nDone.")
