"""
Evaluate Gemma-2-9B LoRA adapters on our 100-sample-per-source test set.
"""

import json
import os
from pathlib import Path
from typing import Optional

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_AO_DIR = Path(__file__).resolve().parents[1] / "activation_oracles"
os.chdir(_AO_DIR)

import sys
sys.path.insert(0, str(_AO_DIR))

import torch
from peft import LoraConfig

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer

from eval_ao_latentqa import eval_ao_latentqa

CACHE_DIR   = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCES     = ["control", "stimulus", "stimulus_completion"]
MODEL_NAME  = "google/gemma-2-9b-it"
LAYER_PCT   = 75

# HF hub adapter paths — same as the existing 50-sample experiments
DEFAULT_LORA_PATHS = [
    "adamkarvonen/checkpoints_latentqa_only_addition_gemma-2-9b-it",
    "adamkarvonen/checkpoints_cls_latentqa_only_addition_gemma-2-9b-it",
    None,
]


def _label(path: Optional[str]) -> str:
    if path is None:
        return "base_model"
    return Path(path).name


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(MODEL_NAME)
    model     = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()

    dummy = LoraConfig()
    model.add_adapter(dummy, adapter_name="default")

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PCT)
    injection_layer = int(os.getenv("AO_INJECTION_LAYER", "1"))
    print(f"act_layer={act_layer}  injection_layer={injection_layer}")

    all_entries = {}
    for src in SOURCES:
        f = CACHE_DIR / f"latentqa_eval_100_gemma_{src}.pt"
        assert f.exists(), f"Cache not found: {f}"
        payload = torch.load(f, weights_only=False)
        assert payload["act_layer"] == act_layer
        all_entries[src] = payload["entries"]
        print(f"Loaded {len(payload['entries'])} entries for source={src}")

    for lora_path in DEFAULT_LORA_PATHS:
        oracle_name = None
        if lora_path is not None:
            oracle_name = base_experiment.load_lora_adapter(model, lora_path)
            print(f"\n=== {lora_path} ===")
        else:
            print("\n=== base model ===")

        per_src = {}
        for src in SOURCES:
            stats = eval_ao_latentqa(
                cache_entries=all_entries[src],
                model=model, tokenizer=tokenizer,
                oracle_lora_name=oracle_name,
                act_layer=act_layer,
                injection_layer=injection_layer,
                eval_batch_size=64,
                steering_coefficient=1.0,
                generation_kwargs={"do_sample": False, "max_new_tokens": 60},
                device=device,
            )
            per_src[src] = stats
            print(f"  {src:28s}  acc={stats['accuracy']:.3f}  ({stats['correct']}/{stats['n']})")

        c = sum(r["correct"] for r in per_src.values())
        n = sum(r["n"]       for r in per_src.values())
        acc = c / n if n else 0.0
        print(f"  {'OVERALL (300)':28s}  acc={acc:.3f}  ({c}/{n})")

        label = _label(lora_path)
        out = RESULTS_DIR / f"latentqa_eval_100_gemma_{label}.json"
        with open(out, "w") as f:
            json.dump({
                "oracle_lora_path": lora_path,
                "model_name": MODEL_NAME,
                "act_layer": act_layer,
                "layer_percent": LAYER_PCT,
                "scoring": "word_overlap_recall >= 0.3",
                "n_per_source": 100,
                "overall": {"accuracy": acc, "correct": c, "n": n},
                "per_source": {s: {"accuracy": r["accuracy"], "correct": r["correct"], "n": r["n"]}
                               for s, r in per_src.items()},
            }, f, indent=2)
        print(f"  Saved → {out}")

        if oracle_name is not None:
            model.delete_adapter(oracle_name)

    print("\nAll done.")
