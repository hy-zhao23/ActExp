"""
Evaluate oracle_vsta checkpoints on our 100-sample-per-source test set.

Reads the per-source caches built by build_latentqa_cache_100.py,
runs all ORACLE_LORA_PATHS on all 3 sources, aggregates, and saves JSON.

Usage:
    cd oracle/tests/ao  (handled by os.chdir)
    python eval_ao_latentqa_100.py

Override checkpoints via env:
    ORACLE_LORA_PATHS="checkpoints/baseline_ao_lora,none,experiments/ao_cls/checkpoints/final"
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

from eval_ao_latentqa import eval_ao_latentqa   # reuse existing eval logic

CACHE_DIR   = Path(__file__).resolve().parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
SOURCES     = ["control", "stimulus", "stimulus_completion"]
MODEL_NAME  = "Qwen/Qwen3-4B"
LAYER_PCT   = 75

_TESTS_AO = Path(__file__).resolve().parent

DEFAULT_LORA_PATHS = [
    "checkpoints/baseline_ao_lora",  # resolves under activation_oracles
    str(_TESTS_AO / "experiments/ao_cls/checkpoints/final"),
    str(_TESTS_AO / "experiments/ao_cls_latentqa/checkpoints/final"),
    str(_TESTS_AO / "experiments/ao_cls_latentqa_ctx/checkpoints/final"),
    None,   # base model
]


def _parse_lora_paths() -> list[Optional[str]]:
    raw = os.getenv("ORACLE_LORA_PATHS")
    if not raw:
        return DEFAULT_LORA_PATHS
    paths: list[Optional[str]] = []
    for item in raw.split(","):
        v = item.strip()
        paths.append(None if not v or v.lower() in {"none", "base"} else v)
    return paths


def _label(path: Optional[str]) -> str:
    if path is None:
        return "base_model"
    p = Path(path)
    if p.name == "final" and p.parent.name == "checkpoints":
        return p.parent.parent.name
    return p.name if p.name else str(p).strip("/").replace("/", "_")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(MODEL_NAME)
    model     = load_model(MODEL_NAME, torch.bfloat16)
    model.eval()

    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PCT)
    injection_layer = int(os.getenv("AO_INJECTION_LAYER", "1"))
    print(f"act_layer={act_layer}  injection_layer={injection_layer}")

    # load all source caches
    all_entries: dict[str, list[dict]] = {}
    for src in SOURCES:
        cache_file = CACHE_DIR / f"latentqa_eval_100_{src}.pt"
        assert cache_file.exists(), f"Cache not found: {cache_file}. Run build_latentqa_cache_100.py first."
        payload = torch.load(cache_file, weights_only=False)
        assert payload["act_layer"] == act_layer, \
            f"Cache layer {payload['act_layer']} != expected {act_layer}"
        all_entries[src] = payload["entries"]
        print(f"Loaded {len(payload['entries'])} entries for source={src}")

    combined_entries = [e for src in SOURCES for e in all_entries[src]]
    print(f"Total entries: {len(combined_entries)} ({len(SOURCES)} sources × 100)")

    lora_paths = _parse_lora_paths()

    for lora_path in lora_paths:
        oracle_name = None
        if lora_path is not None:
            oracle_name = base_experiment.load_lora_adapter(model, lora_path)
            print(f"\n=== {lora_path} ===")
        else:
            print("\n=== base model ===")

        # ── per-source stats ──────────────────────────────────────────────────
        per_source_results = {}
        for src in SOURCES:
            stats = eval_ao_latentqa(
                cache_entries=all_entries[src],
                model=model,
                tokenizer=tokenizer,
                oracle_lora_name=oracle_name,
                act_layer=act_layer,
                injection_layer=injection_layer,
                eval_batch_size=128,
                steering_coefficient=1.0,
                generation_kwargs={"do_sample": False, "max_new_tokens": 60},
                device=device,
            )
            per_source_results[src] = stats
            print(f"  {src:28s}  acc={stats['accuracy']:.3f}  ({stats['correct']}/{stats['n']})")

        # ── aggregate ─────────────────────────────────────────────────────────
        total_correct = sum(r["correct"] for r in per_source_results.values())
        total_n       = sum(r["n"]       for r in per_source_results.values())
        overall_acc   = total_correct / total_n if total_n else 0.0
        print(f"  {'OVERALL (300)':28s}  acc={overall_acc:.3f}  ({total_correct}/{total_n})")

        label = _label(lora_path)
        out   = RESULTS_DIR / f"latentqa_eval_100_{label}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "oracle_lora_path": lora_path,
                    "model_name":       MODEL_NAME,
                    "act_layer":        act_layer,
                    "layer_percent":    LAYER_PCT,
                    "scoring":          "word_overlap_recall >= 0.3",
                    "n_per_source":     100,
                    "overall": {
                        "accuracy": overall_acc,
                        "correct":  total_correct,
                        "n":        total_n,
                    },
                    "per_source": {
                        src: {
                            "accuracy": r["accuracy"],
                            "correct":  r["correct"],
                            "n":        r["n"],
                        }
                        for src, r in per_source_results.items()
                    },
                },
                f,
                indent=2,
            )
        print(f"  Saved → {out}")

        if oracle_name is not None:
            model.delete_adapter(oracle_name)

    print("\nAll done.")
