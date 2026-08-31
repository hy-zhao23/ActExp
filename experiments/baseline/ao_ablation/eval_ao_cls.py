"""
Evaluate a fine-tuned Activation Oracle on classification tasks.

Loads the cache built by build_cls_cache.py.  One TrainingDataPoint per cache
entry (num_positions=1, last-token activation).  Scores exact yes/no match.

Usage:
    python tests/paranet/eval_ao_cls.py
"""

import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.chdir(Path(__file__).resolve().parents[1] / "activation_oracles")

import json
from typing import Optional

import torch
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import nl_probes.base_experiment as base_experiment
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.utils.eval import parse_answer, proportion_confidence, run_evaluation


def _parse_oracle_lora_paths() -> list[Optional[str]]:
    raw = os.getenv("ORACLE_LORA_PATHS")
    if not raw:
        return [
            "checkpoints/baseline_ao_lora",
            None,
        ]

    paths: list[Optional[str]] = []
    for item in raw.split(","):
        value = item.strip()
        if not value or value.lower() in {"none", "base", "base_model"}:
            paths.append(None)
        else:
            paths.append(value)
    return paths


def _make_result_label(oracle_lora_path: Optional[str]) -> str:
    if oracle_lora_path is None:
        return "base_model"

    path = Path(oracle_lora_path)
    if path.name == "final" and path.parent.name:
        return path.parent.name
    return str(path).strip("/").replace("/", "_")


def _get_results_dir(oracle_lora_path: Optional[str], default_dir: Path) -> Path:
    """If checkpoint is under experiments/<setting>/checkpoints, return experiments/<setting>/results."""
    if oracle_lora_path is None:
        return default_dir
    path = Path(oracle_lora_path).resolve()
    for p in path.parents:
        if p.name == "checkpoints" and p.parent.parent.name == "experiments":
            return p.parent / "results"
    return default_dir


def eval_ao_cls(
    cache_entries: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    oracle_lora_name: str | None,
    act_layer: int,
    injection_layer: int,
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
) -> dict:
    """
    Run the oracle on classification cache entries and return accuracy stats.

    Each entry has: last_token_act [D], classification_prompt (already includes
    "Answer with 'Yes' or 'No' only."), target_response ("Yes" or "No").
    """
    injection_submodule = get_hf_submodule(model, injection_layer)

    all_data_points: list[TrainingDataPoint] = []

    for entry in cache_entries:
        acts_1D: torch.Tensor = entry["last_token_act"].to(device).unsqueeze(0)  # [1, D]
        dp = create_training_datapoint(
            datapoint_type="classification",
            prompt=entry["classification_prompt"],
            target_response=entry["target_response"],
            layer=act_layer,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=acts_1D,
            feature_idx=-1,
            ds_label=entry["ds_label"],
            meta_info={
                "ground_truth": entry["target_response"],
                "ds_label": entry["ds_label"],
            },
        )
        all_data_points.append(dp)

    if oracle_lora_name is not None:
        model.set_adapter(oracle_lora_name)

    results = run_evaluation(
        eval_data=all_data_points,
        model=model,
        tokenizer=tokenizer,
        submodule=injection_submodule,
        device=device,
        dtype=torch.bfloat16,
        global_step=-1,
        lora_path=oracle_lora_name,
        eval_batch_size=eval_batch_size,
        steering_coefficient=steering_coefficient,
        generation_kwargs=generation_kwargs,
    )

    correct = 0
    valid_answers = {"yes", "no"}
    format_correct = 0

    for result in results:
        gt = parse_answer(result.meta_info["ground_truth"])
        pred = parse_answer(result.api_response)
        if pred in valid_answers:
            format_correct += 1
        if pred == gt:
            correct += 1

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)
    fmt_rate = format_correct / n if n else 0.0

    return {
        "correct": correct,
        "n": n,
        "accuracy": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "format_rate": fmt_rate,
    }


if __name__ == "__main__":
    # ── configuration ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-4B"
    oracle_lora_paths = _parse_oracle_lora_paths()

    layer_percent = 75
    injection_layer = int(os.getenv("AO_INJECTION_LAYER", "1"))
    steering_coefficient = 1.0
    eval_batch_size = 256
    generation_kwargs = {"do_sample": False, "max_new_tokens": 10}

    cache_dir = Path(__file__).resolve().parent / "cache"
    default_results_dir = Path(__file__).resolve().parent / "results"
    # ─────────────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype)
    model.eval()

    dummy_config = LoraConfig()
    model.add_adapter(dummy_config, adapter_name="default")

    act_layer = layer_percent_to_layer(model_name, layer_percent)
    print(f"Using last-token activations from layer {act_layer} ({layer_percent}%)")

    cache_files = sorted(cache_dir.glob("cls_*.pt"))
    assert cache_files, f"No cls cache files found in {cache_dir}. Run build_cls_cache.py first."

    for oracle_lora_path in oracle_lora_paths:
        oracle_name = None
        if oracle_lora_path is not None:
            oracle_name = base_experiment.load_lora_adapter(model, oracle_lora_path)
            print(f"\n=== Oracle: {oracle_lora_path} ===")
        else:
            print("\n=== Oracle: base model ===")

        all_correct = 0
        all_n = 0
        per_dataset: dict[str, dict] = {}

        for cache_file in tqdm(cache_files, desc="Datasets"):
            payload = torch.load(cache_file, weights_only=False)

            assert payload["act_layer"] == act_layer, (
                f"Cache layer {payload['act_layer']} != expected {act_layer}. "
                "Rebuild the cache with layer_percent=75."
            )

            ds_name = payload["dataset_name"]
            entries = payload["entries"]

            stats = eval_ao_cls(
                cache_entries=entries,
                model=model,
                tokenizer=tokenizer,
                oracle_lora_name=oracle_name,
                act_layer=act_layer,
                injection_layer=injection_layer,
                eval_batch_size=eval_batch_size,
                steering_coefficient=steering_coefficient,
                generation_kwargs=generation_kwargs,
                device=device,
            )

            per_dataset[ds_name] = stats
            all_correct += stats["correct"]
            all_n += stats["n"]
            print(
                f"  {ds_name:35s}  acc={stats['accuracy']:.3f}  "
                f"({stats['correct']}/{stats['n']})  fmt={stats['format_rate']:.3f}"
            )

        p, se, lower, upper = proportion_confidence(all_correct, all_n)
        print(
            f"\nOverall  acc={p:.4f}  ({all_correct}/{all_n})  "
            f"95% CI [{lower:.4f}, {upper:.4f}]  SE={se:.4f}"
        )

        results_dir = _get_results_dir(oracle_lora_path, default_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        label = _make_result_label(oracle_lora_path)
        out_file = results_dir / f"cls_eval_{label}.json"
        with open(out_file, "w") as f:
            json.dump(
                {
                    "oracle_lora_path": oracle_lora_path,
                    "model_name": model_name,
                    "act_layer": act_layer,
                    "layer_percent": layer_percent,
                    "overall": {
                        "accuracy": p,
                        "correct": all_correct,
                        "n": all_n,
                        "se": se,
                        "ci_lower": lower,
                        "ci_upper": upper,
                    },
                    "per_dataset": per_dataset,
                },
                f,
                indent=2,
            )
        print(f"Saved → {out_file}")

        if oracle_name is not None:
            model.delete_adapter(oracle_name)
