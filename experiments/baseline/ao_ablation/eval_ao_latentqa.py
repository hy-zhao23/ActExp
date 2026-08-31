"""
Evaluate a fine-tuned Activation Oracle on LatentQA.

Loads the cache built by build_latentqa_cache.py.  One TrainingDataPoint per
cache entry (num_positions=1, last-token activation).  Uses exact-match
scoring after lowercasing and stripping punctuation.

Usage:
    python tests/paranet/eval_ao_latentqa.py
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
from nl_probes.utils.eval import proportion_confidence, run_evaluation


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


def eval_ao_latentqa(
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
    Run the oracle on latentqa cache entries and return accuracy + per-source stats.

    Each entry has: last_token_act [D], question (str), answer (str),
    label (str), source (str).

    Scoring: generated response contains the ground-truth answer string
    (case-insensitive substring match, since answers are often multi-word).
    """
    injection_submodule = get_hf_submodule(model, injection_layer)

    all_data_points: list[TrainingDataPoint] = []

    for entry in cache_entries:
        acts_1D: torch.Tensor = entry["last_token_act"].to(device).unsqueeze(0)  # [1, D]
        dp = create_training_datapoint(
            datapoint_type="latentqa",
            prompt=entry["question"],
            target_response=entry["answer"],
            layer=act_layer,
            num_positions=1,
            tokenizer=tokenizer,
            acts_BD=acts_1D,
            feature_idx=-1,
            ds_label=entry["label"],
            meta_info={
                "ground_truth": entry["answer"],
                "label": entry["label"],
                "source": entry["source"],
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

    # Word-overlap scoring: fraction of GT content words present in prediction.
    # Threshold ≥ 0.3 counts as correct.  Also record exact-match for reference.
    # Stopwords excluded so common filler words don't inflate the score.
    _STOPWORDS = frozenset(
        "a an the and or but of to in is are was were be been being have has had do does did "
        "will would should could may might shall for with by at from on into as its it its "
        "this that these those i we you he she they all some any its not no nor so yet both "
        "either neither each every more most other such when where who which what how than "
        "if while because since though although s".split()
    )
    WORD_OVERLAP_THRESHOLD = 0.3

    def _content_words(text: str) -> list[str]:
        import re
        tokens = re.findall(r"[a-z]+", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    correct = 0
    exact_correct = 0
    per_source: dict[str, dict] = {}
    samples: list[dict] = []

    for result in results:
        gt: str = result.meta_info["ground_truth"].strip().lower()
        pred: str = result.api_response.strip().lower()
        src: str = result.meta_info["source"]

        # exact substring match (original metric, kept for reference)
        is_exact = gt in pred

        # word-overlap recall
        gt_words = _content_words(gt)
        pred_set = set(_content_words(pred))
        if gt_words:
            recall = sum(1 for w in gt_words if w in pred_set) / len(gt_words)
        else:
            recall = 0.0
        is_correct = recall >= WORD_OVERLAP_THRESHOLD

        if is_correct:
            correct += 1
        if is_exact:
            exact_correct += 1

        if src not in per_source:
            per_source[src] = {"correct": 0, "exact_correct": 0, "n": 0}
        per_source[src]["n"] += 1
        if is_correct:
            per_source[src]["correct"] += 1
        if is_exact:
            per_source[src]["exact_correct"] += 1

        samples.append(
            {
                "question": result.meta_info.get("label", ""),
                "ground_truth": result.meta_info["ground_truth"],
                "prediction": result.api_response,
                "source": src,
                "word_overlap_recall": round(recall, 4),
                "is_correct": is_correct,
                "is_exact": is_exact,
            }
        )

    n = len(results)
    p, se, lower, upper = proportion_confidence(correct, n)

    return {
        "correct": correct,
        "exact_correct": exact_correct,
        "n": n,
        "accuracy": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "per_source": per_source,
        "samples": samples,
    }


if __name__ == "__main__":
    # ── configuration ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-4B"
    oracle_lora_paths = _parse_oracle_lora_paths()

    layer_percent = 75
    injection_layer = int(os.getenv("AO_INJECTION_LAYER", "1"))
    steering_coefficient = 1.0
    eval_batch_size = 128
    generation_kwargs = {"do_sample": False, "max_new_tokens": 60}

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

    cache_file = cache_dir / "latentqa_eval.pt"
    assert cache_file.exists(), f"Cache not found at {cache_file}. Run build_latentqa_cache.py first."

    payload = torch.load(cache_file, weights_only=False)
    assert payload["act_layer"] == act_layer, (
        f"Cache layer {payload['act_layer']} != expected {act_layer}. "
        "Rebuild the cache with layer_percent=75."
    )
    entries = payload["entries"]
    print(f"Loaded {len(entries)} latentqa eval entries")

    for oracle_lora_path in oracle_lora_paths:
        oracle_name = None
        if oracle_lora_path is not None:
            oracle_name = base_experiment.load_lora_adapter(model, oracle_lora_path)
            print(f"\n=== Oracle: {oracle_lora_path} ===")
        else:
            print("\n=== Oracle: base model ===")

        stats = eval_ao_latentqa(
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

        print(
            f"  Overall  word_overlap_acc={stats['accuracy']:.4f}  "
            f"({stats['correct']}/{stats['n']})  "
            f"exact={stats['exact_correct']}/{stats['n']}"
        )
        for src, s in stats["per_source"].items():
            src_p = s["correct"] / s["n"] if s["n"] else 0.0
            print(f"    {src:25s}  acc={src_p:.3f}  ({s['correct']}/{s['n']})")

        results_dir = _get_results_dir(oracle_lora_path, default_results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        label = _make_result_label(oracle_lora_path)
        out_file = results_dir / f"latentqa_eval_{label}.json"
        with open(out_file, "w") as f:
            json.dump(
                {
                    "oracle_lora_path": oracle_lora_path,
                    "model_name": model_name,
                    "act_layer": act_layer,
                    "layer_percent": layer_percent,
                    "scoring": "word_overlap_recall >= 0.3 (content words, stopwords removed)",
                    "overall": {
                        "accuracy": stats["accuracy"],
                        "correct": stats["correct"],
                        "exact_correct": stats["exact_correct"],
                        "n": stats["n"],
                        "se": stats["se"],
                        "ci_lower": stats["ci_lower"],
                        "ci_upper": stats["ci_upper"],
                    },
                    "per_source": stats["per_source"],
                    "samples": stats["samples"],
                },
                f,
                indent=2,
            )
        print(f"Saved → {out_file}")

        if oracle_name is not None:
            model.delete_adapter(oracle_name)
