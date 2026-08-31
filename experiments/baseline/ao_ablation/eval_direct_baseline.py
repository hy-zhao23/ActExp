"""
Direct baseline eval: feed activation_prompt + classification_prompt
directly to the base model (no LoRA, no activation injection).

Compares against the oracle approach in eval_ao_cls.py.

Usage:
    python tests/ao/eval_direct_baseline.py
"""

import os
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.chdir(Path(__file__).resolve().parents[1] / "activation_oracles")

import json

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import parse_answer, proportion_confidence


def eval_direct(
    cache_entries: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    batch_size: int,
    generation_kwargs: dict,
    device: torch.device,
) -> dict:
    """
    Concatenate activation_prompt + classification_prompt as a single user
    message, run the base model directly, parse Yes/No answer.
    """
    all_correct = 0
    all_format_correct = 0
    n = len(cache_entries)

    for start in tqdm(range(0, n, batch_size), desc="  Evaluating", leave=False):
        batch = cache_entries[start : start + batch_size]

        messages_list = [
            [{"role": "user", "content": e["activation_prompt"] + "\n" + e["classification_prompt"]}]
            for e in batch
        ]
        rendered = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            for m in messages_list
        ]
        inputs = tokenizer(
            rendered, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        for i, entry in enumerate(batch):
            new_tokens = output_ids[i, input_len:]
            pred_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = parse_answer(pred_text)
            gt = parse_answer(entry["target_response"])
            if pred in {"yes", "no"}:
                all_format_correct += 1
            if pred == gt:
                all_correct += 1

    p, se, lower, upper = proportion_confidence(all_correct, n)
    return {
        "correct": all_correct,
        "n": n,
        "accuracy": p,
        "se": se,
        "ci_lower": lower,
        "ci_upper": upper,
        "format_rate": all_format_correct / n if n else 0.0,
    }


if __name__ == "__main__":
    # ── configuration ────────────────────────────────────────────────────────
    model_name = "Qwen/Qwen3-4B"
    batch_size = 256
    generation_kwargs = {"do_sample": False, "max_new_tokens": 10}

    cache_dir = Path(__file__).resolve().parent / "cache"
    results_dir = Path(__file__).resolve().parent / "results"
    # ─────────────────────────────────────────────────────────────────────────

    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, dtype)
    model.eval()

    cache_files = sorted(cache_dir.glob("cls_*.pt"))
    assert cache_files, f"No cls cache files found in {cache_dir}"

    all_correct = 0
    all_n = 0
    per_dataset: dict = {}

    for cache_file in tqdm(cache_files, desc="Datasets"):
        payload = torch.load(cache_file, weights_only=False)
        ds_name = payload["dataset_name"]
        entries = payload["entries"]

        stats = eval_direct(
            cache_entries=entries,
            model=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            generation_kwargs=generation_kwargs,
            device=device,
        )

        per_dataset[ds_name] = stats
        all_correct += stats["correct"]
        all_n += stats["n"]
        print(
            f"  {ds_name:35s}  acc={stats['accuracy']:.3f}"
            f"  ({stats['correct']}/{stats['n']})  fmt={stats['format_rate']:.3f}"
        )

    p, se, lower, upper = proportion_confidence(all_correct, all_n)
    print(
        f"\nOverall  acc={p:.4f}  ({all_correct}/{all_n})  "
        f"95% CI [{lower:.4f}, {upper:.4f}]  SE={se:.4f}"
    )

    out_file = results_dir / "cls_eval_direct_baseline.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "method": "direct_baseline",
                "model_name": model_name,
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
