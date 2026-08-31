"""
Qualitative generation eval — per-dataset, 50 samples each.

Datasets:
  Classification (7): fever, geometry_of_truth, relations, sst2, md_gender, snli, ner
  LatentQA      (3): control, stimulus, stimulus_completion

Checkpoint selection: auto-picks the step with lowest eval_loss from metrics.jsonl
(or use --exact-ckpt to pin a specific step dir).

Usage:
    python experiments/training/oracle_eval_gen.py \\
        --config experiments/training/configs/qwen3_4b.yaml \\
        --stage1-ckpt checkpoints/oracle_act_qwen3_4b/final \\
        --ft-dirs \\
            scheme_a:checkpoints/oracle_ft_qwen3_4b/scheme_a \\
            scheme_b:checkpoints/oracle_ft_qwen3_4b/scheme_b \\
        --n-samples 50 \\
        --output-dir out/eval/gen_compare

Output layout:
    out/eval/gen_compare/
        fever.jsonl
        geometry_of_truth.jsonl
        ...
        latentqa_control.jsonl
        latentqa_stimulus.jsonl
        latentqa_stimulus_completion.jsonl
        summary.txt

Each JSONL line: {"dataset": str, "idx": int, "question": str, "gt": str, "stage1": str, "scheme_a": str, ...}
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.adapter import MLPAdapter
from utils.finetune_dataset import (
    CLASSIFICATION_DATASETS,
    CLS_QUESTIONS,
    LATENTQA_SOURCES,
    LATENTQA_DIR,
    DATA_DIR,
    _uint16_to_bf16,
)
from utils.reps_io import open_reps_shard
from utils.train_config import OracleTrainConfig
from utils.train_utils import ACT_TOKEN


# ── checkpoint selection ───────────────────────────────────────────────────────

def best_checkpoint(scheme_dir: Path) -> Path:
    """Pick the step dir with the lowest eval_loss from metrics.jsonl."""
    metrics_path = scheme_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"metrics.jsonl not found in {scheme_dir}"
    records = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    # deduplicate by step (DDP may log twice), keep last occurrence
    seen: dict[int, dict] = {}
    for r in records:
        seen[r["step"]] = r
    best = min(seen.values(), key=lambda x: x["eval_loss"])
    ckpt = scheme_dir / f"step_{best['step']}"
    assert ckpt.exists(), f"checkpoint dir missing: {ckpt}"
    print(
        f"[ckpt] {scheme_dir.name}  best step={best['step']}"
        f"  eval_loss={best['eval_loss']:.4f}"
        f"  (ppl={torch.exp(torch.tensor(best['eval_loss'])).item():.2f})"
    )
    return ckpt


# ── per-dataset loaders ────────────────────────────────────────────────────────

def load_cls_test(dataset: str, n_samples: int, seed: int, repr_dir: Path | None = None) -> list[dict]:
    """Load classification test samples: {vector, question, gt}."""
    if repr_dir is not None:
        # AO eval cache: npy + csv live in the same dir
        rep_dir  = repr_dir
        npy_path = rep_dir / f"{dataset}_test.npy"
        csv_path = rep_dir / f"{dataset}_true_false_test.csv"
    else:
        rep_dir  = DATA_DIR / "representations" / "finetune"
        npy_path = rep_dir / f"classification_{dataset}_test.npy"
        csv_path = DATA_DIR / "raw" / "finetune" / f"{dataset}_true_false_test.csv"

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    N          = len(rows)
    hidden_dim = json.loads((rep_dir / "metadata.json").read_text())["hidden_dim"]
    arr        = open_reps_shard(npy_path, N, hidden_dim)
    question   = CLS_QUESTIONS[dataset]

    rng     = np.random.default_rng(seed)
    indices = rng.choice(N, size=min(n_samples, N), replace=False).tolist()
    return [
        {"idx": i, "vector": _uint16_to_bf16(arr[i].copy()),
         "question": question,
         "gt": "Yes" if rows[i]["label"] == "1" else "No",
         "input_text": rows[i]["statement"]}
        for i in indices
    ]


def load_latentqa_test(source: str, n_samples: int, seed: int, repr_dir: Path | None = None) -> list[dict]:
    """Load latentqa eval samples: {vector, question, gt} (2 QA pairs per item → flatten)."""
    if repr_dir is not None:
        rep_dir   = repr_dir
        lq_dir    = repr_dir          # subsampled source json lives here
        qa_path   = LATENTQA_DIR / "eval" / "qa.json"
    else:
        rep_dir   = DATA_DIR / "representations" / "finetune"
        lq_dir    = LATENTQA_DIR / "eval"
        qa_path   = lq_dir / "qa.json"

    npy_path   = rep_dir / f"latentqa_{source}_eval.npy"
    json_path  = lq_dir / f"{source}.json"

    hidden_dim   = json.loads((rep_dir / "metadata.json").read_text())["hidden_dim"]
    source_items = json.loads(json_path.read_text())
    qa_lookup    = json.loads(qa_path.read_text())

    N   = len(source_items)
    arr = open_reps_shard(npy_path, N, hidden_dim)

    flat: list[dict] = []
    for i, item in enumerate(source_items):
        pairs = qa_lookup.get(item["label"])
        if pairs is None:
            continue
        input_text = item.get("stimulus_user") or item.get("control_user", "")
        for question, answer in pairs:
            flat.append({"idx": i, "vector": _uint16_to_bf16(arr[i].copy()),
                         "question": question, "gt": answer, "input_text": input_text})

    rng     = np.random.default_rng(seed)
    indices = rng.choice(len(flat), size=min(n_samples, len(flat)), replace=False).tolist()
    return [flat[i] for i in indices]


# ── model helpers ──────────────────────────────────────────────────────────────

def load_base(stage1_ckpt: Path, mc, hidden_dim: int, n_tokens: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(stage1_ckpt / "tokenizer")
    tokenizer.padding_side = "right"
    act_token_id = tokenizer.convert_tokens_to_ids(ACT_TOKEN)

    model = AutoModelForCausalLM.from_pretrained(
        mc.name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.resize_token_embeddings(len(tokenizer))
    model.eval()

    adapter = MLPAdapter(hidden_dim, model.config.hidden_size, n_tokens=n_tokens)
    adapter.load_state_dict(torch.load(stage1_ckpt / "adapter.pt", map_location=device))
    adapter.to(device).to(torch.bfloat16).eval()

    return model, tokenizer, adapter, act_token_id


def load_ft(ft_ckpt: Path, base_model, adapter_s1: MLPAdapter,
            n_tokens: int, device: torch.device, scheme: str,
            adapter_name: str = "default"):
    """Wrap base_model with LoRA.  Returns (peft_model, ft_adapter).

    NOTE: PeftModel.from_pretrained modifies base_model's layers in-place.
    Use unique adapter_name per variant so multiple schemes can coexist.
    Callers must call model.set_adapter(name) before inference.
    """
    peft_model = PeftModel.from_pretrained(
        base_model, ft_ckpt / "lora", adapter_name=adapter_name,
        is_trainable=False, autocast_adapter_dtype=True,
    )
    peft_model.eval()

    if scheme == "b" and (ft_ckpt / "adapter.pt").exists():
        adapter = MLPAdapter(
            adapter_s1.net[1].in_features,
            adapter_s1.lm_dim,
            n_tokens=n_tokens,
        )
        adapter.load_state_dict(torch.load(ft_ckpt / "adapter.pt", map_location=device))
        adapter.to(device).to(torch.bfloat16).eval()
    else:
        adapter = adapter_s1

    return peft_model, adapter


# ── generation ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_answer(
    model, tokenizer, adapter: MLPAdapter,
    vector: torch.Tensor, question: str,
    act_token_id: int, n_tokens: int,
    device: torch.device, max_new_tokens: int,
    adapter_name: str | None = None,  # None → disable all LoRA; str → set_adapter(name)
) -> str:
    user_content = ACT_TOKEN * n_tokens + "\n" + question
    prompt_ids: list[int] = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False,
    )

    inp      = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    base_emb = model.get_input_embeddings()(inp)                                # (1, L, D)
    soft_embs = adapter(vector.unsqueeze(0).to(device, dtype=torch.bfloat16))  # (1, n, D)

    embeds   = base_emb.clone()
    act_pos  = [i for i, t in enumerate(prompt_ids) if t == act_token_id]
    for k, pos in enumerate(act_pos):
        embeds[0, pos] = soft_embs[0, k]

    attn = torch.ones(1, embeds.shape[1], dtype=torch.bool, device=device)
    gen_kwargs = dict(
        inputs_embeds=embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if adapter_name is None:
        with model.disable_adapter():
            out = model.generate(**gen_kwargs)
    else:
        model.set_adapter(adapter_name)
        out = model.generate(**gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        required=True)
    parser.add_argument("--stage1-ckpt",   required=True)
    parser.add_argument("--ft-dirs",       nargs="*", default=[],
                        metavar="NAME:DIR",
                        help="scheme dir containing metrics.jsonl; best step auto-selected")
    parser.add_argument("--exact-ckpts",   nargs="*", default=[],
                        metavar="NAME:PATH",
                        help="pin exact checkpoint step dir (skips metrics.jsonl lookup)")
    parser.add_argument("--n-samples",     type=int, default=50,
                        help="default for both cls and lqa (overridden by --n-samples-cls/lqa)")
    parser.add_argument("--n-samples-cls", type=int, default=None,
                        help="override n_samples for classification datasets")
    parser.add_argument("--n-samples-lqa", type=int, default=None,
                        help="override n_samples for latentqa datasets")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--max-new-tokens",type=int, default=128)
    parser.add_argument("--output-dir",    required=True)
    parser.add_argument("--repr-dir",      default=None,
                        help="override repr cache dir (e.g. data/representations/ao_eval_llama)")
    args = parser.parse_args()

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] device={device}  output={out_dir}", flush=True)

    cfg      = OracleTrainConfig.from_yaml(args.config)
    mc       = cfg.model
    n_tokens = cfg.tasks["activation"].adapter.n_tokens

    # ── resolve checkpoints ───────────────────────────────────────────────────
    # NAME → (ckpt_path, scheme)
    variants: dict[str, tuple[Path, str]] = {}
    for spec in args.ft_dirs:
        name, dir_str = spec.split(":", 1)
        scheme = "b" if "scheme_b" in dir_str else "a"
        variants[name] = (best_checkpoint(Path(dir_str)), scheme)
    for spec in args.exact_ckpts:
        name, path_str = spec.split(":", 1)
        scheme = "b" if "scheme_b" in path_str else "a"
        variants[name] = (Path(path_str), scheme)

    # ── load base model + stage1 adapter ─────────────────────────────────────
    hidden_dim = json.loads(
        (DATA_DIR / "representations" / "finetune" / "metadata.json").read_text()
    )["hidden_dim"]

    print("[eval] loading base model + stage1 adapter…", flush=True)
    base_model, tokenizer, adapter_s1, act_token_id = load_base(
        Path(args.stage1_ckpt), mc, hidden_dim, n_tokens, device
    )

    # variant_models: NAME → (model, adapter, disable_lora)
    # Load ft variants first; PeftModel.from_pretrained wraps base_model in-place,
    # so stage1 must also use the peft_model with disable_adapter() to avoid running LoRA.
    peft_model = None
    ft_entries: dict[str, tuple] = {}
    for name, (ckpt, scheme) in variants.items():
        print(f"[eval] loading {name} from {ckpt}…", flush=True)
        ft_model, ft_adapter = load_ft(ckpt, base_model, adapter_s1, n_tokens, device, scheme,
                                       adapter_name=name)
        peft_model = ft_model   # all calls wrap the same base_model → same object
        ft_entries[name] = (ft_model, ft_adapter, name)  # store adapter_name instead of False

    stage1_model   = peft_model if peft_model is not None else base_model
    stage1_no_lora = peft_model is not None
    # stage1 uses disable_adapter=True (sentinel None means no LoRA)
    variant_models: dict[str, tuple] = {"stage1": (stage1_model, adapter_s1, None)}
    variant_models.update(ft_entries)

    variant_names = list(variant_models.keys())

    # ── build dataset list ────────────────────────────────────────────────────
    repr_dir  = Path(args.repr_dir) if args.repr_dir else None
    n_cls     = args.n_samples_cls if args.n_samples_cls is not None else args.n_samples
    n_lqa     = args.n_samples_lqa if args.n_samples_lqa is not None else args.n_samples
    datasets: list[tuple[str, list[dict]]] = []
    for ds in CLASSIFICATION_DATASETS:
        samples = load_cls_test(ds, n_cls, args.seed, repr_dir)
        datasets.append((ds, samples))
        print(f"[data] {ds}: {len(samples)} test samples")
    for src in LATENTQA_SOURCES:
        samples = load_latentqa_test(src, n_lqa, args.seed, repr_dir)
        datasets.append((f"latentqa_{src}", samples))
        print(f"[data] latentqa_{src}: {len(samples)} eval samples")

    # ── run generation ────────────────────────────────────────────────────────
    summary_lines: list[str] = []

    for ds_name, samples in datasets:
        out_path = out_dir / f"{ds_name}.jsonl"
        print(f"\n{'='*70}")
        print(f"[eval] dataset={ds_name}  n={len(samples)}")

        with out_path.open("w") as fout:
            for i, sample in enumerate(samples):
                question   = sample["question"]
                gt         = sample["gt"]
                input_text = sample["input_text"]
                vector     = sample["vector"]
                record   = {
                    "dataset":    ds_name,
                    "idx":        sample["idx"],
                    "question":   question,
                    "gt":         gt,
                    "input_text": input_text,
                }
                for name, (model, adapter, adapter_name) in variant_models.items():
                    pred = generate_answer(
                        model, tokenizer, adapter, vector, question,
                        act_token_id, n_tokens, device, args.max_new_tokens,
                        adapter_name=adapter_name,
                    )
                    record[name] = pred

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                if i < 3:   # print first 3 per dataset for quick sanity check
                    print(f"  [{i+1}] input: {input_text[:120]}")
                    print(f"       Q : {question}")
                    print(f"       GT: {gt}")
                    for name in variant_names:
                        print(f"       {name:10s}: {record[name]}")

        summary_lines.append(f"{ds_name}: {len(samples)} samples  → {out_path.name}")
        print(f"[eval] saved {out_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    summary_path = out_dir / "summary.txt"
    header = (
        f"variants: {', '.join(variant_names)}\n"
        f"n_samples_cls: {n_cls}\n"
        f"n_samples_lqa: {n_lqa}\n"
        f"seed: {args.seed}\n\n"
    )
    summary_path.write_text(header + "\n".join(summary_lines))
    print(f"\n[eval] ✓ all done  summary → {summary_path}")


if __name__ == "__main__":
    main()
