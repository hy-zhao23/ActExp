"""
Temporary script: build Llama-3.1-8B eval cache from AO test set.

Classification: loads AO's 250-sample test PT files, decodes context_input_ids
with Qwen tokenizer, re-tokenizes with Llama, runs forward at layer 27,
saves as uint16 npy + csv.

LatentQA: samples from AO eval JSONs (stimulus / stimulus_completion / control),
runs context text through Llama at layer 27, saves as uint16 npy.

Output: data/representations/ao_eval_llama/
  metadata.json
  {dataset}_test.npy            [N, 4096] uint16 bfloat16
  {dataset}_true_false_test.csv  statement, label
  latentqa_{source}_eval.npy    [N, 4096] uint16 bfloat16

Usage:
  python experiments/eval/build_ao_eval_cache.py
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline/activation_oracles"))

PROJ       = Path(__file__).resolve().parents[2]
AO_DATA    = PROJ / "experiments/baseline/activation_oracles"
OUT_DIR    = PROJ / "data/representations/ao_eval_llama"

LLAMA_NAME = "meta-llama/Llama-3.1-8B-Instruct"
QWEN_NAME  = "Qwen/Qwen3-4B"
LAYER      = 27        # same as our main cache
HIDDEN_DIM = 4096
N_CLS      = 250       # samples per classification dataset
N_LQA      = 100       # total latentqa samples per source
BATCH_SIZE = 32
SEED       = 42

CLS_DATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
]

# Per-dataset rule mapping an AO item → 0/1 label aligned with our eval question.
# AO stores multiple (question, target_output) rows per context; our eval uses a
# single canonical question (utils.finetune_dataset.CLS_QUESTIONS), so we read
# ds_label (or target_output where ds_label is None) and map to 0/1 ourselves.
CLS_LABEL_FN = {
    "geometry_of_truth": lambda it: int(it["ds_label"]),                           # '0'/'1'
    "sst2":              lambda it: 1 if it["ds_label"] == "positive" else 0,
    "md_gender":         lambda it: 1 if it["ds_label"] == "female"   else 0,
    "relations":         lambda it: 1 if it["target_output"] == "Yes" else 0,      # ds_label=None
    "snli":              lambda it: 1 if it["target_output"] == "Yes" else 0,      # ds_label=None
    "ner":               lambda it: 1 if it["target_output"] == "Yes" else 0,      # ds_label=None
}

LQA_SOURCES = ["stimulus", "stimulus_completion", "control"]

AO_SFT_DIR = AO_DATA / "sft_training_data"
LQA_EVAL_DIR = AO_DATA / "datasets/latentqa_datasets/eval"


# ── helpers ────────────────────────────────────────────────────────────────────

def bf16_to_uint16(t: torch.Tensor) -> np.ndarray:
    return t.view(torch.int16).cpu().numpy().view(np.uint16)


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode list of texts → (N, hidden_dim) bfloat16 last-token activations."""
    all_vecs = []
    for start in tqdm(range(0, len(texts), batch_size), desc="  encoding", leave=False):
        batch_texts = texts[start : start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        acts = {}
        def hook_fn(mod, inp, out, _layer=layer):
            acts[_layer] = out[0] if isinstance(out, tuple) else out

        submod = model.model.layers[layer]
        handle = submod.register_forward_hook(hook_fn)
        model(**enc)
        handle.remove()

        hidden = acts[layer]           # (B, L, D)
        attn   = enc["attention_mask"] # (B, L)
        # Last non-pad token per sample, independent of padding side.
        last_idx = attn.shape[1] - 1 - attn.flip(dims=(1,)).argmax(dim=1)
        vecs = hidden[torch.arange(hidden.shape[0], device=device), last_idx]  # (B, D)
        all_vecs.append(vecs.cpu().to(torch.bfloat16))

    return torch.cat(all_vecs, dim=0)


def save_npy(vecs: torch.Tensor, path: Path) -> None:
    arr = bf16_to_uint16(vecs)
    # 统一存 npy（自描述 dtype/shape）；oracle_eval_gen.py 走 open_reps_shard 读。
    np.save(str(path), arr)


# ── classification ─────────────────────────────────────────────────────────────

def find_cls_test_pt(dataset: str) -> Path:
    pattern = f"classification_{dataset}_model_Qwen3-4B_n_250_save_acts_False_test_"
    matches = [f for f in AO_SFT_DIR.glob("*.pt") if f.name.startswith(pattern)]
    assert matches, f"No test PT found for {dataset} in {AO_SFT_DIR}"
    return matches[0]


def build_cls_cache(
    dataset: str,
    model: AutoModelForCausalLM,
    llama_tok: AutoTokenizer,
    qwen_tok: AutoTokenizer,
    device: torch.device,
) -> None:
    pt_path = find_cls_test_pt(dataset)
    print(f"  Loading {pt_path.name}")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    items = data["data"]

    # AO expands each context × num_qa_per_sample paraphrase questions → dedup
    # by context_input_ids, keep the first item per unique context.
    by_ctx: dict[tuple, dict] = {}
    for it in items:
        key = tuple(it["context_input_ids"])
        if key not in by_ctx:
            by_ctx[key] = it
    unique_items = list(by_ctx.values())

    rng = random.Random(SEED)
    if len(unique_items) > N_CLS:
        unique_items = rng.sample(unique_items, N_CLS)

    label_fn = CLS_LABEL_FN[dataset]
    texts, labels = [], []
    for item in unique_items:
        ctx_ids = list(item["context_input_ids"])
        text = qwen_tok.decode(ctx_ids, skip_special_tokens=True)
        texts.append(text)
        labels.append(label_fn(item))

    # encode with Llama
    vecs = encode_texts(texts, model, llama_tok, LAYER, BATCH_SIZE, device)

    # save npy
    npy_path = OUT_DIR / f"{dataset}_test.npy"
    save_npy(vecs, npy_path)

    # save csv
    csv_path = OUT_DIR / f"{dataset}_true_false_test.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["statement", "label"])
        writer.writerows(zip(texts, labels))

    print(f"  → {npy_path.name} ({len(vecs)} samples)")


# ── latentqa ──────────────────────────────────────────────────────────────────

def build_lqa_cache(
    source: str,
    model: AutoModelForCausalLM,
    llama_tok: AutoTokenizer,
    device: torch.device,
) -> None:
    json_path = LQA_EVAL_DIR / f"{source}.json"
    items = json.loads(json_path.read_text())

    rng = random.Random(SEED)
    if len(items) > N_LQA:
        items = rng.sample(items, N_LQA)

    # extract context text: stimulus_user or control_user
    texts = []
    for item in items:
        text = item.get("stimulus_user") or item.get("control_user", "")
        texts.append(text)

    vecs = encode_texts(texts, model, llama_tok, LAYER, BATCH_SIZE, device)

    npy_path = OUT_DIR / f"latentqa_{source}_eval.npy"
    save_npy(vecs, npy_path)

    # save source json (same format as original, but subsampled)
    out_json = OUT_DIR / f"{source}.json"
    out_json.write_text(json.dumps(items, ensure_ascii=False, indent=2))

    print(f"  → {npy_path.name} ({len(vecs)} samples)")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    print(f"Loading Qwen tokenizer ({QWEN_NAME})...")
    qwen_tok = AutoTokenizer.from_pretrained(QWEN_NAME)

    print(f"Loading Llama model ({LLAMA_NAME})...")
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_NAME, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model.eval()
    llama_tok = AutoTokenizer.from_pretrained(LLAMA_NAME)
    llama_tok.padding_side = "left"
    if llama_tok.pad_token is None:
        llama_tok.pad_token = llama_tok.eos_token

    # ── classification ──────────────────────────────────────────────────────
    print("\n=== Classification ===")
    for ds in CLS_DATASETS:
        print(f"\n[{ds}]")
        build_cls_cache(ds, model, llama_tok, qwen_tok, device)

    # ── latentqa ────────────────────────────────────────────────────────────
    print("\n=== LatentQA ===")
    for src in LQA_SOURCES:
        print(f"\n[{src}]")
        build_lqa_cache(src, model, llama_tok, device)

    # ── metadata ────────────────────────────────────────────────────────────
    meta = {
        "model":      LLAMA_NAME,
        "layer":      LAYER,
        "hidden_dim": HIDDEN_DIM,
        "dtype":      "bfloat16 as uint16",
        "n_cls":      N_CLS,
        "n_lqa":      N_LQA,
        "seed":       SEED,
        "source":     "AO test set (context text decoded from Qwen tokenizer)",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nDone. Cache saved to {OUT_DIR}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))
    main()
