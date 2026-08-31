"""
Sanity test: can batch=32 inference fit on a 40G A100 MIG slice?

Loads 200 real prompts sampled across subtypes, runs vLLM at batch=32,
reports peak VRAM + throughput. Tries AWQ first, then bf16 fallback
(commented out — set MODEL to change).

Run:
    python -m scripts.test_vllm_40g
"""

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from scripts.qa_schemas import build_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune"

MODEL       = "Qwen/Qwen3-14B"     # bf16; switch to "Qwen/Qwen3-14B-AWQ" once cached
DTYPE       = "bfloat16"           # set to "auto" for AWQ
QUANTIZATION = None                # set to "awq" for AWQ builds
N_SAMPLES   = 200
BATCH_SIZE  = 32
MAX_NEW_TOK = 300
MAX_MODEL_LEN = 1024               # prompt ≤ 400 + output 300, 1024 plenty
GPU_MEM_UTIL = 0.90


def sample_prompts(tok) -> list[str]:
    """Mix of factual + comprehension (with and without labels).

    Wraps each user prompt in Qwen3 chat template with enable_thinking=False
    (otherwise the model dumps reasoning into <think>...</think> and wastes
    max_new_tokens, destroying the JSON output we want).
    """
    stems = sorted(p.stem for p in RAW_DIR.glob("*.jsonl"))
    per   = max(1, N_SAMPLES // len(stems))
    rng   = random.Random(0)
    prompts = []
    for stem in stems:
        lines = (RAW_DIR / f"{stem}.jsonl").open().readlines()
        rng.shuffle(lines)
        for line in lines[:per]:
            rec = json.loads(line)
            user_msg = build_prompt(stem, rec["text"], rec.get("meta"))
            chat = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(chat)
        if len(prompts) >= N_SAMPLES:
            break
    return prompts[:N_SAMPLES]


def main():
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = sample_prompts(tok)
    prompt_tok_lens = [len(tok(p).input_ids) for p in prompts]
    print(f"[test] prompts={len(prompts)}  "
          f"prompt_tok min/med/max = "
          f"{min(prompt_tok_lens)}/"
          f"{sorted(prompt_tok_lens)[len(prompt_tok_lens)//2]}/"
          f"{max(prompt_tok_lens)}")

    free_b, total_b = torch.cuda.mem_get_info()
    print(f"[test] MIG slice mem_get_info: total={total_b/1e9:.1f} GB  free={free_b/1e9:.1f} GB")

    torch.cuda.reset_peak_memory_stats()

    load_t0 = time.time()
    llm_kwargs = dict(
        model=MODEL,
        dtype=DTYPE,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
    )
    if QUANTIZATION:
        llm_kwargs["quantization"] = QUANTIZATION
    llm = LLM(**llm_kwargs)
    load_secs = time.time() - load_t0
    after_load_gb = torch.cuda.memory_allocated() / 1e9
    print(f"[test] model load {load_secs:.1f}s; alloc after load = {after_load_gb:.1f} GB")

    sampling = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=MAX_NEW_TOK)

    gen_t0 = time.time()
    # Feed everything at once; vLLM does continuous batching — set by max_num_seqs
    outputs = llm.generate(prompts, sampling)
    gen_secs = time.time() - gen_t0

    total_in  = sum(len(o.prompt_token_ids) for o in outputs)
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    req_per_s = len(prompts) / gen_secs
    tok_per_s = (total_in + total_out) / gen_secs
    peak_gb   = torch.cuda.max_memory_allocated() / 1e9

    print(f"[test] gen  {gen_secs:.1f}s  ({req_per_s:.1f} req/s, {tok_per_s:.0f} tok/s)")
    print(f"[test] prompt_tokens_total = {total_in:,}  output_tokens_total = {total_out:,}")
    print(f"[test] peak alloc = {peak_gb:.1f} GB  (of 40 GB slice; util={GPU_MEM_UTIL})")

    extrapolated_secs = 838_847 / req_per_s
    print(f"[test] 838,847 rows serial estimate = {extrapolated_secs/3600:.2f} h")
    print(f"[test] 42 shards on 16 cards (%16)   ≈ {extrapolated_secs/16/60:.0f} min wall")


if __name__ == "__main__":
    main()
