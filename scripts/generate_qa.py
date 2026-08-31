"""
Generate per-record QA pairs using Qwen3-14B via vLLM.

Shard-based submission:
    python -m scripts.generate_qa --shard_id 0          # single shard
    python -m scripts.generate_qa --shard_id $SLURM_ARRAY_TASK_ID

Inputs
------
    data/raw/finetune/{subtype}.jsonl                   # text records
    data/raw/finetune_qa_raw/shards.jsonl               # shard manifest

Outputs
-------
    data/raw/finetune_qa_raw/{subtype}.s{start:06d}-{end:06d}.jsonl

Each output line (raw, source of truth — every request writes one, regardless
of whether parsing later succeeds):
    {"text_idx": int, "mode": "factual"|"comprehension", "raw": str}

Parsing into {gist, comprehension, factual} happens offline in parse_qa.py.

Resume
------
On restart, reads existing raw file, skips text_idx already written.
Flush every FLUSH_EVERY lines — worst case loses <FLUSH_EVERY rows on crash.

Qwen3 thinking mode must be DISABLED (otherwise outputs get wrapped in
<think>...</think> and eat max_new_tokens before reaching the JSON).
We apply chat template with enable_thinking=False.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scripts.qa_schemas import QA_SCHEMAS, build_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune"
OUT_DIR      = PROJECT_ROOT / "data" / "raw" / "finetune_qa_raw"
MANIFEST     = OUT_DIR / "shards.jsonl"

MODEL_NAME      = "Qwen/Qwen3-14B"
MAX_NEW_TOKENS  = 300
MAX_MODEL_LEN   = 1024
BATCH_SIZE      = 256            # feed vLLM in chunks; continuous batching inside
FLUSH_EVERY     = 100
GPU_MEM_UTIL    = 0.90


def load_shard(shard_id: int) -> dict:
    with MANIFEST.open() as f:
        for line in f:
            s = json.loads(line)
            if s["shard_id"] == shard_id:
                return s
    raise ValueError(f"shard_id={shard_id} not found in {MANIFEST}")


def load_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open() as f:
        for line in f:
            done.add(json.loads(line)["text_idx"])
    return done


def build_chat(tok, subtype: str, rec: dict) -> str:
    user_msg = build_prompt(subtype, rec["text"], rec.get("meta"))
    return tok.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def run_shard(shard: dict, model: str, batch_size: int, max_new_tokens: int):
    subtype = shard["subtype"]
    start   = shard["start"]
    end     = shard["end"]

    if subtype not in QA_SCHEMAS:
        raise KeyError(f"no QA_SCHEMA for subtype={subtype}")
    mode = QA_SCHEMAS[subtype]["mode"]

    src_path = RAW_DIR / f"{subtype}.jsonl"
    out_path = OUT_DIR / f"{subtype}.s{start:06d}-{end:06d}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_path)
    print(f"[shard {shard['shard_id']}] {subtype} [{start}:{end}] mode={mode} resume={len(done):,}")

    tok = AutoTokenizer.from_pretrained(model)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=model,
        dtype="bfloat16",
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
    )
    sampling = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=max_new_tokens)

    out_fp = out_path.open("a")
    batch_prompts: list[str] = []
    batch_idxs:    list[int] = []
    written = 0

    def flush():
        nonlocal written
        if not batch_prompts:
            return
        outputs = llm.generate(batch_prompts, sampling)
        for idx, out in zip(batch_idxs, outputs):
            raw = out.outputs[0].text
            rec = {"text_idx": idx, "mode": mode, "raw": raw}
            out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written % FLUSH_EVERY == 0:
                out_fp.flush()
        batch_prompts.clear()
        batch_idxs.clear()

    with src_path.open() as f:
        for i, line in enumerate(f):
            if i < start or i >= end:
                continue
            if i in done:
                continue
            rec = json.loads(line)
            batch_prompts.append(build_chat(tok, subtype, rec))
            batch_idxs.append(i)
            if len(batch_prompts) >= batch_size:
                flush()

    flush()
    out_fp.close()
    total_in_file = len(done) + written
    print(f"[shard {shard['shard_id']}] done  written={written:,}  total_in_file={total_in_file:,}/{end-start:,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id",       type=int, required=True)
    ap.add_argument("--model",          default=MODEL_NAME)
    ap.add_argument("--batch",          type=int, default=BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args()

    shard = load_shard(args.shard_id)
    run_shard(shard, args.model, args.batch, args.max_new_tokens)


if __name__ == "__main__":
    main()
