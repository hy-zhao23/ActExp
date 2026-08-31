"""
Preprocess Wikipedia (first paragraph) and peS2o (abstract) into JSONL.
Each text: complete sentences, 10–64 tokens (Qwen3-4B tokenizer).

Workflow:
  # 1. SLURM array: each rank processes its shard
  python download_pile_subsets.py --rank $RANK --world-size $N --out-dir data/raw/pretrain

  # 2. After all ranks finish: merge shards + remove HF cache
  python download_pile_subsets.py --merge --world-size $N --out-dir data/raw/pretrain

Outputs:
  data/raw/pretrain/wikipedia.jsonl   1.2M lines
  data/raw/pretrain/scientific.jsonl  0.8M lines
"""

import argparse
import json
import os
import re
import shutil

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_TOKENS = 64
MIN_TOKENS = 10


# ── text helpers ──────────────────────────────────────────────────────────────

def truncate_sentences(text: str, tok, max_tok: int) -> str | None:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    result = ""
    for s in sentences:
        c = (result + " " + s).strip() if result else s.strip()
        if len(tok.encode(c, add_special_tokens=False)) > max_tok:
            break
        result = c
    return result or None


def wiki_first_para(text: str) -> str | None:
    for p in text.split("\n\n"):
        p = p.strip()
        if p and not p.startswith("==") and len(p) >= 80:
            return p
    return None


def pes2o_abstract(text: str) -> str | None:
    # line 0 = title, line 1 = blank, line 2+ = abstract/body
    body = "\n".join(text.split("\n")[2:]).strip()
    para = body.split("\n\n")[0].strip()
    return re.sub(r'\n+', ' ', para) if para else None


# ── core writer ───────────────────────────────────────────────────────────────

def process(source_iter, text_fn, out_path: str, n_target: int, tok) -> None:
    if os.path.exists(out_path):
        print(f"[skip] {out_path}")
        return

    tmp = out_path + ".tmp"
    # 断点续传：统计已写行数，流式跳过对应数量的有效样本
    n_done = 0
    if os.path.exists(tmp):
        with open(tmp) as f:
            n_done = sum(1 for _ in f)
        print(f"[resume] {os.path.basename(tmp)}: already {n_done:,}, continuing...")

    n_saved = n_done
    n_skip  = n_done
    with open(tmp, "a" if n_done else "w") as f, \
         tqdm(total=n_target, initial=n_done, desc=os.path.basename(out_path)) as pbar:
        for ex in source_iter:
            raw = text_fn(ex)
            if not raw:
                continue
            text = truncate_sentences(raw, tok, MAX_TOKENS)
            if not text or len(tok.encode(text, add_special_tokens=False)) < MIN_TOKENS:
                continue
            if n_skip:
                n_skip -= 1
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n_saved += 1
            pbar.update(1)
            if n_saved >= n_target:
                break

    if n_saved < n_target:
        print(f"[warn] {out_path}: only {n_saved:,} / {n_target:,}")
    os.rename(tmp, out_path)


# ── merge + cleanup ───────────────────────────────────────────────────────────

def merge(out_dir: str, world_size: int) -> None:
    for name in ("wikipedia", "scientific"):
        out_path = os.path.join(out_dir, f"{name}.jsonl")
        shards = [os.path.join(out_dir, f"{name}_rank{r}.jsonl") for r in range(world_size)]
        with open(out_path, "w") as fout:
            for shard in shards:
                with open(shard) as fin:
                    shutil.copyfileobj(fin, fout)
                os.remove(shard)
        n = sum(1 for _ in open(out_path))
        print(f"[merge] {out_path}  {n:,} lines")

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for ds_dir in ("wikimedia___wikipedia", "allenai___pe_s2o"):
        path = os.path.join(hf_home, "datasets", ds_dir)
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"[cleanup] removed {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir",     default="data/raw/pretrain")
    parser.add_argument("--model-name",  default="Qwen/Qwen3-4B")
    parser.add_argument("--rank",        type=int, default=0)
    parser.add_argument("--world-size",  type=int, default=1)
    parser.add_argument("--n-wikipedia", type=int, default=1_200_000)
    parser.add_argument("--n-scientific",type=int, default=800_000)
    parser.add_argument("--merge",       action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.merge:
        merge(args.out_dir, args.world_size)
        return

    tok = AutoTokenizer.from_pretrained(args.model_name)
    n_wiki = args.n_wikipedia  // args.world_size
    n_sci  = args.n_scientific // args.world_size

    process(
        source_iter=load_dataset("wikimedia/wikipedia", "20231101.en",
                                 split="train", streaming=True
                                 ).shard(args.world_size, args.rank),
        text_fn=lambda ex: wiki_first_para(ex["text"]),
        out_path=os.path.join(args.out_dir, f"wikipedia_rank{args.rank}.jsonl"),
        n_target=n_wiki,
        tok=tok,
    )

    process(
        source_iter=load_dataset("allenai/peS2o",
                                 split="train", streaming=True, trust_remote_code=True
                                 ).shard(args.world_size, args.rank),
        text_fn=lambda ex: pes2o_abstract(ex["text"]),
        out_path=os.path.join(args.out_dir, f"scientific_rank{args.rank}.jsonl"),
        n_target=n_sci,
        tok=tok,
    )


if __name__ == "__main__":
    main()
