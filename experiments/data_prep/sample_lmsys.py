"""
Sample short user turns from LMSYS-Chat-1M into the finetune source format.

Filters (per conversation, looking at the FIRST user turn only):
  1. language == "English"
  2. openai_moderation[0].flagged == False
  3. 30 <= len(text) <= 400 chars  (≈ 8–80 tok)
  4. exact-text dedup
  5. drop turns starting with "NAME_" (PII redaction artifact at sentence head)

Output:
  data/raw/finetune/lmsys_user.jsonl  with one record per kept turn:
    {"text": <str>, "source": "lmsys", "subtype": "user",
     "meta": {"model": ..., "language": "English",
              "conv_id": ..., "turn_idx": 0, "redacted": <bool>}}

Usage:
  python experiments/data_prep/sample_lmsys.py --target 10000
"""

import argparse
import json
import os
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
OUT_PATH = PROJ / "data/raw/finetune_v2/lmsys_user.jsonl"
MIN_LEN, MAX_LEN = 30, 400


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",      type=int, default=10_000)
    ap.add_argument("--max-scanned", type=int, default=400_000,
                    help="hard cap on conversations scanned before giving up")
    ap.add_argument("--out",         type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))
    from datasets import load_dataset

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] streaming lmsys/lmsys-chat-1m ...")
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

    seen: set[int] = set()
    n_scanned = n_kept = 0
    n_lang = n_flag = n_len = n_dup = n_pii = 0

    with open(out_path, "w") as fp:
        for sample in ds:
            n_scanned += 1
            if n_scanned > args.max_scanned:
                print(f"[warn] hit max-scanned={args.max_scanned} before target")
                break

            if sample["language"] != "English":
                n_lang += 1
                continue
            if sample["openai_moderation"][0]["flagged"]:
                n_flag += 1
                continue

            text = sample["conversation"][0]["content"].strip()
            if not (MIN_LEN <= len(text) <= MAX_LEN):
                n_len += 1
                continue
            if text.startswith("NAME_"):
                n_pii += 1
                continue

            h = hash(text)
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)

            rec = {
                "text":    text,
                "source":  "lmsys",
                "subtype": "user",
                "meta": {
                    "model":     sample["model"],
                    "language":  sample["language"],
                    "conv_id":   sample["conversation_id"],
                    "turn_idx":  0,
                    "redacted":  sample["redacted"],
                },
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

            if n_kept % 1000 == 0:
                print(f"  kept={n_kept}  scanned={n_scanned}  "
                      f"(lang={n_lang} flag={n_flag} len={n_len} dup={n_dup} pii={n_pii})")

            if n_kept >= args.target:
                break

    print(f"[done] wrote {n_kept} rows → {out_path}")
    print(f"  scanned={n_scanned}  filtered: lang={n_lang} flag={n_flag} "
          f"len={n_len} dup={n_dup} pii={n_pii}")


if __name__ == "__main__":
    main()
