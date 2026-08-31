"""
Sample 4 diverse pretrain supplements for Stage 1 alignment.

Sources (target ~50k each, may underflow if upstream is small):
  c4         — allenai/c4 (English)             general web
  oasst      — OpenAssistant/oasst1 (en)        instruction dialogue (turn-level)
  hh_rlhf    — Anthropic/hh-rlhf "chosen"       single helpful turn
  pile_misc  — monology/pile-uncopyrighted      Pile-CC + StackExchange + USPTO + FreeLaw + ...
                                                (excludes wikipedia/github/arxiv/math/pubmed)

All texts are
  - sentence-truncated to ≤64 Qwen3-4B tokens, ≥10 tokens
  - drop markdown/section headers (== / # / leading <)
  - drop short ALL-CAPS lines (likely titles)
  - exact-text dedup per source

Outputs:
  data/raw/finetune_diversified/{c4,oasst,hh_rlhf,pile_misc}.jsonl

Stage 2 won't pick these up (no QA file). They're for Stage 1 only.
"""

import json
import os
import re
from pathlib import Path

PROJ      = Path(__file__).resolve().parents[2]
OUT_DIR   = PROJ / "data/raw/finetune_diversified"
PER_SRC   = 50_000
MAX_TOK   = 64
MIN_TOK   = 10
MIN_CHARS = 30


# ── filters ────────────────────────────────────────────────────────────────

_HDR_PREFIXES = ("==", "#", "<", "* ", "- ", "[")
_URL_RE = re.compile(r"https?://\S+")


def looks_like_header(text: str) -> bool:
    """Likely a title/header line — skip."""
    s = text.strip()
    if not s:
        return True
    if s.startswith(_HDR_PREFIXES):
        return True
    # short all-caps line (likely headline)
    if s.isupper() and len(s.split()) < 12:
        return True
    # too few alpha chars (likely list/code/symbols)
    n_alpha = sum(c.isalpha() for c in s)
    if n_alpha < len(s) * 0.5:
        return True
    return False


def truncate_sentences(text: str, tokenizer, max_tok: int) -> str | None:
    """Greedy sentence packing up to max_tok tokens."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = ""
    for s in sentences:
        cand = (out + " " + s).strip() if out else s.strip()
        if len(tokenizer.encode(cand, add_special_tokens=False)) > max_tok:
            break
        out = cand
    return out or None


def clean_text(raw: str) -> str:
    """Strip URLs, collapse whitespace, drop bullet/quote prefixes."""
    s = _URL_RE.sub("", raw)
    s = re.sub(r"\s+", " ", s).strip()
    # strip leading "Q:" / "A:" / "Title:" style prefixes (one line)
    s = re.sub(r"^(Q|A|Title|Subject|Re|RE|Summary|Abstract)\s*[:.]\s*", "", s)
    return s


def extract_paragraph(text: str) -> str | None:
    """First content paragraph — skip headers."""
    for p in re.split(r"\n{2,}", text):
        p = p.strip()
        if len(p) < MIN_CHARS:
            continue
        if looks_like_header(p):
            continue
        return clean_text(p)
    return None


# ── per-source streaming generators ────────────────────────────────────────

def stream_c4():
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", streaming=True, split="train",
                      trust_remote_code=False)
    for x in ds:
        p = extract_paragraph(x.get("text", ""))
        if p:
            yield p


def stream_oasst():
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    for x in ds:
        if x.get("lang") != "en":
            continue
        text = x.get("text", "")
        if not text:
            continue
        # one whole message, no need to split paragraphs
        cleaned = clean_text(text)
        if cleaned and not looks_like_header(cleaned):
            yield cleaned


def stream_hh_rlhf():
    """HH-RLHF format: '\\n\\nHuman: <q>\\n\\nAssistant: <a>\\n\\nHuman: ...'"""
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    turn_re = re.compile(r"\n\n(Human|Assistant):\s*(.*?)(?=\n\n(?:Human|Assistant):|\Z)", re.S)
    for x in ds:
        chosen = x.get("chosen", "")
        for role, content in turn_re.findall(chosen):
            cleaned = clean_text(content)
            if cleaned and not looks_like_header(cleaned):
                yield cleaned


_PILE_KEEP = {
    "Pile-CC", "StackExchange", "USPTO Backgrounds", "FreeLaw",
    "NIH ExPorter", "HackerNews", "Enron Emails", "OpenSubtitles",
    "PhilPapers", "Ubuntu IRC",
}

def stream_pile_misc():
    from datasets import load_dataset
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    for x in ds:
        meta_set = x.get("meta", {}).get("pile_set_name", "")
        if meta_set not in _PILE_KEEP:
            continue
        p = extract_paragraph(x.get("text", ""))
        if p:
            yield p


# ── main loop ──────────────────────────────────────────────────────────────

def run_source(name: str, generator, tokenizer, target: int = PER_SRC,
               max_scanned: int | None = None):
    out_path = OUT_DIR / f"{name}.jsonl"
    seen: set[int] = set()
    n_kept = n_scanned = n_short = n_long = n_dup = 0
    print(f"[{name}] streaming → {out_path}")
    with out_path.open("w") as fp:
        for text in generator:
            n_scanned += 1
            truncated = truncate_sentences(text, tokenizer, MAX_TOK)
            if truncated is None or len(truncated) < MIN_CHARS:
                n_short += 1
                continue
            n_tok = len(tokenizer.encode(truncated, add_special_tokens=False))
            if n_tok < MIN_TOK:
                n_short += 1
                continue
            if n_tok > MAX_TOK:
                n_long += 1
                continue
            h = hash(truncated)
            if h in seen:
                n_dup += 1
                continue
            seen.add(h)

            rec = {
                "text":    truncated,
                "source":  "pretrain",
                "subtype": name,
                "meta":    {},
            }
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

            if n_kept % 5000 == 0:
                print(f"  [{name}] kept={n_kept}/{target}  scanned={n_scanned}  "
                      f"short={n_short} long={n_long} dup={n_dup}")
            if n_kept >= target:
                break
            if max_scanned and n_scanned >= max_scanned:
                print(f"[{name}] hit max_scanned={max_scanned}, stopping early")
                break

    print(f"[done {name}] kept={n_kept} scanned={n_scanned} "
          f"short={n_short} long={n_long} dup={n_dup}")
    return n_kept


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(PROJ / "tmp/huggingface"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    n = {}
    n["c4"]        = run_source("c4",        stream_c4(),        tok, target=PER_SRC, max_scanned=300_000)
    n["oasst"]     = run_source("oasst",     stream_oasst(),     tok, target=PER_SRC)
    n["hh_rlhf"]   = run_source("hh_rlhf",   stream_hh_rlhf(),   tok, target=PER_SRC)
    n["pile_misc"] = run_source("pile_misc", stream_pile_misc(), tok, target=PER_SRC, max_scanned=500_000)

    print()
    print("─" * 50)
    total = sum(n.values())
    for k, v in n.items():
        print(f"  {k:<12} {v:>7,}")
    print("─" * 50)
    print(f"  {'total':<12} {total:>7,}")


if __name__ == "__main__":
    main()
