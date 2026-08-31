"""Stage 2 finetune data preparation — single-file pipeline.

One filter function per (dataset, subtype). Each returns a record dict or None:
    {"text": str, "source": str, "subtype": str, "meta": dict}

Entry points
------------
    python -m scripts.prepare_data --source wikipedia [--limit N]
    python -m scripts.prepare_data --all [--limit N]

Output: data/raw/finetune/{source}_{subtype}.jsonl

Sources
-------
  wikipedia  (7 subtypes: person, place, event, concept, organization, work, generic)
  scientific (1 subtype:  abstract)
  ag_news    (1 subtype:  news)
  tweeteval  (3 subtypes: sentiment, emotion, stance)
  sst2 / md_gender / ner   (classification CSVs; archived)
  latentqa   (1 subtype:  control)
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Callable, Iterator, Optional

from transformers import AutoTokenizer

# ════════════════════════════════════════════════════════════════════════════
# Paths, constants, shared utilities
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "raw"
PRETRAIN_DIR = RAW_DIR / "pretrain"
FINETUNE_OUT = RAW_DIR / "finetune"
ARCHIVE_DIR  = RAW_DIR / "finetune_v0_archive"

LATENTQA_CONTROL = (
    PROJECT_ROOT / "experiments/baseline/activation_oracles/datasets"
    / "latentqa_datasets/train/control.json"
)

MAX_TOKENS     = 64
MIN_CHARS      = 30
TOKENIZER_NAME = "Qwen/Qwen3-4B"


_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _tok


def token_count(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


_ABBREV = {
    "no", "u.s", "u.k", "e.g", "i.e", "dr", "mr", "mrs", "ms", "st",
    "jr", "sr", "inc", "ltd", "co", "corp", "prof", "vs", "etc",
    "ad", "bc", "b.c", "a.d", "ph.d", "m.d",
}


def first_sentence(text: str) -> str:
    """Lead sentence for length truncation; avoids breaking on abbreviation dots."""
    text = text.strip()
    for m in re.finditer(r"([.!?])(?=\s+[A-Z0-9])", text):
        end = m.start()
        prev = text[:end].split()[-1] if end > 0 else ""
        if prev.lower().rstrip(".") in _ABBREV:
            continue
        return text[:end + 1].strip()
    return text.split("\n\n", 1)[0][:500].strip()


def lead_window(text: str, max_chars: int = 800) -> str:
    """Wider window for classification parsing (not length truncation)."""
    return text.split("\n\n", 1)[0][:max_chars].strip()


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def open_jsonl_writers(base: Path, names: list) -> dict:
    base.mkdir(parents=True, exist_ok=True)
    return {n: (base / f"{n}.jsonl").open("w") for n in names}


def write_record(fp, rec: dict) -> None:
    fp.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_record(text: str, source: str, subtype: str,
                meta: Optional[dict] = None) -> Optional[dict]:
    """Length-validate + assemble record. Returns None if filtered out."""
    text = text.strip()
    if len(text) < MIN_CHARS:
        return None
    if token_count(text) > MAX_TOKENS:
        trunc = first_sentence(text)
        if token_count(trunc) > MAX_TOKENS or len(trunc) < MIN_CHARS:
            return None
        text = trunc
    return {"text": text, "source": source, "subtype": subtype, "meta": meta or {}}


# ════════════════════════════════════════════════════════════════════════════
# Wikipedia — subtype classifier + 7 filter functions
# ════════════════════════════════════════════════════════════════════════════

PERSON_NOUNS = {
    "physicist", "chemist", "biologist", "mathematician", "scientist", "researcher",
    "writer", "novelist", "poet", "author", "journalist", "editor",
    "politician", "statesman", "diplomat", "president", "king", "queen", "emperor",
    "prime", "minister", "senator", "governor", "ambassador",
    "actor", "actress", "singer", "musician", "composer", "artist", "painter",
    "director", "filmmaker", "producer", "screenwriter",
    "engineer", "architect", "philosopher", "theologian",
    "general", "soldier", "admiral", "warrior", "commander",
    "athlete", "player", "coach", "boxer", "footballer",
    "businessman", "entrepreneur", "economist", "banker",
    "professor", "teacher", "scholar", "historian", "linguist",
    "doctor", "surgeon", "psychologist", "psychiatrist",
    "lawyer", "judge", "attorney",
    "explorer", "astronaut", "inventor",
    "priest", "monk", "bishop", "saint",
}

PLACE_NOUNS = {
    "city", "town", "village", "capital", "municipality", "borough", "commune",
    "country", "nation", "state", "province", "region", "territory", "district",
    "continent",
    "river", "stream", "lake", "sea", "ocean", "bay", "strait", "gulf",
    "mountain", "peak", "hill", "valley", "plateau", "ridge",
    "island", "peninsula", "archipelago", "atoll",
    "forest", "desert", "park", "reserve", "wilderness",
    "neighborhood", "suburb", "locality",
}

EVENT_NOUNS = {
    "war", "battle", "revolution", "rebellion", "uprising", "insurrection", "coup",
    "treaty", "agreement", "accord", "protocol", "pact",
    "pandemic", "epidemic", "outbreak",
    "crisis", "recession", "depression",
    "election", "referendum", "convention", "summit",
    "conference", "meeting", "congress",
    "festival", "tournament", "championship", "olympics", "games",
    "earthquake", "hurricane", "flood", "tsunami", "storm", "eruption",
    "massacre", "genocide", "disaster", "incident", "attack",
}

CONCEPT_NOUNS = {
    "theory", "concept", "principle", "law", "doctrine", "hypothesis",
    "method", "technique", "approach", "system", "framework", "algorithm",
    "philosophy", "ideology", "movement",
    "phenomenon", "process", "procedure", "reaction",
    "genre", "style", "form", "field", "branch", "discipline",
    "unit", "measurement", "quantity", "constant",
    "symptom", "condition", "syndrome", "disorder", "illness", "disease",
    "element", "compound", "substance", "material", "molecule", "mineral",
    "species", "genus", "family", "breed", "animal", "organism", "plant",
    "language", "alphabet", "script", "dialect",
    "standard", "protocol", "format",
    "deity", "deities", "god", "goddess", "spirit",
    "function", "operation", "equation", "formula",
    "food", "dish", "beverage",
}

ORG_NOUNS = {
    "company", "corporation", "firm", "business", "enterprise", "conglomerate",
    "organization", "organisation", "institution", "foundation", "association",
    "agency", "bureau", "department", "ministry", "commission",
    "university", "college", "school", "academy", "institute",
    "party", "coalition", "federation",
    "church", "temple", "monastery",
    "team", "club", "league",
    "bank", "newspaper", "magazine", "network", "broadcaster",
}

WORK_NOUNS = {
    "novel", "book", "poem", "play", "essay", "manuscript",
    "film", "movie", "documentary", "series",
    "album", "song", "symphony", "opera", "musical", "soundtrack",
    "painting", "sculpture", "artwork",
    "game",
    "journal",
}

_WIKI_SUBTYPE_ORDER = [
    ("person",       PERSON_NOUNS),
    ("work",         WORK_NOUNS),
    ("place",        PLACE_NOUNS),
    ("event",        EVENT_NOUNS),
    ("organization", ORG_NOUNS),
    ("concept",      CONCEPT_NOUNS),
]

_PARENS_RE    = re.compile(r"\([^)]*\)")
_LEAD_VERB_RE = re.compile(
    r"(?P<subject>.{1,200}?)\s+\b(?:is|was|were|are)\b\s+"
    r"(?:(?:a|an|the|one of|one of the)\s+)?"
    r"(?P<descriptor>.{3,250})",
    re.IGNORECASE,
)


def _wiki_parse_lead(text: str):
    s = lead_window(text, max_chars=800)
    s = _PARENS_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    m = _LEAD_VERB_RE.match(s)
    if m is None:
        return None
    return m.group("subject").strip(" ,;"), m.group("descriptor").lower()


def classify_wikipedia(text: str) -> Optional[str]:
    """Return subtype name, or None if unclassifiable (→ 'generic')."""
    parsed = _wiki_parse_lead(text)
    if parsed is None:
        return None
    _, descriptor = parsed
    words = set(re.findall(r"[a-z]+", descriptor[:250]))
    for subtype, nouns in _WIKI_SUBTYPE_ORDER:
        if words & nouns:
            return subtype
    return None


def _wiki_with_entity(text: str, subtype: str) -> Optional[dict]:
    rec = make_record(text, "wikipedia", subtype)
    if rec is None:
        return None
    parsed = _wiki_parse_lead(text)
    if parsed is not None:
        rec["meta"]["entity"] = parsed[0]
    return rec


def filter_wikipedia_person      (text): return _wiki_with_entity(text, "person")
def filter_wikipedia_place       (text): return _wiki_with_entity(text, "place")
def filter_wikipedia_event       (text): return _wiki_with_entity(text, "event")
def filter_wikipedia_concept     (text): return _wiki_with_entity(text, "concept")
def filter_wikipedia_organization(text): return _wiki_with_entity(text, "organization")
def filter_wikipedia_work        (text): return _wiki_with_entity(text, "work")
def filter_wikipedia_generic     (text): return make_record(text, "wikipedia", "generic")


WIKI_FILTERS = {
    "person":       filter_wikipedia_person,
    "place":        filter_wikipedia_place,
    "event":        filter_wikipedia_event,
    "concept":      filter_wikipedia_concept,
    "organization": filter_wikipedia_organization,
    "work":         filter_wikipedia_work,
    "generic":      filter_wikipedia_generic,
}


# ════════════════════════════════════════════════════════════════════════════
# Scientific — field classifier + objective extraction
# ════════════════════════════════════════════════════════════════════════════

_FIELD_KEYWORDS = {
    "medicine": {"patient", "clinical", "treatment", "disease", "hospital", "diagnosis",
                 "surgery", "therapy", "tumor", "tumour", "cancer", "pharmac", "drug",
                 "medic", "symptom", "syndrome", "icu"},
    "biology":  {"gene", "protein", "cell", "molecular", "bacteria", "species",
                 "enzyme", "dna", "rna", "microbio", "genomic", "phylogen"},
    "physics":  {"quantum", "particle", "relativity", "cosmolog", "electromagnet",
                 "thermo", "photon", "fermion", "boson", "gravit"},
    "chemistry":{"chemical", "reaction", "compound", "synthes", "catalyst", "polymer",
                 "oxidation", "molecule"},
    "math":     {"theorem", "equation", "algebra", "topolog", "manifold", "inequality",
                 "polynomial", "conjecture", "lemma"},
    "cs":       {"algorithm", "computation", "network", "dataset", "neural", "machine",
                 "software", "database", "classifier"},
    "social":   {"social", "economic", "policy", "education", "behavior", "survey",
                 "population", "gender", "political"},
}


def _sci_field(text: str) -> str:
    lower = text.lower()[:600]
    words = set(re.findall(r"[a-z]+", lower))
    for field, kws in _FIELD_KEYWORDS.items():
        if any(kw in words or kw in lower for kw in kws):
            return field
    return "other"


_OBJECTIVE_RES = [
    re.compile(r"(?:OBJECTIVE|PURPOSE|AIM|GOAL)S?\s+(?:was\s+)?(?:to\s+)?([^.]{10,200}\.)", re.IGNORECASE),
    re.compile(r"\bwe\s+(?:investigate|examine|evaluate|propose|present|study|analyze)[^.]{5,200}\.", re.IGNORECASE),
    re.compile(r"\bthis\s+(?:study|paper|work|article|research)\s+(?:investigates|examines|evaluates|proposes|presents|explores|analyzes)[^.]{5,200}\.", re.IGNORECASE),
]


def _sci_objective(text: str) -> Optional[str]:
    window = lead_window(text, max_chars=600)
    for pat in _OBJECTIVE_RES:
        m = pat.search(window)
        if m:
            obj = m.group(0).strip() if m.lastindex is None or m.lastindex < 1 else m.group(1).strip()
            if 20 <= len(obj) <= 250:
                return obj
    return None


def filter_scientific(text: str) -> Optional[dict]:
    rec = make_record(text, "scientific", "abstract")
    if rec is None:
        return None
    rec["meta"]["field"] = _sci_field(text)
    obj = _sci_objective(text)
    if obj:
        rec["meta"]["objective"] = obj
    return rec


# ════════════════════════════════════════════════════════════════════════════
# AG News
# ════════════════════════════════════════════════════════════════════════════

_AGNEWS_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Tech"}


def filter_ag_news(item: dict) -> Optional[dict]:
    text = item["text"].strip().replace("\\", "")
    rec = make_record(text, "ag_news", "news")
    if rec is None:
        return None
    rec["meta"]["category"] = _AGNEWS_MAP[item["label"]]
    return rec


# ════════════════════════════════════════════════════════════════════════════
# TweetEval — sentiment / emotion / stance
# ════════════════════════════════════════════════════════════════════════════

_SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}
_EMOTION_MAP   = {0: "anger", 1: "joy", 2: "optimism", 3: "sadness"}
_STANCE_MAP    = {0: "none", 1: "against", 2: "favor"}

_USER_RE = re.compile(r"@\w+")
_URL_RE  = re.compile(r"https?://\S+")
_HASH_RE = re.compile(r"#(\w+)")


def _clean_tweet(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _USER_RE.sub("@user", text)
    text = _HASH_RE.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def filter_tweeteval_sentiment(item: dict) -> Optional[dict]:
    rec = make_record(_clean_tweet(item["text"]), "tweeteval", "sentiment")
    if rec is None:
        return None
    rec["meta"]["sentiment"] = _SENTIMENT_MAP[item["label"]]
    return rec


def filter_tweeteval_emotion(item: dict) -> Optional[dict]:
    rec = make_record(_clean_tweet(item["text"]), "tweeteval", "emotion")
    if rec is None:
        return None
    rec["meta"]["emotion"] = _EMOTION_MAP[item["label"]]
    return rec


def filter_tweeteval_stance(item: dict, target: str) -> Optional[dict]:
    rec = make_record(_clean_tweet(item["text"]), "tweeteval", "stance")
    if rec is None:
        return None
    rec["meta"]["stance"] = _STANCE_MAP[item["label"]]
    rec["meta"]["target"] = target
    return rec


# ════════════════════════════════════════════════════════════════════════════
# CLS datasets — sst2 / md_gender / ner (archived CSVs)
# ════════════════════════════════════════════════════════════════════════════

def filter_sst2(row: dict) -> Optional[dict]:
    rec = make_record(row["statement"].strip(), "sst2", "review")
    if rec is None:
        return None
    rec["meta"]["sentiment"] = "positive" if row["label"] == "1" else "negative"
    return rec


def filter_md_gender(row: dict) -> Optional[dict]:
    rec = make_record(row["statement"].strip(), "md_gender", "scene")
    if rec is None:
        return None
    rec["meta"]["gender"] = "female" if row["label"] == "1" else "male"
    return rec


_NER_RE = re.compile(r"^(?P<text>.+?)\s*Named entity:\s*(?P<entity>.+?)$", re.DOTALL)


def filter_ner(row: dict) -> Optional[dict]:
    if row["label"] != "1":                      # only keep correctly-labelled entities
        return None
    m = _NER_RE.match(row["statement"].strip())
    if m is None:
        return None
    rec = make_record(m.group("text").strip(), "ner", "entity")
    if rec is None:
        return None
    rec["meta"]["named_entity"] = m.group("entity").strip()
    return rec


# ════════════════════════════════════════════════════════════════════════════
# LatentQA control
# ════════════════════════════════════════════════════════════════════════════

def filter_latentqa_control(item: dict) -> Optional[dict]:
    rec = make_record(item["control_user"].strip(), "latentqa", "control")
    if rec is None:
        return None
    label = (item.get("label") or "").strip()
    if label:
        rec["meta"]["category"] = label
        if "-" in label:
            rec["meta"]["domain"] = label.split("-", 1)[0]
    return rec


# ════════════════════════════════════════════════════════════════════════════
# Per-source runners
# ════════════════════════════════════════════════════════════════════════════

def run_wikipedia(limit=None) -> dict:
    """Each record gets meta['orig_idx'] = line index in pretrain/wikipedia.jsonl,
    so downstream can match Stage 1's seeded train/test split."""
    src = PRETRAIN_DIR / "wikipedia.jsonl"
    names = [f"wikipedia_{k}" for k in WIKI_FILTERS]
    writers = open_jsonl_writers(FINETUNE_OUT, names)
    counts = {k: 0 for k in WIKI_FILTERS}
    try:
        for i, item in enumerate(read_jsonl(src)):
            if limit is not None and i >= limit:
                break
            text = item["text"]
            subtype = classify_wikipedia(text) or "generic"
            rec = WIKI_FILTERS[subtype](text)
            if rec is None:
                continue
            rec["meta"]["orig_idx"] = i                    # ← Stage 1 split matching
            write_record(writers[f"wikipedia_{subtype}"], rec)
            counts[subtype] += 1
    finally:
        for fp in writers.values(): fp.close()
    return counts


def run_scientific(limit=None) -> dict:
    src = PRETRAIN_DIR / "scientific.jsonl"
    writers = open_jsonl_writers(FINETUNE_OUT, ["scientific"])
    counts = {"scientific": 0, "with_objective": 0}
    try:
        for i, item in enumerate(read_jsonl(src)):
            if limit is not None and i >= limit:
                break
            rec = filter_scientific(item["text"])
            if rec is None:
                continue
            rec["meta"]["orig_idx"] = i                    # ← Stage 1 split matching
            write_record(writers["scientific"], rec)
            counts["scientific"] += 1
            if rec["meta"].get("objective"):
                counts["with_objective"] += 1
    finally:
        for fp in writers.values(): fp.close()
    return counts


def run_ag_news(limit=None) -> dict:
    from datasets import load_dataset
    writers = open_jsonl_writers(FINETUNE_OUT, ["ag_news"])
    counts = {"ag_news": 0}
    try:
        ds = load_dataset("fancyzhx/ag_news", split="train")
        for i, item in enumerate(ds):
            if limit is not None and i >= limit:
                break
            rec = filter_ag_news(item)
            if rec is None: continue
            write_record(writers["ag_news"], rec)
            counts["ag_news"] += 1
    finally:
        for fp in writers.values(): fp.close()
    return counts


def run_tweeteval(limit=None) -> dict:
    from datasets import load_dataset
    names = ["tweeteval_sentiment", "tweeteval_emotion", "tweeteval_stance"]
    writers = open_jsonl_writers(FINETUNE_OUT, names)
    counts = {n: 0 for n in names}
    stance_subsets = ["stance_abortion", "stance_atheism", "stance_climate",
                      "stance_feminist", "stance_hillary"]
    try:
        for item in load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train"):
            if limit is not None and counts["tweeteval_sentiment"] >= limit: break
            rec = filter_tweeteval_sentiment(item)
            if rec is None: continue
            write_record(writers["tweeteval_sentiment"], rec)
            counts["tweeteval_sentiment"] += 1

        for item in load_dataset("cardiffnlp/tweet_eval", "emotion", split="train"):
            if limit is not None and counts["tweeteval_emotion"] >= limit: break
            rec = filter_tweeteval_emotion(item)
            if rec is None: continue
            write_record(writers["tweeteval_emotion"], rec)
            counts["tweeteval_emotion"] += 1

        for subset in stance_subsets:
            target = subset.replace("stance_", "")
            for item in load_dataset("cardiffnlp/tweet_eval", subset, split="train"):
                if limit is not None and counts["tweeteval_stance"] >= limit: break
                rec = filter_tweeteval_stance(item, target)
                if rec is None: continue
                write_record(writers["tweeteval_stance"], rec)
                counts["tweeteval_stance"] += 1
    finally:
        for fp in writers.values(): fp.close()
    return counts


def _run_csv(fname: str, source: str, filter_fn: Callable, limit=None) -> int:
    path = ARCHIVE_DIR / fname
    if not path.exists():
        print(f"[warn] missing: {path}")
        return 0
    fp = (FINETUNE_OUT / f"{source}.jsonl").open("w")
    FINETUNE_OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    try:
        with path.open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                if limit is not None and i >= limit:
                    break
                rec = filter_fn(row)
                if rec is None: continue
                write_record(fp, rec)
                n += 1
    finally:
        fp.close()
    return n


def run_sst2     (limit=None): return {"sst2":      _run_csv("sst2_true_false_train.csv",      "sst2",      filter_sst2,      limit)}
def run_md_gender(limit=None): return {"md_gender": _run_csv("md_gender_true_false_train.csv", "md_gender", filter_md_gender, limit)}
def run_ner      (limit=None): return {"ner":       _run_csv("ner_true_false_train.csv",       "ner",       filter_ner,       limit)}


def run_latentqa(limit=None) -> dict:
    writers = open_jsonl_writers(FINETUNE_OUT, ["latentqa_control"])
    counts = {"latentqa_control": 0}
    try:
        for i, item in enumerate(json.loads(LATENTQA_CONTROL.read_text())):
            if limit is not None and i >= limit:
                break
            rec = filter_latentqa_control(item)
            if rec is None: continue
            write_record(writers["latentqa_control"], rec)
            counts["latentqa_control"] += 1
    finally:
        for fp in writers.values(): fp.close()
    return counts


# ════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ════════════════════════════════════════════════════════════════════════════

RUNNERS = {
    "wikipedia":  run_wikipedia,
    "scientific": run_scientific,
    "ag_news":    run_ag_news,
    "tweeteval":  run_tweeteval,
    "sst2":       run_sst2,
    "md_gender":  run_md_gender,
    "ner":        run_ner,
    "latentqa":   run_latentqa,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=list(RUNNERS) + ["all"], required=True)
    p.add_argument("--limit", type=int, default=None,
                   help="cap samples per source (for testing)")
    args = p.parse_args()

    targets = list(RUNNERS) if args.source == "all" else [args.source]
    for name in targets:
        print(f"[{name}] start")
        counts = RUNNERS[name](limit=args.limit)
        print(f"[{name}] {counts}")


if __name__ == "__main__":
    main()
