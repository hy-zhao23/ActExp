"""Rebuttal R4: LLM-as-judge eval — TOPIC / DETAILS, 0-5 rubric.

Scores 100 stratified UAV answers (ours_self_qwen3_4b) with three open
instruct judges of increasing size (14B / 27B / 70B, no thinking mode).
Judges see only passage + question + candidate answer (no reference).

Usage (local judges are separate GPU processes; gpt41mini is API, login node ok):
    python rebuttal/llm_judge.py --judge qwen14b [--set bykind]
    python rebuttal/llm_judge.py --judge gpt41mini --set bykind
    python rebuttal/llm_judge.py --aggregate [--set bykind]

Sample sets:
    orig    100 slots stratified over 15 datasets (v1, kept for the record)
    bykind  50 per slot_kind (gist/comp/fact) = 150, spread across datasets

Outputs under out/eval/rebuttal_llm_judge/:
    samples_100.jsonl / samples_150_bykind.jsonl   fixed sample sets (seed 0)
    scores_<judge>.jsonl / scores_bykind_<judge>.jsonl
    summary.md / summary_bykind.md
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
VARIANT = "ours_self_qwen3_4b"
EVAL_DIR = PROJ / "out" / "eval" / VARIANT
OUT_DIR = PROJ / "out" / "eval" / "rebuttal_llm_judge"
N_SAMPLES = 100
SEED = 0
MAX_PASSAGE_CHARS = 6000

JUDGES = {
    "qwen14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma27b": "google/gemma-3-27b-it",
    "glm32b": "zai-org/GLM-4-32B-0414",
    "gpt41mini": "gpt-4.1-mini",  # OpenAI API, key = $OPENAI_API_KEY_MY
}
API_JUDGES = {"gpt41mini"}
N_PER_KIND = 50  # bykind set: 50 × {gist, comp, fact}

SYSTEM_PROMPT = """\
You are a strict and careful evaluator. You will be shown a source passage, a
question about the passage, and a candidate answer. The candidate answer was
produced by a system that could not see the passage directly, so judge only
what the answer says, not its style or length.

Score the candidate answer on two independent dimensions, each on a 0-5
integer scale.

### Dimension 1: TOPIC — how close is the answer's subject to the passage's?
Judge whether the candidate answer talks about the passage's actual subject,
giving graded credit for closeness. Correctness of fine details does NOT
matter here — only whether the answer is about the relevant topic.
- 5: Exactly the passage's subject — the same event, the same person or
     organization, the same place, or the same specific theme the question
     asks about.
- 4: Not the exact subject, but immediately adjacent to it — e.g. a closely
     similar event of the same kind, a nearby location, a person working in
     the same specific field or organization, the same product category.
- 3: The same general area — e.g. the same type of event in a different
     place, the same region or country but a different site, people in the
     same broad profession, the same industry.
- 2: Only broad-domain overlap — e.g. both are about sports, politics, or
     technology, but the actual subject is different.
- 1: Barely related — connected only by isolated words or a generic theme.
- 0: Completely off topic, empty, or a refusal / non-answer.

### Dimension 2: DETAILS — how close are the concrete specifics?
Judge the concrete details in the candidate answer (names of people, places
and organizations; dates and years; numbers; events) against the passage,
giving graded credit for near-misses. The passage is the sole ground truth.
Full credit requires EXACT matches: a name counts as correct only if it is
exactly the same name as in the passage.
- 5: All checkable specifics exactly match the passage — identical names,
     identical dates/numbers; nothing fabricated.
- 4: Same entities with minor surface deviations — e.g. an abbreviated or
     partial form of the correct name (missing middle name, initials, a
     common transliteration variant), a year off by at most one, a correctly
     rounded number.
- 3: Recognizably close but not the same — e.g. a similar-sounding or
     similarly formatted name, a year within a few years of the true one, a
     number of the right magnitude; OR the answer is so generic that it
     contains almost no checkable specifics, while not contradicting the
     passage.
- 2: Specifics are wrong but of the right type and format — e.g. a plausible
     person's name where the true name is entirely different, a date in the
     right era, a place in the wrong country.
- 1: Nearly all specifics are wrong or fabricated, with only a trace of
     connection to the passage.
- 0: Entirely fabricated, or directly contradicts the passage on its main
     point.

Scoring rules:
- Judge only against the passage; do not use outside knowledge to fill gaps.
- The two dimensions are independent: an answer can be on the right topic
  with wrong details (high TOPIC, low DETAILS), or off topic while reusing
  correct names from the passage (low TOPIC, high DETAILS).
- Do not reward verbosity; a short precise answer outranks a long vague one.
- If the candidate answer is empty or unintelligible, give 0 on both.

Respond with exactly one line of JSON and nothing else:
{"topic": <0-5>, "details": <0-5>}"""

USER_TEMPLATE = """\
### Passage
{input_text}

### Question
{question}

### Candidate answer
{prediction}

Score the candidate answer now. Output only the JSON line."""

# ── vs-gt variant: judge against the reference answer, no passage shown ──
SYSTEM_PROMPT_GT = """\
You are a strict and careful evaluator. You will be shown a question, a
reference answer (the ground truth), and a candidate answer. The candidate
answer was produced by a system that could not see the original text, so
judge only what the answer says, not its style or length.

Score the candidate answer against the reference answer on two independent
dimensions, each on a 0-5 integer scale.

### Dimension 1: TOPIC — how close is the candidate's subject to the reference's?
Judge whether the candidate answer talks about the same subject matter as the
reference answer, giving graded credit for closeness. Correctness of fine
details does NOT matter here — only whether it is about the relevant topic.
- 5: Exactly the reference's subject — the same event, the same person or
     organization, the same place, or the same specific theme.
- 4: Not the exact subject, but immediately adjacent to it — e.g. a closely
     similar event of the same kind, a nearby location, a person working in
     the same specific field or organization, the same product category.
- 3: The same general area — e.g. the same type of event in a different
     place, the same region or country but a different site, people in the
     same broad profession, the same industry.
- 2: Only broad-domain overlap — e.g. both are about sports, politics, or
     technology, but the actual subject is different.
- 1: Barely related — connected only by isolated words or a generic theme.
- 0: Completely off topic, empty, or a refusal / non-answer.

### Dimension 2: DETAILS — how close are the concrete specifics?
Judge the concrete details in the candidate answer (names of people, places
and organizations; dates and years; numbers; events) against the reference
answer, giving graded credit for near-misses. The reference answer is the
sole ground truth. Full credit requires EXACT matches: a name counts as
correct only if it is exactly the same name as in the reference.
- 5: All checkable specifics exactly match the reference — identical names,
     identical dates/numbers; nothing fabricated.
- 4: Same entities with minor surface deviations — e.g. an abbreviated or
     partial form of the correct name (missing middle name, initials, a
     common transliteration variant), a year off by at most one, a correctly
     rounded number.
- 3: Recognizably close but not the same — e.g. a similar-sounding or
     similarly formatted name, a year within a few years of the true one, a
     number of the right magnitude; OR the answer is so generic that it
     contains almost no checkable specifics, while not contradicting the
     reference.
- 2: Specifics are wrong but of the right type and format — e.g. a plausible
     person's name where the true name is entirely different, a date in the
     right era, a place in the wrong country.
- 1: Nearly all specifics are wrong or fabricated, with only a trace of
     connection to the reference.
- 0: Entirely fabricated, or directly contradicts the reference on its main
     point.

Scoring rules:
- Judge only against the reference answer; do not use outside knowledge.
- The two dimensions are independent: an answer can be on the right topic
  with wrong details (high TOPIC, low DETAILS), and vice versa.
- Do not reward verbosity; a short precise answer outranks a long vague one.
- If the candidate answer is empty or unintelligible, give 0 on both.

Respond with exactly one line of JSON and nothing else:
{"topic": <0-5>, "details": <0-5>}"""

USER_TEMPLATE_GT = """\
### Question
{question}

### Reference answer
{gt}

### Candidate answer
{prediction}

Score the candidate answer now. Output only the JSON line."""


def _load_eval_rows() -> dict:
    by_ds = {}
    for f in sorted(EVAL_DIR.glob("*.jsonl")):
        if f.name == "scores.jsonl":
            continue
        by_ds[f.stem] = [json.loads(l) for l in f.open()]
    return by_ds


def _slim(ds: str, r: dict) -> dict:
    return {
        "dataset": ds, "idx": r["idx"], "slot_kind": r["slot_kind"],
        "question": r["question"], "input_text": r["input_text"],
        "prediction": r[VARIANT],
    }


def build_samples(set_name: str = "orig") -> list[dict]:
    """Fixed judged sample set (created once, seed fixed)."""
    fname = "samples_100.jsonl" if set_name == "orig" else "samples_150_bykind.jsonl"
    sample_file = OUT_DIR / fname
    if sample_file.exists():
        return [json.loads(l) for l in sample_file.open()]

    by_ds = _load_eval_rows()
    rng = np.random.default_rng(SEED)
    samples = []
    if set_name == "orig":
        datasets = sorted(by_ds)
        base, extra = divmod(N_SAMPLES, len(datasets))
        extra_ds = set(rng.choice(datasets, extra, replace=False))
        for ds in datasets:
            n = base + (ds in extra_ds)
            idxs = rng.choice(len(by_ds[ds]), n, replace=False)
            samples += [_slim(ds, by_ds[ds][int(i)]) for i in sorted(idxs)]
    elif set_name == "bykind":
        # 50 per slot_kind, spread evenly over the datasets that have it
        pools = {}  # kind -> {ds: [rows]}
        for ds, rows in by_ds.items():
            for r in rows:
                pools.setdefault(r["slot_kind"], {}).setdefault(ds, []).append(r)
        for kind in sorted(pools):
            ds_names = sorted(pools[kind])
            base, extra = divmod(N_PER_KIND, len(ds_names))
            extra_ds = set(rng.choice(ds_names, extra, replace=False))
            for ds in ds_names:
                n = base + (ds in extra_ds)
                pool = pools[kind][ds]
                idxs = rng.choice(len(pool), n, replace=False)
                samples += [_slim(ds, pool[int(i)]) for i in sorted(idxs)]
    else:
        raise ValueError(set_name)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sample_file.open("w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[samples:{set_name}] wrote {len(samples)} to {sample_file}")
    return samples


def parse_scores(text: str):
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        t, de = int(d["topic"]), int(d["details"])
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not (0 <= t <= 5 and 0 <= de <= 5):
        return None
    return t, de


def _scores_file(judge_key: str, set_name: str, vs: str = "passage") -> Path:
    prefix = "scores_" if set_name == "orig" else f"scores_{set_name}_"
    if vs == "gt":
        prefix += "gtref_"
    return OUT_DIR / f"{prefix}{judge_key}.jsonl"


def _messages(samples: list[dict], vs: str = "passage") -> list[list[dict]]:
    if vs == "gt":
        return [
            [{"role": "system", "content": SYSTEM_PROMPT_GT},
             {"role": "user", "content": USER_TEMPLATE_GT.format(
                 gt=s["gt"], question=s["question"],
                 prediction=s["prediction"])}]
            for s in samples
        ]
    return [
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.format(
             input_text=s["input_text"][:MAX_PASSAGE_CHARS],
             question=s["question"], prediction=s["prediction"])}]
        for s in samples
    ]


def _generate_vllm(model: str, messages: list) -> list[str]:
    import torch
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count(),
              max_model_len=8192, gpu_memory_utilization=0.90,
              enforce_eager=False)
    params = SamplingParams(temperature=0.0, max_tokens=24)
    return [o.outputs[0].text for o in llm.chat(messages, params)]


def _generate_openai(model: str, messages: list) -> list[str]:
    import os
    import time

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY_MY"])
    outs = []
    for i, msgs in enumerate(messages):
        for attempt in range(4):
            try:
                resp = client.chat.completions.create(
                    model=model, messages=msgs, temperature=0.0,
                    max_tokens=24)
                outs.append(resp.choices[0].message.content or "")
                break
            except Exception as e:
                if attempt == 3:
                    print(f"[openai] sample {i} failed for good: {e}")
                    outs.append("")
                else:
                    time.sleep(2 ** attempt)
        if (i + 1) % 25 == 0:
            print(f"[openai] {i + 1}/{len(messages)}")
    return outs


def run_judge(judge_key: str, set_name: str = "orig", vs: str = "passage"):
    model = JUDGES[judge_key]
    samples = build_samples(set_name)
    if vs == "gt":
        assert all(s.get("gt") for s in samples), "samples lack gt field"
    messages = _messages(samples, vs)
    if judge_key in API_JUDGES:
        raws = _generate_openai(model, messages)
    else:
        raws = _generate_vllm(model, messages)

    out_file = _scores_file(judge_key, set_name, vs)
    n_bad = 0
    with out_file.open("w") as f:
        for s, raw in zip(samples, raws):
            parsed = parse_scores(raw.strip())
            if parsed is None:
                n_bad += 1
                rec = {**{k: s[k] for k in ("dataset", "idx", "slot_kind")},
                       "judge": judge_key, "topic": None, "details": None,
                       "raw": raw.strip()}
            else:
                rec = {**{k: s[k] for k in ("dataset", "idx", "slot_kind")},
                       "judge": judge_key, "topic": parsed[0],
                       "details": parsed[1]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{judge_key}@{set_name}:{vs}] wrote {len(samples)} rows "
          f"({n_bad} unparseable) to {out_file}")


def _stat_row(label, rows):
    t = np.array([r["topic"] for r in rows])
    d = np.array([r["details"] for r in rows])
    return (f"| {label} | {len(rows)} "
            f"| {t.mean():.2f} | {(t >= 4).mean():.0%} | {(t >= 3).mean():.0%} "
            f"| {d.mean():.2f} | {(d >= 4).mean():.0%} | {(d >= 3).mean():.0%} |")


def aggregate(set_name: str = "orig", vs: str = "passage"):
    """Per-judge stats, overall + by slot_kind. Judges reported separately
    (no pooling across judges — they have different leniency calibrations)."""
    desc = (f"{N_SAMPLES} slots stratified over 15 datasets" if set_name == "orig"
            else f"{N_PER_KIND} per slot_kind (gist/comp/fact), spread over datasets")
    header = ["| judge | n | TOPIC mean | ≥4 | ≥3 | DETAILS mean | ≥4 | ≥3 |",
              "|---|---|---|---|---|---|---|---|"]
    lines = [
        f"# Rebuttal R4: LLM-judge scores (TOPIC / DETAILS, 0-5) — set `{set_name}`, vs `{vs}`",
        "",
        f"System judged: `{VARIANT}` — {desc}, seed={SEED}. " + (
            "Judges see passage+question+answer only; scored independently."
            if vs == "passage" else
            "Judges see question+reference answer+candidate only (no passage); "
            "scored independently."),
        "", "## Overall", ""] + header
    per_judge = {}
    for key in JUDGES:
        f = _scores_file(key, set_name, vs)
        if not f.exists():
            print(f"[aggregate] missing {f}, skipping")
            continue
        rows = [json.loads(l) for l in f.open()]
        ok = [r for r in rows if r["topic"] is not None]
        if len(ok) < len(rows):
            print(f"[aggregate] {key}: {len(rows) - len(ok)} unparseable dropped")
        per_judge[key] = ok
        lines.append(_stat_row(f"{key} ({JUDGES[key].split('/')[-1]})", ok))
    for kind in ("fact", "comp", "gist"):
        sub_lines = []
        for key, ok in per_judge.items():
            sub = [r for r in ok if r["slot_kind"] == kind]
            if sub:
                sub_lines.append(_stat_row(key, sub))
        if sub_lines:
            lines += ["", f"## {kind}", ""] + header + sub_lines
    lines += ["", "Score distributions (counts over valid rows, 0→5):", ""]
    for key, ok in per_judge.items():
        tc = np.bincount([r["topic"] for r in ok], minlength=6)
        dc = np.bincount([r["details"] for r in ok], minlength=6)
        lines.append(f"- {key}: TOPIC {list(tc)}  DETAILS {list(dc)}")
    tag = set_name if vs == "passage" else f"{set_name}_gtref"
    out_md = OUT_DIR / ("summary.md" if (set_name, vs) == ("orig", "passage")
                        else f"summary_{tag}.md")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out_md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--set", default="orig", choices=["orig", "bykind"])
    ap.add_argument("--vs", default="passage", choices=["passage", "gt"],
                    help="judge against the source passage or the reference answer")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.set, args.vs)
    elif args.judge:
        run_judge(args.judge, args.set, args.vs)
    else:
        build_samples(args.set)
