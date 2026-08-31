"""
Per-subtype QA schemas for Stage 2 training.

Two modes
---------
  "factual"       — Wikipedia 7 subtypes. LLM generates 0-4 factual Q-A pairs
                    grounded in the text. No gist. Purpose: test adapter's
                    ability to preserve specific facts.

  "comprehension" — 9 other subtypes. LLM generates 1 gist answer + 1-3
                    comprehension Q-A pairs. For subtypes with metadata
                    labels (sst2, ner, ag_news, ...), the label is given as
                    a focus hint only — the answer must describe content in
                    distinctive phrases from the text, NEVER outputting the
                    categorical label word itself.

Gist questions are domain-agnostic (shared GIST_POOL). At training time we
randomly sample one phrasing and pair it with the LLM-generated gist answer.
"""

# ── 51 domain-agnostic gist phrasings ───────────────────────────────────────

GIST_POOL = [
    "Summarize this text in one sentence.",
    "Paraphrase this text briefly.",
    "Describe what this text is about.",
    "What topic does this text cover?",
    "What is the main subject of this text?",
    "What is this text discussing?",
    "What does this text convey?",
    "Provide a brief description of this content.",
    "What is the main idea of this text?",
    "State the central idea of this text briefly.",
    "Give a short summary of this text.",
    "Write a concise summary of this passage.",
    "Explain this text in a brief way.",
    "What is this passage mainly about?",
    "What is the core message of this text?",
    "What is the key point of this content?",
    "What information does this text present?",
    "What does this passage describe?",
    "What is being talked about in this text?",
    "What is the focus of this text?",
    "Identify the main point of this passage.",
    "State the gist of this text.",
    "What is the overall meaning of this text?",
    "What is this content mainly conveying?",
    "Give the main takeaway from this text.",
    "What is the central theme of this passage?",
    "Describe the main content of this text briefly.",
    "What is the primary idea expressed here?",
    "What does this passage mainly convey?",
    "Summarize the main point of this content.",
    "What is the text trying to say overall?",
    "Briefly explain what this text discusses.",
    "What subject matter does this passage address?",
    "What is the text mainly focused on?",
    "What broad idea does this text express?",
    "What is the essential meaning of this passage?",
    "What concept is this text mainly about?",
    "What issue or topic is this text addressing?",
    "Provide the gist of this passage.",
    "Briefly state what this text is about.",
    "What does this text mainly describe?",
    "What is the primary focus of this passage?",
    "What message is this text communicating?",
    "What is the general idea presented here?",
    "How would you briefly describe this text?",
    "What is this passage trying to explain?",
    "What is the text communicating in general?",
    "What is the main point being made here?",
    "What is this content about in brief?",
    "What does this passage mean overall?",
]


# ── per-subtype schema ───────────────────────────────────────────────────────

QA_SCHEMAS: dict[str, dict] = {

    # Factual mode (Wikipedia) ─────────────────────────────────────────────

    "wikipedia_person": {
        "mode":      "factual",
        "fact_hint": "biographical facts — nationality, profession, dates, achievements, affiliations, relationships",
    },
    "wikipedia_place": {
        "mode":      "factual",
        "fact_hint": "geographic facts — location, country, size, population, landmarks, notable features",
    },
    "wikipedia_event": {
        "mode":      "factual",
        "fact_hint": "event details — dates, participants, locations, causes, outcomes, scale",
    },
    "wikipedia_concept": {
        "mode":      "factual",
        "fact_hint": "concept attributes — origin, components, examples, applications, defining properties",
    },
    "wikipedia_organization": {
        "mode":      "factual",
        "fact_hint": "organizational facts — founding date, location, purpose, leadership, size, activities",
    },
    "wikipedia_work": {
        "mode":      "factual",
        "fact_hint": "production/release facts — creator, year, genre, cast, setting, reception, awards",
    },
    "wikipedia_generic": {
        "mode":      "factual",
        "fact_hint": "any specific facts — numbers, names, dates, places, relations",
    },

    # Comprehension mode (all other sources) ──────────────────────────────

    "scientific": {
        "mode":        "comprehension",
        "extra_hints": "research field, general methodology, overall approach",
    },

    "ag_news": {
        "mode":        "comprehension",
        "extra_hints": "the overall event, main actors, general context",
        "label_field": "category",
        "focus_dim":   "the subject area or field of the news reported",
    },

    "tweeteval_sentiment": {
        "mode":        "comprehension",
        "extra_hints": "what is being talked about, overall tone",
        "label_field": "sentiment",
        "focus_dim":   "the speaker's attitude or evaluative tone",
    },
    "tweeteval_emotion": {
        "mode":        "comprehension",
        "extra_hints": "trigger or object of the feeling, overall mood",
        "label_field": "emotion",
        "focus_dim":   "the emotional state expressed by the speaker",
    },
    "tweeteval_stance": {
        "mode":        "comprehension",
        "extra_hints": "subject of debate, what the speaker argues",
        "label_field": "stance",
        "focus_dim":   "the position taken on the subject",
    },

    "sst2": {
        "mode":        "comprehension",
        "extra_hints": "aspects praised or criticized, overall verdict",
        "label_field": "sentiment",
        "focus_dim":   "the overall evaluation or attitude expressed toward the subject",
    },
    "md_gender": {
        "mode":        "comprehension",
        "extra_hints": "context, actions, setting",
        "label_field": "gender",
        "focus_dim":   "the subject being depicted in the scene",
    },
    "ner": {
        "mode":        "comprehension",
        "extra_hints": "context in which the entity appears, surrounding events",
        "label_field": "named_entity",
        "focus_dim":   "the specific named entity mentioned in the text",
    },

    "latentqa_control": {
        "mode":        "comprehension",
        "extra_hints": "behavior described, purpose, method implied",
        "label_field": "category",
        "focus_dim":   "the skill, activity, or behavior being instructed",
    },
}


# ── prompts ──────────────────────────────────────────────────────────────────

FACTUAL_PROMPT = """You are generating training data for a language model. Read the text and produce factual question-answer pairs grounded STRICTLY in the text — do not invent or paraphrase facts beyond what is stated.

TEXT:
{text}

Generate up to 4 factual Q-A pairs. Focus on: {fact_hint}.

Rules:
- Include a pair ONLY when both the question and answer are directly supported by the text.
- Each Q should ask about one specific fact (a date, name, place, number, relation, etc.).
- If the text has abundant facts, produce up to 4 pairs. If sparse, produce fewer (even 1 is OK).
- Do NOT use outside knowledge. Do NOT infer or speculate.

Output strictly as valid JSON (no extra text):
{{
  "factual": [
    {{"q": "<question>", "a": "<answer>"}},
    ...
  ]
}}
"""


COMPREHENSION_PROMPT = """You are generating training data for a language model. Read the text and produce comprehension-style QA about the OVERALL CONTENT (what it's about, tone, context, intent) — not specific isolated facts.

TEXT:
{text}
{focus_block}
Produce:

A) "gist": ONE sentence that summarises or paraphrases what the text is about. Do NOT begin with phrases like "This text ..." or "The article ..."; write it as a standalone summary.

B) "comprehension": 1-3 Q-A pairs about themes, tone, intent, or general context. Focus on: {extra_hints}.

Rules — read carefully:
- All answers MUST be grounded in the text, using distinctive phrases from the text itself.
- Do NOT use outside knowledge or infer beyond the text.
{label_rule}
GOOD answer examples (content-grounded, uses text phrases):
  "A young woman sitting before a mirror"
  "Praised as a masterpiece of human interaction"
  "Strategies being monitored and modified"

BAD answer examples (categorical labels — DO NOT DO):
  "The sentiment is positive."
  "This is a metacognition task."
  "This is World news."

Output strictly as valid JSON (no extra text):
{{
  "gist": "<one-sentence paraphrase>",
  "comprehension": [
    {{"q": "<question>", "a": "<answer grounded in specific text phrases>"}},
    ...
  ]
}}
"""


def build_prompt(subtype: str, text: str, meta: dict | None = None) -> str:
    schema = QA_SCHEMAS[subtype]
    if schema["mode"] == "factual":
        return FACTUAL_PROMPT.format(text=text, fact_hint=schema["fact_hint"])

    # comprehension — optionally inject focus_dim + label hint
    focus_block, label_rule = "", ""
    lf = schema.get("label_field")
    focus_dim = schema.get("focus_dim")
    if lf and focus_dim and meta and meta.get(lf):
        lbl_val = meta[lf]
        focus_block = (
            f'\nFOCUS HINT: The key dimension of interest is {focus_dim}.\n'
            f'A reference label from the dataset is "{lbl_val}" — given only as guidance.\n'
        )
        label_rule = (
            f'- At least ONE of your Q-A pairs must address this key dimension.\n'
            f'- NEVER output the label word "{lbl_val}" (or close synonyms) directly in any answer. '
            f'Describe it through content using distinctive phrases from the text.\n'
        )

    return COMPREHENSION_PROMPT.format(
        text=text,
        focus_block=focus_block,
        extra_hints=schema["extra_hints"],
        label_rule=label_rule,
    )


def subtype_from_record(rec: dict) -> str:
    src = rec["source"]
    sub = rec["subtype"]
    return f"wikipedia_{sub}" if src == "wikipedia" else src
