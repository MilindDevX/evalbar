"""FeedbackOS classifier: prompt v1 (single theme) vs v2 (secondary themes + mixed sentiment).

    evalbar run examples/feedbackos/eval.py:v1 --out runs/feedbackos-v1.json
    evalbar run examples/feedbackos/eval.py:v2 --out runs/feedbackos-v2.json

Needs GROQ_API_KEY. Groq's free tier allows 8K tokens/minute for this model, so cases
are spaced EVALBAR_MIN_INTERVAL seconds apart (default 9, the spacing FeedbackOS uses).
"""

import hashlib
import os
from pathlib import Path

from adapter import MODEL, OUTPUT_SCHEMA, make_classifier, system_prompt

from evalbar import Eval, Score, Scorer, load_dataset
from evalbar.scorers import contains, exact, json_schema, regex

HERE = Path(__file__).parent
DATASET = load_dataset(HERE / "cases.jsonl")
MIN_INTERVAL_S = float(os.environ.get("EVALBAR_MIN_INTERVAL", "9"))


def theme_and_sentiment(output, case):
    """Both headline labels right at once (what a reviewer needs to trust the row)."""
    hit = (output.get("theme") == case.expected["theme"]
           and output.get("sentiment") == case.expected["sentiment"])
    return Score(name="theme+sentiment", value=float(hit), passed=hit,
                 reason="" if hit else f"got {output.get('theme')}/{output.get('sentiment')}")


SCORERS = [
    exact("theme"),
    exact("sentiment"),
    exact("product_area"),
    Scorer("theme+sentiment", theme_and_sentiment, expects="theme"),
    contains("secondary_themes"),  # only the 8 mixed cases carry this expectation
    json_schema(OUTPUT_SCHEMA, name="valid_output"),
    regex("summary", r"^\S+(\s+\S+){0,14}$", name="summary<=15 words"),
]


def feedbackos_eval(version: str, source_commit: str) -> Eval:
    return Eval(
        name=f"feedbackos-{version}",
        pipeline=make_classifier(version),
        dataset=DATASET,
        scorers=SCORERS,
        min_interval_s=MIN_INTERVAL_S,
        metadata={
            "prompt": version,
            "prompt_sha256": hashlib.sha256(system_prompt(version).encode()).hexdigest()[:12],
            "prompt_source": f"feedbackos@{source_commit}:lib/ai/prompts.ts",
            "model": MODEL,
            "dataset": "examples/feedbackos/cases.jsonl (50 synthetic)",
        },
    )


v1 = feedbackos_eval("v1", "9a896d9")  # last commit before secondary themes (2026-10-11)
v2 = feedbackos_eval("v2", "59e4902")  # production prompt as of 2026-10-16 (same since fe6596c)
