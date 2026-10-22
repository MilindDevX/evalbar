"""Python port of FeedbackOS's feedback classifier call (lib/ai/classify.ts + lib/ai/prompts.ts).

Kept faithful to production on purpose: same system prompts (copied verbatim from the
FeedbackOS git history into prompts/), same user prompt, model, temperature, token cap,
JSON mode, reasoning effort, short-text guard and product-area fallback. Differences:

- Rate-limit retries: production retries through its queue; here a 429 is retried in
  process after the provider's suggested wait. Only the successful call is timed and traced.
- Output validation: production throws on schema errors; here the parsed JSON is returned
  and the json_schema scorer measures validity, so invalid outputs are counted, not hidden.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import litellm

from evalbar import completion

PROMPTS = Path(__file__).parent / "prompts"
MODEL = "groq/openai/gpt-oss-120b"  # FeedbackOS: 'openai/gpt-oss-120b' on Groq's OpenAI-compatible API
COMPLETION_TOKENS = 512  # CLASSIFICATION_COMPLETION_TOKENS
MAX_TEXT_CHARS = 8000  # MAX_CLASSIFICATION_TEXT_CHARS
PRODUCT_AREAS = ["Dashboard", "Authentication", "API", "Billing", "Onboarding", "Integrations", "Other"]
THEMES = ["bug_problem", "feature_request", "usability_issue", "question_support", "praise", "other"]

# ClassificationOutputSchema (zod) as JSON Schema. secondary_themes is optional because the
# zod schema defaults it to [] (v1 never emits it).
OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["theme", "sentiment", "product_area", "confidence", "summary", "requires_human_review"],
    "properties": {
        "theme": {"enum": THEMES},
        "secondary_themes": {"type": "array", "items": {"enum": THEMES}, "maxItems": 2},
        "sentiment": {"enum": ["positive", "negative", "neutral", "mixed"]},
        "product_area": {"type": "string", "maxLength": 100},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string", "maxLength": 200},
        "requires_human_review": {"type": "boolean"},
    },
}


def system_prompt(version: str) -> str:
    return (PROMPTS / f"{version}.txt").read_text(encoding="utf-8")


def build_user_prompt(raw_text: str, product_areas: list[str], max_chars: int = MAX_TEXT_CHARS) -> str:
    text = raw_text if len(raw_text) <= max_chars else f"{raw_text[:max_chars]}\n[Content truncated for classification]"
    return f'''AVAILABLE PRODUCT AREAS FOR THIS ORGANIZATION:
{", ".join(product_areas)}

CUSTOMER FEEDBACK TO CLASSIFY:
"""
{text}
"""

Classify this feedback accurately and completely.'''


def make_classifier(version: str, product_areas: list[str] = PRODUCT_AREAS, max_rate_limit_retries: int = 4):
    prompt = system_prompt(version)

    def classify(case_input: dict) -> dict:
        raw_text = case_input["text"]
        if len(raw_text.strip().split()) < 5:
            # Short-text guard from classifyFeedback: no LLM call.
            return {
                "theme": "other",
                "secondary_themes": [],
                "sentiment": "neutral",
                "product_area": product_areas[0] if product_areas else "Other",
                "confidence": 0.3,
                "summary": "Feedback too short to classify reliably",
                "requires_human_review": True,
            }

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set: add it to .env in the folder you run evalbar from")
        response = _call_with_rate_limit_retries(
            max_rate_limit_retries,
            model=MODEL,
            temperature=0,
            max_tokens=COMPLETION_TOKENS,
            response_format={"type": "json_object"},
            reasoning_effort="low",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": build_user_prompt(raw_text, product_areas)},
            ],
            num_retries=0,
            timeout=30,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response content")
        output = json.loads(content)
        output.setdefault("secondary_themes", [])

        fallback = next((area for area in product_areas if "other" in area.lower()), None)
        if output.get("product_area") == "Other" and fallback:
            output["product_area"] = fallback
        elif output.get("product_area") not in product_areas:
            raise ValueError(f"LLM returned product area outside the provided list: {output.get('product_area')}")
        return output

    return classify


def _call_with_rate_limit_retries(retries: int, **kwargs):
    for attempt in range(retries + 1):
        try:
            return completion(**kwargs)
        except litellm.RateLimitError as exc:
            if attempt == retries:
                raise
            time.sleep(_retry_after_seconds(exc, attempt))
    raise AssertionError("unreachable")


def _retry_after_seconds(exc: Exception, attempt: int) -> float:
    match = re.search(r"try again in ([\d.]+)(ms|s)", str(exc))
    if match:
        value = float(match.group(1))
        return (value / 1000 if match.group(2) == "ms" else value) + 0.5
    return min(60.0, 5.0 * 2**attempt)
