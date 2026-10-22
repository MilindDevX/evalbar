"""The FeedbackOS example adapter, end to end through LiteLLM's mock_response (no network)."""

import json
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "feedbackos"
sys.path.insert(0, str(EXAMPLE))

import adapter  # noqa: E402

from evalbar import Trace  # noqa: E402


@pytest.fixture
def mock_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-not-a-real-key")
    sent = []
    real_completion = adapter.completion

    def respond_with(payload):
        def fake_completion(**kwargs):
            sent.append(kwargs)
            return real_completion(**kwargs, mock_response=json.dumps(payload))

        monkeypatch.setattr(adapter, "completion", fake_completion)

    return respond_with, sent


def label(**overrides):
    base = {"theme": "bug_problem", "sentiment": "negative", "product_area": "API", "confidence": 0.9,
            "summary": "API returns 500 on emoji.", "requires_human_review": False}
    return {**base, **overrides}


def test_sends_the_production_prompt_model_and_settings(mock_llm):
    respond_with, sent = mock_llm
    respond_with(label())

    with Trace() as trace:
        output = adapter.make_classifier("v2")({"text": "POST returns 500 when the text has an emoji."})

    (request,) = sent
    assert request["model"] == "groq/openai/gpt-oss-120b"
    assert request["temperature"] == 0 and request["max_tokens"] == 512
    assert request["response_format"] == {"type": "json_object"}
    assert request["reasoning_effort"] == "low"
    assert request["messages"][0]["content"] == adapter.system_prompt("v2")
    assert "secondary_themes" in request["messages"][0]["content"]
    assert "API, Billing" in request["messages"][1]["content"]
    assert output["secondary_themes"] == []  # zod default for v1-style output
    assert trace.llm_calls[0].cost_usd is not None


def test_v1_prompt_has_no_secondary_themes():
    assert "secondary_themes" not in adapter.system_prompt("v1")


def test_short_feedback_skips_the_llm(mock_llm):
    _, sent = mock_llm

    output = adapter.make_classifier("v2")({"text": "ok see attached"})

    assert sent == []
    assert output["requires_human_review"] is True and output["product_area"] == "Dashboard"


def test_unknown_product_area_is_an_error_like_production(mock_llm):
    respond_with, _ = mock_llm
    respond_with(label(product_area="Mobile app"))

    with pytest.raises(ValueError, match="outside the provided list"):
        adapter.make_classifier("v2")({"text": "The mobile app crashes on launch every time."})


def test_missing_key_fails_with_a_clear_message(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_API_KEY is not set"):
        adapter.make_classifier("v2")({"text": "The export button does nothing at all."})


def test_rate_limit_wait_is_read_from_the_provider_message():
    error = Exception("Rate limit reached ... Please try again in 7.25s. Need more tokens?")

    assert adapter._retry_after_seconds(error, 0) == pytest.approx(7.75)
    assert adapter._retry_after_seconds(Exception("busy"), 2) == 20.0
