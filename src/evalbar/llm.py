"""Traced LLM calls through LiteLLM."""

from __future__ import annotations

import os
import time
from typing import Any

# Use the price table bundled with the installed LiteLLM instead of fetching the latest one
# at import time: runs stay reproducible, and the run file records the LiteLLM version.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402

from .trace import record_llm_call


def completion(**kwargs: Any) -> Any:
    """Call litellm.completion with the same arguments and record usage, latency and cost.

    Cost comes from LiteLLM's price table (list price). A model with no known price is
    recorded as None so reports never treat it as free.
    """
    started = time.perf_counter()
    response = litellm.completion(**kwargs)
    latency_s = time.perf_counter() - started

    usage = getattr(response, "usage", None)
    record_llm_call(
        model=str(kwargs.get("model", getattr(response, "model", "unknown"))),
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        latency_s=latency_s,
        cost_usd=_cost(response),
    )
    return response


def _cost(response: Any) -> float | None:
    # LiteLLM prices the call while it still knows the provider. completion_cost() on the
    # response alone can guess wrong: Groq returns model="openai/gpt-oss-120b" with no
    # provider hint, so the lookup goes to OpenAI and finds no price.
    hidden = getattr(response, "_hidden_params", None) or {}
    if hidden.get("response_cost") is not None:
        return float(hidden["response_cost"])
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        return None
    return float(cost)
