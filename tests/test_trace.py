import pytest

from evalbar.llm import completion
from evalbar.trace import Trace, current_trace, record_llm_call, traced_tool


def test_records_nothing_outside_a_trace():
    assert current_trace() is None
    record_llm_call(model="m", prompt_tokens=1, completion_tokens=1, latency_s=0.1, cost_usd=0.0)


def test_trace_collects_llm_calls_and_totals():
    with Trace() as trace:
        record_llm_call(model="a", prompt_tokens=100, completion_tokens=20, latency_s=0.5, cost_usd=0.001)
        record_llm_call(model="a", prompt_tokens=50, completion_tokens=10, latency_s=0.25, cost_usd=0.002)

    assert current_trace() is None
    assert trace.prompt_tokens == 150
    assert trace.completion_tokens == 30
    assert trace.cost_usd == pytest.approx(0.003)
    assert trace.llm_latency_s == pytest.approx(0.75)


def test_unknown_cost_makes_trace_cost_unknown_not_zero():
    with Trace() as trace:
        record_llm_call(model="a", prompt_tokens=1, completion_tokens=1, latency_s=0.1, cost_usd=0.001)
        record_llm_call(model="b", prompt_tokens=1, completion_tokens=1, latency_s=0.1, cost_usd=None)

    assert trace.cost_usd is None


def test_traced_tool_records_name_args_result_and_errors():
    @traced_tool
    def search(query, k=3):
        return [query] * k

    @traced_tool(name="explode")
    def broken():
        raise ValueError("boom")

    with Trace() as trace:
        assert search("cats", k=2) == ["cats", "cats"]
        with pytest.raises(ValueError):
            broken()

    first, second = trace.tool_calls
    assert first.name == "search"
    assert first.args == {"query": "cats", "k": 2}
    assert first.result == ["cats", "cats"]
    assert first.error is None and first.latency_s >= 0
    assert second.name == "explode" and "boom" in second.error


def test_trace_round_trips_through_dict():
    with Trace() as trace:
        record_llm_call(model="a", prompt_tokens=3, completion_tokens=4, latency_s=0.1, cost_usd=None)

    assert Trace.from_dict(trace.to_dict()).to_dict() == trace.to_dict()


def test_completion_wrapper_records_litellm_usage_and_cost():
    with Trace() as trace:
        response = completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hi"}],
            mock_response="hi there",
        )

    assert response.choices[0].message.content == "hi there"
    (call,) = trace.llm_calls
    assert call.model == "gpt-4o-mini"
    assert call.prompt_tokens > 0 and call.completion_tokens > 0
    assert call.cost_usd is not None and call.cost_usd > 0


def test_completion_wrapper_reports_unknown_price_as_none():
    with Trace() as trace:
        completion(
            model="openai/evalbar-no-such-model",
            messages=[{"role": "user", "content": "x"}],
            mock_response="y",
        )

    assert trace.llm_calls[0].cost_usd is None


def test_completion_wrapper_prices_groq_responses_that_name_an_openai_model(monkeypatch):
    """Groq answers with model='openai/gpt-oss-120b' and no provider hint; completion_cost()
    then looks the model up under OpenAI and fails. The price LiteLLM computed at call time
    (_hidden_params['response_cost']) is the right one."""
    import litellm

    import evalbar.llm as llm

    real_completion = litellm.completion

    def groq_like_completion(**kwargs):
        response = real_completion(**kwargs, mock_response="ok")
        response.model = "openai/gpt-oss-120b"
        response._hidden_params.pop("custom_llm_provider", None)
        response._hidden_params["response_cost"] = 2.295e-05
        return response

    monkeypatch.setattr(llm.litellm, "completion", groq_like_completion)

    with Trace() as trace:
        completion(model="groq/openai/gpt-oss-120b", messages=[{"role": "user", "content": "x"}])

    assert trace.llm_calls[0].cost_usd == pytest.approx(2.295e-05)
