import json

import pytest

from evalbar.dataset import Case
from evalbar.report import render, summarize
from evalbar.runner import Eval, run
from evalbar.scorers import llm_judge, recall_at_k
from evalbar.trace import record_llm_call


def case(expected=None, input="q"):
    return Case(id="c1", input=input, expected=expected or {}, metadata={})


# recall@k ------------------------------------------------------------------


def test_recall_at_k_counts_relevant_ids_in_the_top_k():
    scorer = recall_at_k(k=2)

    score = scorer({"retrieved": ["d1", "d2", "d3"]}, case({"relevant": ["d1", "d3"]}))

    assert score.name == "recall@2"
    assert score.value == pytest.approx(0.5)
    assert not score.passed
    assert "d3" in score.reason


def test_recall_accepts_dicts_with_ids_and_a_pass_threshold():
    scorer = recall_at_k("docs", relevant_key="gold", k=3, pass_threshold=0.5)

    score = scorer({"docs": [{"id": "a"}, {"id": "b"}]}, case({"gold": ["b", "z"]}))

    assert score.value == pytest.approx(0.5) and score.passed


def test_recall_skips_cases_without_relevant_ids_and_fails_missing_field():
    scorer = recall_at_k(k=5)

    assert scorer({"retrieved": []}, case()) is None
    assert scorer({"retrieved": []}, case({"relevant": []})) is None
    assert not scorer({"answer": "x"}, case({"relevant": ["a"]})).passed


# llm_judge -----------------------------------------------------------------


def judge(verdict, **kwargs):
    reply = verdict if isinstance(verdict, str) else json.dumps(verdict)
    return llm_judge("Is the answer correct and polite?", model="gpt-4o-mini", mock_response=reply, **kwargs)


def test_judge_returns_the_models_score_and_reason():
    score = judge({"score": 0.8, "reason": "correct, a bit curt"})({"answer": "Paris"}, case({"answer": "Paris"}))

    assert score.name == "judge"
    assert score.value == pytest.approx(0.8)
    assert score.passed
    assert score.reason == "correct, a bit curt"


def test_judge_threshold_and_clamping():
    low = judge({"score": 0.4, "reason": "wrong"}, pass_threshold=0.5)({"a": 1}, case())
    high = judge({"score": 7, "reason": "?"})({"a": 1}, case())

    assert not low.passed
    assert high.value == 1.0


def test_unparseable_verdict_fails_visibly():
    score = judge("I think it is fine")({"a": 1}, case())

    assert not score.passed
    assert score.reason.startswith("judge error")


def test_judge_prompt_contains_rubric_input_expected_and_output(monkeypatch):
    import evalbar.scorers as scorers

    seen = {}
    real = scorers.completion

    def spy(**kwargs):
        seen.update(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(scorers, "completion", spy)
    polite = judge({"score": 1, "reason": "ok"}, name="polite")
    polite({"answer": "Paris!"}, case({"answer": "Paris"}, input="Capital?"))

    prompt = seen["messages"][-1]["content"]
    for part in ("Is the answer correct and polite?", "Capital?", "Paris!", '"answer": "Paris"'):
        assert part in prompt
    assert seen["temperature"] == 0
    assert seen["response_format"] == {"type": "json_object"}


def test_judge_can_be_limited_to_cases_with_an_expectation():
    scorer = judge({"score": 1, "reason": "ok"}, expects="rubric_applies")

    assert scorer({"a": 1}, case()) is None
    assert scorer({"a": 1}, case({"rubric_applies": True})).passed


# judge cost is kept apart from pipeline cost ---------------------------------


def test_judge_calls_are_traced_separately_from_the_pipeline():
    def pipeline(text):
        record_llm_call(model="fake", prompt_tokens=100, completion_tokens=10, latency_s=0.2, cost_usd=0.01)
        return {"answer": text}

    dataset = [Case(id="a", input="hi", expected={}, metadata={})]
    result = run(Eval(name="j", pipeline=pipeline, dataset=dataset, scorers=[judge({"score": 1, "reason": "ok"})]))

    trace = result.cases[0].trace
    assert trace.cost_usd == pytest.approx(0.01)
    assert trace.prompt_tokens == 100
    assert trace.judge_cost_usd > 0
    assert [call.role for call in trace.llm_calls] == ["pipeline", "judge"]

    summary = summarize(result)
    assert summary.cost_usd == pytest.approx(0.01)
    assert summary.judge_cost_usd == pytest.approx(trace.judge_cost_usd)
    assert "judge cost" in render(result)
