import pytest

from evalbar.dataset import Case
from evalbar.report import load_run, render, save_run, summarize
from evalbar.runner import Eval, run
from evalbar.scorers import exact, regex
from evalbar.trace import record_llm_call


def pipeline(text):
    cost = None if text == "unpriced" else 0.01
    record_llm_call(model="fake", prompt_tokens=10, completion_tokens=5, latency_s=0.1, cost_usd=cost)
    if text == "boom":
        raise RuntimeError("nope")
    return {"label": text}


def make_run():
    dataset = [
        Case(id="ok", input="cat", expected={"label": "cat"}, metadata={}),
        Case(id="wrong", input="dog", expected={"label": "cat"}, metadata={}),
        Case(id="err", input="boom", expected={"label": "boom"}, metadata={}),
        Case(id="unpriced", input="unpriced", expected={}, metadata={}),
    ]
    scorers = [exact("label"), regex("label", r"^[a-z]+$", name="lowercase")]
    return run(Eval(name="demo", pipeline=pipeline, dataset=dataset, scorers=scorers))


def test_summary_aggregates_scores_over_applicable_cases():
    summary = summarize(make_run())

    label = summary.scorers["exact:label"]
    assert (label.n, label.passed) == (3, 1)
    assert label.mean == pytest.approx(1 / 3)
    lowercase = summary.scorers["lowercase"]
    assert (lowercase.n, lowercase.passed) == (4, 3)


def test_summary_totals_tokens_cost_errors_and_latency():
    summary = summarize(make_run())

    assert summary.n_cases == 4
    assert summary.n_errors == 1
    assert summary.prompt_tokens == 40 and summary.completion_tokens == 20
    assert summary.cost_usd == pytest.approx(0.03)
    assert summary.n_unpriced_cases == 1
    assert summary.latency_p50_s >= 0 and summary.latency_p95_s >= summary.latency_p50_s
    assert summary.llm_latency_p50_s == pytest.approx(0.1)


def test_render_shows_scores_costs_and_failures():
    text = render(make_run(), width=120)

    assert "demo" in text
    assert "exact:label" in text and "33.3%" in text
    assert "$0.0300" in text and "1 case(s) had no known price" in text
    assert "wrong" in text and "expected 'cat', got 'dog'" in text
    assert "err" in text and "RuntimeError: nope" in text


def test_save_and_load_run_round_trip(tmp_path):
    result = make_run()
    path = tmp_path / "runs" / "demo.json"

    save_run(result, path)

    assert load_run(path).to_dict() == result.to_dict()
