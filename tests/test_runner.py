import pytest

from evalbar.dataset import Case
from evalbar.runner import Eval, RunResult, run
from evalbar.scorers import Score, exact
from evalbar.trace import record_llm_call


def cases():
    return [
        Case(id="a", input="hello", expected={"output": "HELLO"}, metadata={}),
        Case(id="b", input="boom", expected={"output": "BOOM"}, metadata={}),
        Case(id="c", input="skip", expected={}, metadata={}),
    ]


def upper_pipeline(text):
    record_llm_call(model="fake", prompt_tokens=len(text), completion_tokens=1, latency_s=0.01, cost_usd=0.5)
    if text == "boom":
        raise RuntimeError("provider exploded")
    return text.upper()


def test_run_scores_each_case_with_its_own_trace():
    result = run(Eval(name="upper", pipeline=upper_pipeline, dataset=cases(), scorers=[exact()]))

    assert isinstance(result, RunResult)
    a, b, c = result.cases
    assert a.output == "HELLO" and a.error is None
    assert a.scores == [Score(name="exact", value=1.0, passed=True, reason="")]
    assert a.trace.prompt_tokens == 5
    assert c.trace.prompt_tokens == 4
    assert c.scores == []


def test_pipeline_error_is_kept_and_fails_applicable_scorers():
    result = run(Eval(name="upper", pipeline=upper_pipeline, dataset=cases(), scorers=[exact()]))

    b = result.cases[1]
    assert b.output is None
    assert "RuntimeError: provider exploded" in b.error
    assert b.scores == [Score(name="exact", value=0.0, passed=False, reason="pipeline error")]
    assert b.trace.prompt_tokens == 4


def test_plain_function_scorer_is_accepted():
    def length_ok(output, case):
        return Score(name="len", value=1.0, passed=len(output) < 10, reason="")

    result = run(Eval(name="x", pipeline=str.upper, dataset=cases()[:1], scorers=[length_ok]))

    assert result.cases[0].scores[0].name == "length_ok"


def test_min_interval_spaces_out_case_starts():
    clock = {"now": 0.0}
    sleeps = []

    def fake_clock():
        return clock["now"]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    def slow_pipeline(text):
        clock["now"] += 1.0
        return text

    evaluation = Eval(name="x", pipeline=slow_pipeline, dataset=cases(), scorers=[], min_interval_s=4.0)
    run(evaluation, clock=fake_clock, sleep=fake_sleep)

    assert sleeps == [3.0, 3.0]


def test_limit_and_progress_callback():
    seen = []
    result = run(
        Eval(name="x", pipeline=str.upper, dataset=cases(), scorers=[]),
        limit=2,
        on_case=lambda case_result: seen.append(case_result.case_id),
    )

    assert [c.case_id for c in result.cases] == ["a", "b"]
    assert seen == ["a", "b"]


def test_run_result_round_trips_and_records_versions():
    result = run(Eval(name="upper", pipeline=upper_pipeline, dataset=cases(), scorers=[exact()],
                      metadata={"prompt": "v2"}))

    data = result.to_dict()
    assert data["schema_version"] == 1
    assert data["metadata"]["prompt"] == "v2"
    assert "evalbar" in data["environment"] and "litellm" in data["environment"]
    assert RunResult.from_dict(data).to_dict() == data


def test_eval_requires_unique_scorer_names():
    with pytest.raises(ValueError, match="duplicate scorer name 'exact'"):
        Eval(name="x", pipeline=str.upper, dataset=cases(), scorers=[exact(), exact()])


def test_score_names_follow_the_scorer_name():
    def both_right(output, case):
        return Score(name="a different label", value=1.0, passed=True)

    result = run(Eval(name="x", pipeline=str.upper, dataset=cases()[:1], scorers=[both_right]))

    assert result.scorer_names == ["both_right"]
    assert result.cases[0].scores[0].name == "both_right"


def test_case_metadata_is_kept_in_the_run_file():
    dataset = [Case(id="a", input="x", expected={}, metadata={"group": "mixed"})]

    result = run(Eval(name="x", pipeline=str.upper, dataset=dataset, scorers=[]))

    assert result.cases[0].metadata == {"group": "mixed"}
    assert RunResult.from_dict(result.to_dict()).cases[0].metadata == {"group": "mixed"}
