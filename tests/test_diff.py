import pytest

from evalbar.dataset import Case
from evalbar.diff import check_gates, diff_runs, render_diff, render_markdown
from evalbar.runner import Eval, run
from evalbar.scorers import exact, regex
from evalbar.trace import record_llm_call

EXPECTED = {"a": "cat", "b": "dog", "c": "cow", "d": "pig"}


def make_run(name, answers, cost=0.01, ids=("a", "b", "c", "d"), extra_scorers=()):
    """answers maps case id -> output label; the label 'boom' raises inside the pipeline."""

    def pipeline(case_id):
        record_llm_call(model="fake", prompt_tokens=100, completion_tokens=10, latency_s=0.5, cost_usd=cost)
        if answers[case_id] == "boom":
            raise RuntimeError("provider down")
        return {"label": answers[case_id]}

    dataset = [Case(id=i, input=i, expected={"label": EXPECTED[i]}, metadata={}) for i in ids]
    return run(Eval(name=name, pipeline=pipeline, dataset=dataset, scorers=[exact("label"), *extra_scorers]))


BASE = {"a": "cat", "b": "dog", "c": "cow", "d": "horse"}  # 3/4


def test_reports_rates_delta_regressions_and_fixes():
    head = make_run("head", {"a": "cat", "b": "wolf", "c": "cow", "d": "pig"})  # b regressed, d fixed

    diff = diff_runs(make_run("base", BASE), head)

    (label,) = diff.scorers
    assert label.name == "exact:label"
    assert (label.n, label.base_passed, label.head_passed) == (4, 3, 3)
    assert label.delta == pytest.approx(0.0)
    assert label.regressions == ["b"]
    assert label.fixes == ["d"]


def test_compares_only_cases_present_in_both_runs():
    head = make_run("head", {"a": "cat", "b": "dog", "c": "cow"}, ids=("a", "b", "c"))

    diff = diff_runs(make_run("base", BASE), head)

    assert diff.removed_cases == ["d"] and diff.added_cases == []
    assert diff.scorers[0].n == 3
    assert diff.scorers[0].base_rate == pytest.approx(1.0)


def test_new_errors_count_as_regressions_and_are_listed():
    head = make_run("head", {"a": "cat", "b": "boom", "c": "cow", "d": "horse"})

    diff = diff_runs(make_run("base", BASE), head)

    assert diff.new_errors == ["b"]
    assert diff.scorers[0].regressions == ["b"]


def test_cost_tokens_and_latency_per_case():
    diff = diff_runs(make_run("base", BASE, cost=0.01), make_run("head", BASE, cost=0.015))

    assert diff.base_cost_per_case == pytest.approx(0.01)
    assert diff.head_cost_per_case == pytest.approx(0.015)
    assert diff.cost_change == pytest.approx(0.5)
    assert diff.head_tokens_per_case == pytest.approx(110)
    assert diff.head_llm_latency_p50_s == pytest.approx(0.5)


def test_unknown_cost_disables_cost_change():
    diff = diff_runs(make_run("base", BASE), make_run("head", BASE, cost=None))

    assert diff.head_cost_per_case is None and diff.cost_change is None


def test_gate_fails_when_a_scorer_drops_more_than_the_threshold():
    head = make_run("head", {"a": "cat", "b": "wolf", "c": "cow", "d": "horse"})  # 3/4 -> 2/4
    diff = diff_runs(make_run("base", BASE), head)

    failures = check_gates(diff, fail_on_drop=0.10)

    assert len(failures) == 1
    assert "exact:label dropped 25.0 pts (75.0% -> 50.0%)" in failures[0]
    assert check_gates(diff, fail_on_drop=0.30) == []


def test_gate_with_zero_threshold_allows_equal_and_better():
    diff = diff_runs(make_run("base", BASE), make_run("head", BASE))

    assert check_gates(diff, fail_on_drop=0.0) == []


def test_gate_can_be_limited_to_named_scorers():
    lowercase = [regex("label", r"^[a-z]+$", name="lowercase")]
    head = make_run("head", {"a": "cat", "b": "wolf", "c": "cow", "d": "horse"}, extra_scorers=lowercase)
    diff = diff_runs(make_run("base", BASE, extra_scorers=lowercase), head)

    assert check_gates(diff, fail_on_drop=0.0, scorers=["exact:label"])
    assert check_gates(diff, fail_on_drop=0.0, scorers=["lowercase"]) == []
    with pytest.raises(ValueError, match="unknown scorer 'nope'"):
        check_gates(diff, fail_on_drop=0.0, scorers=["nope"])


def test_cost_gate():
    diff = diff_runs(make_run("base", BASE, cost=0.01), make_run("head", BASE, cost=0.015))

    (failure,) = check_gates(diff, max_cost_increase=0.2)
    assert "cost per case rose 50%" in failure
    assert check_gates(diff, max_cost_increase=0.6) == []


def test_text_and_markdown_rendering_name_the_flipped_cases():
    head = make_run("head", {"a": "cat", "b": "wolf", "c": "cow", "d": "pig"})
    diff = diff_runs(make_run("base", BASE), head)

    text = render_diff(diff, width=120)
    markdown = render_markdown(diff, failures=["exact:label dropped"])

    for output in (text, markdown):
        assert "exact:label" in output and "75.0%" in output
        assert "b" in output and "regressed" in output.lower()
    assert markdown.startswith("### evalbar diff: base -> head")
    assert "| exact:label |" in markdown
    assert "**Gate failed**" in markdown


def test_cost_gate_fails_loudly_when_cost_is_unknown():
    diff = diff_runs(make_run("base", BASE), make_run("head", BASE, cost=None))

    (failure,) = check_gates(diff, max_cost_increase=0.2)
    assert "cost per case is unknown" in failure


def test_cost_gate_from_free_baseline():
    free = make_run("base", BASE, cost=0.0)

    assert check_gates(diff_runs(free, make_run("head", BASE, cost=0.0)), max_cost_increase=0.0) == []
    (failure,) = check_gates(diff_runs(free, make_run("head", BASE, cost=0.01)), max_cost_increase=0.5)
    assert "from $0" in failure
