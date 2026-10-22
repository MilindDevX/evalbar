"""Compare two runs case by case and decide whether a CI gate passes."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass

from rich import box
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from .report import format_cost, percentile
from .runner import CaseResult, RunResult

_EPSILON = 1e-9
_MAX_IDS = 10


@dataclass(frozen=True)
class ScorerDiff:
    name: str
    n: int
    base_passed: int
    head_passed: int
    regressions: list[str]
    fixes: list[str]

    @property
    def base_rate(self) -> float:
        return self.base_passed / self.n if self.n else 0.0

    @property
    def head_rate(self) -> float:
        return self.head_passed / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        return self.head_rate - self.base_rate


@dataclass(frozen=True)
class RunDiff:
    base_name: str
    head_name: str
    n_common: int
    scorers: list[ScorerDiff]
    added_cases: list[str]
    removed_cases: list[str]
    new_errors: list[str]
    fixed_errors: list[str]
    base_cost_per_case: float | None
    head_cost_per_case: float | None
    base_tokens_per_case: float
    head_tokens_per_case: float
    base_llm_latency_p50_s: float
    head_llm_latency_p50_s: float

    @property
    def cost_change(self) -> float | None:
        """Relative change in cost per case, or None when either side has unpriced calls."""
        if self.base_cost_per_case is None or self.head_cost_per_case is None or not self.base_cost_per_case:
            return None
        return self.head_cost_per_case / self.base_cost_per_case - 1


def diff_runs(base: RunResult, head: RunResult) -> RunDiff:
    """Compare on the cases both runs share, so a changed dataset cannot fake a gain."""
    base_cases = {case.case_id: case for case in base.cases}
    head_cases = {case.case_id: case for case in head.cases}
    common = [case_id for case_id in base_cases if case_id in head_cases]
    pairs = [(base_cases[case_id], head_cases[case_id]) for case_id in common]

    scorer_names = [name for name in base.scorer_names if name in head.scorer_names]
    scorers = [_diff_scorer(name, pairs) for name in scorer_names]

    return RunDiff(
        base_name=base.name,
        head_name=head.name,
        n_common=len(common),
        scorers=scorers,
        added_cases=[case_id for case_id in head_cases if case_id not in base_cases],
        removed_cases=[case_id for case_id in base_cases if case_id not in head_cases],
        new_errors=[b.case_id for b, h in pairs if b.error is None and h.error is not None],
        fixed_errors=[b.case_id for b, h in pairs if b.error is not None and h.error is None],
        base_cost_per_case=_cost_per_case([b for b, _ in pairs]),
        head_cost_per_case=_cost_per_case([h for _, h in pairs]),
        base_tokens_per_case=_tokens_per_case([b for b, _ in pairs]),
        head_tokens_per_case=_tokens_per_case([h for _, h in pairs]),
        base_llm_latency_p50_s=_llm_p50([b for b, _ in pairs]),
        head_llm_latency_p50_s=_llm_p50([h for _, h in pairs]),
    )


def check_gates(diff: RunDiff, *, fail_on_drop: float | None = None, scorers: Sequence[str] | None = None,
                max_cost_increase: float | None = None) -> list[str]:
    """Return one message per failed gate; an empty list means the gate passes."""
    known = {scorer.name for scorer in diff.scorers}
    for name in scorers or []:
        if name not in known:
            raise ValueError(f"unknown scorer '{name}' (in both runs: {', '.join(sorted(known)) or 'none'})")

    failures = []
    if fail_on_drop is not None:
        for scorer in diff.scorers:
            if scorers and scorer.name not in scorers:
                continue
            drop = scorer.base_rate - scorer.head_rate
            if drop > fail_on_drop + _EPSILON:
                cases = _ids(scorer.regressions)
                failures.append(
                    f"{scorer.name} dropped {drop * 100:.1f} pts ({scorer.base_rate:.1%} -> {scorer.head_rate:.1%}),"
                    f" allowed {fail_on_drop * 100:.1f}; regressed: {cases}"
                )
    if max_cost_increase is not None:
        failure = _cost_gate(diff, max_cost_increase)
        if failure:
            failures.append(failure)
    return failures


def _cost_gate(diff: RunDiff, max_increase: float) -> str | None:
    base, head = diff.base_cost_per_case, diff.head_cost_per_case
    if base is None or head is None:
        # A requested gate must not pass on missing data.
        return "cost gate requested but cost per case is unknown (some LLM calls have no known price)"
    if base == 0:
        return f"cost per case rose from $0 to {format_cost(head)}" if head > 0 else None
    change = head / base - 1
    if change > max_increase + _EPSILON:
        return (f"cost per case rose {change:.0%} ({format_cost(base)} -> {format_cost(head)}), "
                f"allowed {max_increase:.0%}")
    return None


def diff_renderable(diff: RunDiff) -> Group:
    parts: list = [Text(f"evalbar diff: {diff.base_name} -> {diff.head_name}  ({diff.n_common} shared cases)",
                        style="bold")]
    if diff.added_cases or diff.removed_cases:
        parts.append(Text(f"  only in head: {_ids(diff.added_cases)}; only in base: {_ids(diff.removed_cases)}",
                          style="yellow"))

    table = Table(title="Scores (pass rate)", title_justify="left", box=box.SIMPLE_HEAD)
    table.add_column("scorer", no_wrap=True)
    for column in ("n", "base", "head", "change", "down", "up"):
        table.add_column(column, justify="right", no_wrap=True)
    for scorer in diff.scorers:
        style = "red" if scorer.delta < -_EPSILON else ("green" if scorer.delta > _EPSILON else "")
        table.add_row(scorer.name, str(scorer.n), f"{scorer.base_rate:.1%}", f"{scorer.head_rate:.1%}",
                      Text(f"{scorer.delta * 100:+.1f}", style=style), str(len(scorer.regressions)),
                      str(len(scorer.fixes)))
    parts.append(table)
    for scorer in diff.scorers:
        if scorer.regressions:
            parts.append(Text(f"regressed {scorer.name}: {_ids(scorer.regressions)}", style="red"))
        if scorer.fixes:
            parts.append(Text(f"fixed {scorer.name}: {_ids(scorer.fixes)}", style="green"))

    usage = Table(title="Usage per case", title_justify="left", box=box.SIMPLE_HEAD)
    for column in ("metric", "base", "head", "change"):
        usage.add_column(column, justify="left" if column == "metric" else "right")
    for row in _usage_rows(diff):
        usage.add_row(*row)
    parts.append(usage)
    if diff.new_errors:
        parts.append(Text(f"new errors: {_ids(diff.new_errors)}", style="red"))
    if diff.fixed_errors:
        parts.append(Text(f"fixed errors: {_ids(diff.fixed_errors)}", style="green"))
    return Group(*parts)


def render_diff(diff: RunDiff, width: int | None = None) -> str:
    console = Console(file=io.StringIO(), width=width or 120, record=True, color_system=None)
    console.print(diff_renderable(diff))
    return console.export_text()


def render_markdown(diff: RunDiff, failures: Sequence[str] = ()) -> str:
    """GitHub-flavoured markdown, e.g. for $GITHUB_STEP_SUMMARY or a PR comment."""
    lines = [f"### evalbar diff: {diff.base_name} -> {diff.head_name}", "",
             f"{diff.n_common} shared cases."]
    if diff.added_cases or diff.removed_cases:
        lines.append(f"Only in head: {_ids(diff.added_cases)}. Only in base: {_ids(diff.removed_cases)}.")
    lines += ["", "| scorer | cases | base | head | change | regressed | fixed |", "|---|---|---|---|---|---|---|"]
    for scorer in diff.scorers:
        lines.append(f"| {scorer.name} | {scorer.n} | {scorer.base_rate:.1%} | {scorer.head_rate:.1%} | "
                     f"{scorer.delta * 100:+.1f} pts | {_ids(scorer.regressions)} | {_ids(scorer.fixes)} |")
    lines += ["", "| per case | base | head | change |", "|---|---|---|---|"]
    lines += [f"| {' | '.join(row)} |" for row in _usage_rows(diff)]
    if diff.new_errors:
        lines += ["", f"New errors: {_ids(diff.new_errors)}"]
    lines.append("")
    if failures:
        lines.append("**Gate failed**")
        lines += [f"- {failure}" for failure in failures]
    else:
        lines.append("Gate passed.")
    return "\n".join(lines) + "\n"


def _diff_scorer(name: str, pairs: list[tuple[CaseResult, CaseResult]]) -> ScorerDiff:
    n = base_passed = head_passed = 0
    regressions, fixes = [], []
    for base_case, head_case in pairs:
        before = _passed(base_case, name)
        after = _passed(head_case, name)
        if before is None or after is None:
            continue
        n += 1
        base_passed += before
        head_passed += after
        if before and not after:
            regressions.append(base_case.case_id)
        elif after and not before:
            fixes.append(base_case.case_id)
    return ScorerDiff(name, n, base_passed, head_passed, regressions, fixes)


def _passed(case: CaseResult, name: str) -> bool | None:
    return next((score.passed for score in case.scores if score.name == name), None)


def _cost_per_case(cases: list[CaseResult]) -> float | None:
    if not cases:
        return None
    costs = [case.trace.cost_usd for case in cases]
    if any(cost is None for cost in costs):
        return None
    return sum(costs) / len(cases)  # type: ignore[arg-type]


def _tokens_per_case(cases: list[CaseResult]) -> float:
    if not cases:
        return 0.0
    return sum(case.trace.prompt_tokens + case.trace.completion_tokens for case in cases) / len(cases)


def _llm_p50(cases: list[CaseResult]) -> float:
    return percentile(sorted(case.trace.llm_latency_s for case in cases if case.trace.llm_calls), 0.5)


def _usage_rows(diff: RunDiff) -> list[tuple[str, str, str, str]]:
    def cost(value: float | None) -> str:
        return "unknown" if value is None else format_cost(value)

    def relative(before: float, after: float, floor: float = 0.0) -> str:
        return f"{after / before - 1:+.0%}" if before > floor else ""

    cost_change = "" if diff.cost_change is None else f"{diff.cost_change:+.0%}"
    return [
        ("cost (list price)", cost(diff.base_cost_per_case), cost(diff.head_cost_per_case), cost_change),
        ("tokens", f"{diff.base_tokens_per_case:,.0f}", f"{diff.head_tokens_per_case:,.0f}",
         relative(diff.base_tokens_per_case, diff.head_tokens_per_case)),
        ("LLM latency p50", f"{diff.base_llm_latency_p50_s:.2f}s", f"{diff.head_llm_latency_p50_s:.2f}s",
         # Sub-50 ms latencies are mocks or caches; a percentage of them is noise.
         relative(diff.base_llm_latency_p50_s, diff.head_llm_latency_p50_s, floor=0.05)),
    ]


def _ids(case_ids: Sequence[str]) -> str:
    if not case_ids:
        return "-"
    shown = ", ".join(case_ids[:_MAX_IDS])
    return shown if len(case_ids) <= _MAX_IDS else f"{shown} (+{len(case_ids) - _MAX_IDS} more)"
