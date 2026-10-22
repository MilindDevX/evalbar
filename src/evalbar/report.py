"""Aggregate a run, render it for the terminal, and read/write JSON run files."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.table import Table
from rich.text import Text

from .runner import RunResult


@dataclass(frozen=True)
class ScorerSummary:
    n: int
    passed: int
    mean: float

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0


@dataclass(frozen=True)
class Summary:
    name: str
    n_cases: int
    n_errors: int
    scorers: dict[str, ScorerSummary]
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    n_unpriced_cases: int
    judge_calls: int
    judge_cost_usd: float
    latency_p50_s: float
    latency_p95_s: float
    llm_latency_p50_s: float
    llm_latency_p95_s: float


def summarize(run: RunResult) -> Summary:
    scorers = {}
    for name in run.scorer_names:
        scores = [score for case in run.cases for score in case.scores if score.name == name]
        scorers[name] = ScorerSummary(
            n=len(scores),
            passed=sum(score.passed for score in scores),
            mean=sum(score.value for score in scores) / len(scores) if scores else 0.0,
        )
    latencies = sorted(case.latency_s for case in run.cases)
    # Time spent inside model calls only: excludes rate-limit waits and local work.
    llm_latencies = sorted(case.trace.llm_latency_s for case in run.cases if case.trace.pipeline_calls)
    return Summary(
        name=run.name,
        n_cases=len(run.cases),
        n_errors=sum(case.error is not None for case in run.cases),
        scorers=scorers,
        llm_calls=sum(len(case.trace.pipeline_calls) for case in run.cases),
        prompt_tokens=sum(case.trace.prompt_tokens for case in run.cases),
        completion_tokens=sum(case.trace.completion_tokens for case in run.cases),
        cost_usd=sum(call.cost_usd or 0.0 for case in run.cases for call in case.trace.pipeline_calls),
        n_unpriced_cases=sum(
            1 for case in run.cases if case.trace.pipeline_calls and case.trace.cost_usd is None
        ),
        judge_calls=sum(len(case.trace.judge_calls) for case in run.cases),
        judge_cost_usd=sum(call.cost_usd or 0.0 for case in run.cases for call in case.trace.judge_calls),
        latency_p50_s=percentile(latencies, 0.50),
        latency_p95_s=percentile(latencies, 0.95),
        llm_latency_p50_s=percentile(llm_latencies, 0.50),
        llm_latency_p95_s=percentile(llm_latencies, 0.95),
    )


def report_renderable(run: RunResult, max_failures: int = 15) -> Group:
    summary = summarize(run)
    parts: list = []

    header = Text()
    header.append(f"evalbar run: {run.name}", style="bold")
    header.append(f"  {summary.n_cases} cases, {summary.n_errors} errors, started {run.started_at}")
    parts.append(header)
    if run.metadata:
        parts.append(Text("  " + ", ".join(f"{key}={value}" for key, value in run.metadata.items()), style="dim"))

    scores = Table(title="Scores", title_justify="left")
    for column in ("scorer", "cases", "passed", "pass rate", "mean"):
        scores.add_column(column, justify="left" if column == "scorer" else "right", no_wrap=True)
    for name, scorer in summary.scorers.items():
        scores.add_row(name, str(scorer.n), str(scorer.passed), f"{scorer.pass_rate:.1%}", f"{scorer.mean:.3f}")
    parts.append(scores)

    usage = Table(title="Usage", title_justify="left")
    usage.add_column("metric")
    usage.add_column("value", justify="right")
    per_case = summary.n_cases or 1
    usage.add_row("LLM calls", str(summary.llm_calls))
    usage.add_row("prompt tokens", f"{summary.prompt_tokens:,}")
    usage.add_row("completion tokens", f"{summary.completion_tokens:,}")
    usage.add_row("cost (list price)", format_cost(summary.cost_usd))
    usage.add_row("cost per case", format_cost(summary.cost_usd / per_case))
    if summary.judge_calls:
        usage.add_row("judge calls", str(summary.judge_calls))
        usage.add_row("judge cost (not in cost above)", format_cost(summary.judge_cost_usd))
    usage.add_row("case latency p50 / p95", f"{summary.latency_p50_s:.2f}s / {summary.latency_p95_s:.2f}s")
    usage.add_row("LLM latency p50 / p95", f"{summary.llm_latency_p50_s:.2f}s / {summary.llm_latency_p95_s:.2f}s")
    parts.append(usage)
    if summary.n_unpriced_cases:
        parts.append(Text(f"{summary.n_unpriced_cases} case(s) had no known price; cost excludes them.",
                          style="yellow"))

    failures = [
        (case.case_id, score.name, score.reason)
        for case in run.cases
        if case.error is None
        for score in case.scores
        if not score.passed
    ]
    errors = [(case.case_id, case.error) for case in run.cases if case.error is not None]
    if failures:
        table = Table(title=f"Failures ({len(failures)})", title_justify="left")
        table.add_column("case")
        table.add_column("scorer")
        table.add_column("reason", overflow="fold")
        for case_id, name, reason in failures[:max_failures]:
            table.add_row(case_id, name, reason)
        parts.append(table)
        if len(failures) > max_failures:
            parts.append(Text(f"... {len(failures) - max_failures} more in the run file", style="dim"))
    if errors:
        table = Table(title=f"Errors ({len(errors)})", title_justify="left")
        table.add_column("case")
        table.add_column("error", overflow="fold")
        for case_id, error in errors[:max_failures]:
            table.add_row(case_id, error or "")
        parts.append(table)
    return Group(*parts)


def render(run: RunResult, width: int | None = None, max_failures: int = 15) -> str:
    """The terminal report as plain text."""
    console = Console(file=io.StringIO(), width=width or 120, record=True, color_system=None)
    console.print(report_renderable(run, max_failures))
    return console.export_text()


def save_run(run: RunResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run.to_dict(), indent=2, ensure_ascii=False, default=repr) + "\n",
                    encoding="utf-8")
    return path


def load_run(path: str | Path) -> RunResult:
    return RunResult.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def format_cost(usd: float) -> str:
    return f"${usd:.4f}" if usd >= 0.01 or usd == 0 else f"${usd:.6f}"


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)
