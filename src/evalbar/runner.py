"""Run a pipeline over a dataset, one trace per case."""

from __future__ import annotations

import importlib.metadata
import platform
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

from .dataset import Case
from .scorers import Score, ScoreFn, Scorer, as_scorer
from .trace import Trace

SCHEMA_VERSION = 1


@dataclass
class Eval:
    name: str
    pipeline: Callable[[Any], Any]
    dataset: Sequence[Case]
    scorers: Sequence[Scorer | ScoreFn]
    min_interval_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scorers = [as_scorer(scorer) for scorer in self.scorers]
        seen: set[str] = set()
        for scorer in self.scorers:
            if scorer.name in seen:
                raise ValueError(f"duplicate scorer name '{scorer.name}'; pass name= to tell them apart")
            seen.add(scorer.name)


@dataclass
class CaseResult:
    case_id: str
    input: Any
    expected: dict[str, Any]
    output: Any
    error: str | None
    latency_s: float
    trace: Trace
    scores: list[Score]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "expected": self.expected,
            "metadata": self.metadata,
            "output": self.output,
            "error": self.error,
            "latency_s": self.latency_s,
            "trace": self.trace.to_dict(),
            "scores": [score.__dict__ for score in self.scores],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=data["case_id"],
            input=data["input"],
            expected=data["expected"],
            metadata=data.get("metadata", {}),
            output=data["output"],
            error=data["error"],
            latency_s=data["latency_s"],
            trace=Trace.from_dict(data["trace"]),
            scores=[Score(**score) for score in data["scores"]],
        )


@dataclass
class RunResult:
    name: str
    started_at: str
    finished_at: str
    scorer_names: list[str]
    cases: list[CaseResult]
    metadata: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "scorer_names": self.scorer_names,
            "metadata": self.metadata,
            "environment": self.environment,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported run file schema_version {version!r} (expected {SCHEMA_VERSION})")
        return cls(
            name=data["name"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            scorer_names=list(data["scorer_names"]),
            cases=[CaseResult.from_dict(case) for case in data["cases"]],
            metadata=data.get("metadata", {}),
            environment=data.get("environment", {}),
        )


def run(
    evaluation: Eval,
    *,
    limit: int | None = None,
    on_case: Callable[[CaseResult], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Run every case sequentially. Pipeline exceptions become case errors, never crashes."""
    cases = list(evaluation.dataset)[:limit] if limit is not None else list(evaluation.dataset)
    started_at = _now()
    results: list[CaseResult] = []
    last_start: float | None = None

    for case in cases:
        if last_start is not None and evaluation.min_interval_s > 0:
            wait = evaluation.min_interval_s - (clock() - last_start)
            if wait > 0:
                sleep(wait)
        last_start = clock()
        result = _run_case(evaluation, case)
        results.append(result)
        if on_case is not None:
            on_case(result)

    return RunResult(
        name=evaluation.name,
        started_at=started_at,
        finished_at=_now(),
        scorer_names=[scorer.name for scorer in evaluation.scorers],
        cases=results,
        metadata=dict(evaluation.metadata),
        environment=_environment(),
    )


def _run_case(evaluation: Eval, case: Case) -> CaseResult:
    output: Any = None
    error: str | None = None
    started = time.perf_counter()
    with Trace() as trace:
        try:
            output = evaluation.pipeline(case.input)
        except Exception as exc:
            # Spans recorded before the failure stay on the trace: a failed call still cost money.
            error = f"{type(exc).__name__}: {exc}"
    latency_s = time.perf_counter() - started

    scores: list[Score] = []
    for scorer in evaluation.scorers:
        if error is not None:
            if scorer.applies(case):
                scores.append(Score(name=scorer.name, value=0.0, passed=False, reason="pipeline error"))
            continue
        try:
            score = scorer(output, case)
        except Exception as exc:
            score = Score(name=scorer.name, value=0.0, passed=False,
                          reason=f"scorer error: {type(exc).__name__}: {exc}")
        if score is not None:
            # Reports group by scorer name, so a custom scorer cannot rename its scores.
            scores.append(replace(score, name=scorer.name))

    return CaseResult(
        case_id=case.id,
        input=case.input,
        expected=case.expected,
        output=output,
        error=error,
        latency_s=latency_s,
        trace=trace,
        scores=scores,
        metadata=case.metadata,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _environment() -> dict[str, str]:
    env = {"python": platform.python_version()}
    for package in ("evalbar", "litellm"):
        try:
            env[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            env[package] = "unknown"
    return env
