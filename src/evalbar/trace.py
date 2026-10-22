"""Per-case traces: LLM calls (tokens, latency, cost) and tool calls.

The active trace lives in a context variable, so code deep inside a pipeline can record
spans without being handed the trace. Recording outside a trace is a no-op, which lets
the same pipeline run in production unchanged.

Each LLM call carries a role: "pipeline" for the system under test, "judge" for calls an
LLM-as-judge scorer makes. Totals on a Trace are pipeline-only, so grading a run never
inflates the cost of the system being graded.
"""

from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from typing import Any

_current: ContextVar[Trace | None] = ContextVar("evalbar_trace", default=None)
_role: ContextVar[str] = ContextVar("evalbar_role", default="pipeline")


@dataclass
class LLMCall:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float | None
    role: str = "pipeline"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: Any
    latency_s: float
    error: str | None = None


@dataclass
class Trace:
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    _token: Token | None = field(default=None, repr=False, compare=False)

    def __enter__(self) -> Trace:
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    @contextmanager
    def recording_as(self, role: str) -> Iterator[Trace]:
        """Re-open this trace with LLM calls tagged `role` (the runner uses "judge" while scoring)."""
        role_token = _role.set(role)
        try:
            with self:
                yield self
        finally:
            _role.reset(role_token)

    @property
    def pipeline_calls(self) -> list[LLMCall]:
        return [call for call in self.llm_calls if call.role == "pipeline"]

    @property
    def judge_calls(self) -> list[LLMCall]:
        return [call for call in self.llm_calls if call.role == "judge"]

    @property
    def prompt_tokens(self) -> int:
        return sum(call.prompt_tokens for call in self.pipeline_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(call.completion_tokens for call in self.pipeline_calls)

    @property
    def llm_latency_s(self) -> float:
        return sum(call.latency_s for call in self.pipeline_calls)

    @property
    def cost_usd(self) -> float | None:
        """Pipeline cost, or None when any call has no known price (unknown is not free)."""
        return _total_cost(self.pipeline_calls)

    @property
    def judge_cost_usd(self) -> float | None:
        return _total_cost(self.judge_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": [asdict(call) for call in self.llm_calls],
            "tool_calls": [asdict(call) for call in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trace:
        return cls(
            llm_calls=[LLMCall(**call) for call in data.get("llm_calls", [])],
            tool_calls=[ToolCall(**call) for call in data.get("tool_calls", [])],
        )


def _total_cost(calls: list[LLMCall]) -> float | None:
    costs = [call.cost_usd for call in calls]
    if any(cost is None for cost in costs):
        return None
    return sum(costs)  # type: ignore[arg-type]


def current_trace() -> Trace | None:
    return _current.get()


def record_llm_call(*, model: str, prompt_tokens: int, completion_tokens: int, latency_s: float,
                    cost_usd: float | None) -> None:
    trace = _current.get()
    if trace is not None:
        trace.llm_calls.append(LLMCall(model, prompt_tokens, completion_tokens, latency_s, cost_usd, _role.get()))


def record_tool_call(*, name: str, args: dict[str, Any], result: Any, latency_s: float,
                     error: str | None = None) -> None:
    trace = _current.get()
    if trace is not None:
        trace.tool_calls.append(ToolCall(name, args, result, latency_s, error))


def traced_tool(fn=None, *, name: str | None = None):
    """Decorator that records each call of a tool function in the active trace."""

    def decorate(func):
        tool_name = name or func.__name__
        signature = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            started = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                record_tool_call(name=tool_name, args=dict(bound.arguments), result=None,
                                 latency_s=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")
                raise
            record_tool_call(name=tool_name, args=dict(bound.arguments), result=result,
                             latency_s=time.perf_counter() - started)
            return result

        return wrapper

    return decorate(fn) if fn is not None else decorate
