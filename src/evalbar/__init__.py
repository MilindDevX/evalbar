"""evalbar: run an LLM pipeline over test cases, trace tokens/latency/cost, and score it."""

from .dataset import Case, DatasetError, load_dataset
from .diff import RunDiff, check_gates, diff_runs
from .llm import completion
from .report import load_run, render, save_run, summarize
from .runner import CaseResult, Eval, RunResult, run
from .scorers import Score, Scorer
from .trace import Trace, current_trace, record_llm_call, record_tool_call, traced_tool

__all__ = [
    "Case",
    "CaseResult",
    "DatasetError",
    "Eval",
    "RunDiff",
    "RunResult",
    "Score",
    "Scorer",
    "Trace",
    "check_gates",
    "completion",
    "current_trace",
    "diff_runs",
    "load_dataset",
    "load_run",
    "record_llm_call",
    "record_tool_call",
    "render",
    "run",
    "save_run",
    "summarize",
    "traced_tool",
]
