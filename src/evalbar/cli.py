"""Command line: `evalbar run`, `show`, `diff` and `html`."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from .diff import check_gates, diff_renderable, diff_runs, render_markdown
from .html import render_html
from .report import load_run, report_renderable, save_run
from .runner import CaseResult, Eval, run


class UsageError(Exception):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalbar", description="Run, trace and score LLM pipelines.")
    commands = parser.add_subparsers(dest="command", required=True)

    run_cmd = commands.add_parser("run", help="run an Eval defined in a Python file")
    run_cmd.add_argument("target", help="path/to/eval.py or path/to/eval.py:eval_variable")
    run_cmd.add_argument("--out", help="run file to write (default: runs/<name>-<timestamp>.json)")
    run_cmd.add_argument("--limit", type=int, help="only run the first N cases (smoke test)")
    run_cmd.add_argument("--quiet", action="store_true", help="no per-case progress lines")
    run_cmd.add_argument("--env-file", default=".env",
                         help="KEY=VALUE file loaded before the eval (default: .env if present); "
                              "variables already set win")

    show_cmd = commands.add_parser("show", help="print the report for a saved run file")
    show_cmd.add_argument("run_file")

    html_cmd = commands.add_parser("html", help="write a self-contained HTML report for a run file")
    html_cmd.add_argument("run_file")
    html_cmd.add_argument("--out", help="HTML file to write (default: the run file with .html)")

    diff_cmd = commands.add_parser("diff", help="compare two run files; exit 1 when a gate fails")
    diff_cmd.add_argument("base", help="baseline run file (e.g. committed on main)")
    diff_cmd.add_argument("head", help="new run file")
    diff_cmd.add_argument("--fail-on-drop", type=float, metavar="FRACTION",
                          help="fail when a scorer's pass rate drops by more than this (0.02 = 2 points)")
    diff_cmd.add_argument("--scorer", action="append", metavar="NAME",
                          help="only gate on this scorer (repeatable; default: all shared scorers)")
    diff_cmd.add_argument("--max-cost-increase", type=float, metavar="FRACTION",
                          help="fail when cost per case rises by more than this (0.2 = 20%%)")
    diff_cmd.add_argument("--format", choices=["text", "markdown"], default="text",
                          help="markdown suits $GITHUB_STEP_SUMMARY and PR comments")

    args = parser.parse_args(argv)
    console = Console()
    try:
        if args.command == "run":
            return _run(args, console)
        if args.command == "diff":
            return _diff(args, console)
        if args.command == "html":
            return _html(args, console)
        return _show(args, console)
    except UsageError as exc:
        print(f"evalbar: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, console: Console) -> int:
    load_env_file(args.env_file)
    evaluation = load_eval(args.target)
    total = min(len(evaluation.dataset), args.limit) if args.limit is not None else len(evaluation.dataset)
    progress = {"done": 0}

    def on_case(result: CaseResult) -> None:
        progress["done"] += 1
        if args.quiet:
            return
        failed = [score.name for score in result.scores if not score.passed]
        status = "ERROR" if result.error else ("fail: " + ", ".join(failed) if failed else "ok")
        print(f"[{progress['done']}/{total}] {result.case_id} {result.latency_s:.2f}s {status}", file=sys.stderr)

    result = run(evaluation, limit=args.limit, on_case=on_case)
    out = args.out or f"runs/{evaluation.name}-{datetime.now():%Y%m%d-%H%M%S}.json"
    path = save_run(result, out)
    console.print(report_renderable(result))
    console.print(f"run file: {path}", style="dim")
    return 0


def _show(args: argparse.Namespace, console: Console) -> int:
    path = Path(args.run_file)
    if not path.is_file():
        raise UsageError(f"run file not found: {path}")
    console.print(report_renderable(load_run(path)))
    return 0


def _html(args: argparse.Namespace, console: Console) -> int:
    path = Path(args.run_file)
    if not path.is_file():
        raise UsageError(f"run file not found: {path}")
    out = Path(args.out) if args.out else path.with_suffix(".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(load_run(path)), encoding="utf-8")
    console.print(f"wrote {out}")
    return 0


def _diff(args: argparse.Namespace, console: Console) -> int:
    runs = []
    for run_file in (args.base, args.head):
        path = Path(run_file)
        if not path.is_file():
            raise UsageError(f"run file not found: {path}")
        runs.append(load_run(path))
    diff = diff_runs(*runs)
    try:
        failures = check_gates(diff, fail_on_drop=args.fail_on_drop, scorers=args.scorer,
                               max_cost_increase=args.max_cost_increase)
    except ValueError as exc:
        raise UsageError(str(exc)) from exc

    if args.format == "markdown":
        print(render_markdown(diff, failures), end="")
    else:
        console.print(diff_renderable(diff))
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


def load_env_file(path: str) -> None:
    """Set variables from a KEY=VALUE file without overriding the environment. Values are never printed."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def load_eval(target: str) -> Eval:
    """Import a Python file and return the Eval named after ':' (or its only Eval)."""
    file_part, _, name = target.partition(":") if target.count(":") == 1 else (target, "", "")
    path = Path(file_part)
    if not path.is_file():
        raise UsageError(f"eval file not found: {path}")

    module_name = f"evalbar_user_{abs(hash(path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise UsageError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    evals = {key: value for key, value in vars(module).items() if isinstance(value, Eval)}
    if name:
        if name not in evals:
            raise UsageError(f"{path} has no Eval named '{name}' (found: {', '.join(evals) or 'none'})")
        return evals[name]
    if len(evals) != 1:
        found = ", ".join(evals) or "none"
        raise UsageError(f"{path} defines {len(evals)} Evals ({found}); pick one with {path}:<name>")
    return next(iter(evals.values()))


if __name__ == "__main__":
    raise SystemExit(main())
