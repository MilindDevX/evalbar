"""A single self-contained HTML page for a run: no external assets, no scripts.

Everything that came from the pipeline, the dataset or a model is escaped, because LLM
output is untrusted text.
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from .report import format_cost, summarize
from .runner import CaseResult, RunResult

_CSS = """
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --ink2:#5f6368; --line:#e3e3df;
  --ok:#1a7f37; --ok-bg:#e6f4ea; --bad:#b42318; --bad-bg:#fdecea; --warn:#9a6700; --warn-bg:#fff4d6;
  --bar:#2f6fde; --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
@media (prefers-color-scheme: dark) { :root { --bg:#141416; --card:#1d1d20; --ink:#ececf1; --ink2:#a1a1aa;
  --line:#2e2e33; --ok:#4ac26b; --ok-bg:#12261a; --bad:#ff7b72; --bad-bg:#2d1414; --warn:#e3b341;
  --warn-bg:#2b230f; --bar:#6c9ef8; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 16px; margin: 28px 0 10px; }
.muted { color: var(--ink2); font-size: 13px; }
.chips span { display:inline-block; margin:6px 6px 0 0; padding:2px 8px; border:1px solid var(--line);
  border-radius:999px; font-size:12px; color:var(--ink2); }
.cards { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap:10px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px; }
.card strong { display:block; font-size:20px; } .card small { color:var(--ink2); }
.tw { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; font-size:14px; }
tr:last-child > td { border-bottom:0; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--ink2); font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.03em; }
td.num, th.num { text-align:right; font-variant-numeric: tabular-nums; }
.bar { height:6px; background:var(--line); border-radius:3px; min-width:80px; }
.bar i { display:block; height:6px; background:var(--bar); border-radius:3px; }
details.case { background:var(--card); border:1px solid var(--line); border-radius:10px; margin:8px 0; }
details.case > summary { cursor:pointer; padding:10px 12px; display:flex; gap:10px; align-items:baseline;
  flex-wrap:wrap; }
details.case > div { padding:0 12px 12px; }
.badge { font-size:12px; font-weight:600; padding:1px 8px; border-radius:6px; }
.pass .badge { color:var(--ok); background:var(--ok-bg); } .fail .badge { color:var(--bad); background:var(--bad-bg); }
.error .badge { color:var(--warn); background:var(--warn-bg); }
.cid { font-family:var(--mono); font-weight:600; }
pre { font-family:var(--mono); font-size:12.5px; background:var(--bg); border:1px solid var(--line);
  border-radius:8px; padding:8px 10px; white-space:pre-wrap; word-break:break-word; margin:4px 0 10px; }
.grid2 { display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:0 16px; }
.label { font-size:12px; color:var(--ink2); text-transform:uppercase; letter-spacing:.03em; margin-top:8px; }
.ok { color:var(--ok); } .bad { color:var(--bad); }
"""


def render_html(run: RunResult) -> str:
    summary = summarize(run)
    head = (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>evalbar: {_e(run.name)}</title><style>{_CSS}</style></head><body><main>")

    chips = "".join(f"<span>{_e(key)}: {_e(value)}</span>"
                    for key, value in {**run.metadata, **run.environment}.items())
    header = (f"<h1>{_e(run.name)}</h1><div class=\"muted\">{summary.n_cases} cases, {summary.n_errors} errors, "
              f"started {_e(run.started_at)}, finished {_e(run.finished_at)}</div><div class=\"chips\">{chips}</div>")

    per_case = summary.n_cases or 1
    cards = [
        ("cost (list price)", format_cost(summary.cost_usd), f"{format_cost(summary.cost_usd / per_case)} per case"),
        ("tokens", f"{summary.prompt_tokens + summary.completion_tokens:,}",
         f"{summary.prompt_tokens:,} prompt / {summary.completion_tokens:,} completion"),
        ("LLM latency p50", f"{summary.llm_latency_p50_s:.2f}s", f"p95 {summary.llm_latency_p95_s:.2f}s"),
        ("errors", str(summary.n_errors), f"of {summary.n_cases} cases"),
    ]
    if summary.judge_calls:
        cards.append(("judge cost", format_cost(summary.judge_cost_usd), f"{summary.judge_calls} calls, not in cost"))
    if summary.n_unpriced_cases:
        cards.append(("unpriced cases", str(summary.n_unpriced_cases), "cost excludes them"))
    cards_html = "".join(f'<div class="card"><small>{_e(title)}</small><strong>{_e(value)}</strong>'
                         f"<small>{_e(note)}</small></div>" for title, value, note in cards)

    score_rows = "".join(
        f"<tr><td>{_e(name)}</td><td class=\"num\">{s.n}</td><td class=\"num\">{s.passed}</td>"
        f"<td class=\"num\">{s.pass_rate:.1%}</td><td><div class=\"bar\"><i style=\"width:{s.pass_rate * 100:.1f}%\">"
        f"</i></div></td><td class=\"num\">{s.mean:.3f}</td></tr>"
        for name, s in summary.scorers.items()
    )
    scores_html = ('<h2>Scores</h2><div class="tw"><table><tr><th>scorer</th><th class="num">cases</th>'
                   '<th class="num">passed</th><th class="num">pass rate</th><th></th>'
                   '<th class="num">mean</th></tr>'
                   f"{score_rows}</table></div>")

    ordered = sorted(run.cases, key=_case_rank)
    cases_html = "<h2>Cases (failures first)</h2>" + "".join(_case(case) for case in ordered)
    return head + header + f'<h2>Usage</h2><div class="cards">{cards_html}</div>' + scores_html + cases_html + \
        "</main></body></html>\n"


def _case_rank(case: CaseResult) -> int:
    if case.error is not None:
        return 0
    return 1 if any(not score.passed for score in case.scores) else 2


def _case(case: CaseResult) -> str:
    status = ["error", "fail", "pass"][_case_rank(case)]
    failed = [score.name for score in case.scores if not score.passed]
    cost = case.trace.cost_usd
    summary_line = (f'<span class="badge">{status}</span><span class="cid">{_e(case.case_id)}</span>'
                    f'<span class="muted">{case.latency_s:.2f}s · '
                    f'{"unpriced" if cost is None else format_cost(cost)}</span>'
                    + (f'<span class="bad">{_e(", ".join(failed))}</span>' if failed else ""))

    body = ['<div class="grid2">',
            f'<div><div class="label">input</div><pre>{_e(_pretty(case.input))}</pre></div>',
            f'<div><div class="label">expected</div><pre>{_e(_pretty(case.expected))}</pre></div></div>',
            f'<div class="label">output</div><pre>{_e(_pretty(case.output))}</pre>']
    if case.error:
        body.append(f'<div class="label">error</div><pre class="bad">{_e(case.error)}</pre>')
    if case.metadata:
        body.append(f'<div class="label">metadata</div><pre>{_e(_pretty(case.metadata))}</pre>')
    if case.scores:
        rows = "".join(f'<tr><td>{_e(s.name)}</td><td class="{"ok" if s.passed else "bad"}">'
                       f'{"pass" if s.passed else "fail"}</td><td class="num">{s.value:.2f}</td>'
                       f"<td>{_e(s.reason)}</td></tr>" for s in case.scores)
        body.append('<div class="label">scores</div><div class="tw"><table><tr><th>scorer</th><th>result</th>'
                    f'<th class="num">value</th><th>reason</th></tr>{rows}</table></div>')
    if case.trace.llm_calls:
        rows = "".join(f"<tr><td>{_e(c.role)}</td><td>{_e(c.model)}</td><td class=\"num\">{c.prompt_tokens:,}</td>"
                       f"<td class=\"num\">{c.completion_tokens:,}</td><td class=\"num\">{c.latency_s:.2f}s</td>"
                       f"<td class=\"num\">{'unpriced' if c.cost_usd is None else format_cost(c.cost_usd)}</td></tr>"
                       for c in case.trace.llm_calls)
        body.append('<div class="label">LLM calls</div><div class="tw"><table><tr><th>role</th><th>model</th>'
                    '<th class="num">prompt</th><th class="num">completion</th><th class="num">latency</th>'
                    f'<th class="num">cost</th></tr>{rows}</table></div>')
    if case.trace.tool_calls:
        rows = "".join(f"<tr><td>{_e(t.name)}</td><td><pre>{_e(_pretty(t.args))}</pre></td>"
                       f"<td><pre>{_e(t.error or _pretty(t.result))}</pre></td>"
                       f"<td class=\"num\">{t.latency_s:.3f}s</td></tr>" for t in case.trace.tool_calls)
        body.append('<div class="label">tool calls</div><div class="tw"><table><tr><th>tool</th><th>args</th>'
                    f'<th>result</th><th class="num">latency</th></tr>{rows}</table></div>')

    return (f'<details class="case {status}" id="case-{_e(case.case_id)}"><summary>{summary_line}</summary>'
            f'<div>{"".join(body)}</div></details>')


def _pretty(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=repr)


def _e(value: Any) -> str:
    return escape(str(value), quote=True)
