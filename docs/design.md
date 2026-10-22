# evalbar design

Status: draft, 2026-10-22. Owner: Milind Bansal (MilindDevX).

## One-line pitch

Point evalbar at any Python function that calls an LLM, give it a file of test cases, and get back a scored, traced run (tokens, latency, cost per case) that you can compare against the last run and gate CI on.

## Problem

Teams shipping LLM features change prompts and models often. Each change can silently make outputs worse, slower or more expensive. The questions after every change are always the same:

1. Did quality go up or down, and on which cases?
2. What did it cost, and how slow was it?
3. Can CI (continuous integration, the automated checks that run on every pull request) block a change that makes quality drop?

Existing tools answer these, but they are large frameworks with their own config languages, platforms or metric zoos. evalbar answers only these three questions, from plain Python, with plain JSON output.

## Goals

- **Any callable.** The system under test is a Python function `pipeline(input) -> output`. An LLM call, a RAG (retrieval-augmented generation: fetch documents, then ask the model with them) chain or an agent loop all fit.
- **Plain datasets.** Test cases in JSONL (one JSON object per line) or YAML. Each case has `id`, `input`, `expected`, optional `metadata`.
- **Trace per case.** Every case records its LLM calls (model, prompt and completion tokens, latency, cost) and tool calls (name, arguments, result, latency).
- **Provider-agnostic calls and cost.** LLM calls go through LiteLLM, which speaks to 100+ providers with one API and has a maintained price table, so cost comes from the same place as the call.
- **Small, composable scorers.** A scorer is a function `(output, case) -> Score`. Built-in: exact match, contains, regex, JSON-schema validity (M1); LLM-as-judge with a rubric and retrieval recall@k (M3).
- **Reports for people and machines.** A terminal table, a JSON run file (the source of truth), and later an HTML page.
- **Diffs and CI gates.** Compare two run files; exit non-zero when a metric drops more than a threshold. Ship a GitHub Action example.

## Non-goals

- Red-teaming, jailbreak or safety scanning (promptfoo does this well).
- A hosted platform, dashboard server, user accounts or a database.
- Production observability of live traffic (Langfuse, Arize Phoenix and LangSmith exist for that). evalbar traces offline test runs only.
- Prompt management or a prompt templating language.
- Synthetic dataset generation.
- A large catalogue of LLM-judged metrics. One rubric judge, written clearly, beats twenty opaque ones.
- Distributed or async-first execution. Sequential with optional thread concurrency and rate limiting is enough for datasets of hundreds of cases.

## Core API sketch

```python
from evalbar import Eval, load_dataset, completion, tool_call
from evalbar.scorers import exact, contains, regex, json_schema

def pipeline(case_input: dict) -> dict:
    # Any code. LLM calls made through evalbar.completion() are traced automatically.
    response = completion(model="groq/openai/gpt-oss-120b", messages=[...])
    return json.loads(response.choices[0].message.content)

feedback_eval = Eval(
    name="classifier-v2",
    pipeline=pipeline,
    dataset=load_dataset("cases.jsonl"),
    scorers=[
        exact("theme"),                    # output["theme"] == expected["theme"]
        contains("secondary_themes"),      # expected items appear in output list
        regex("summary", r"^\S+(\s+\S+){0,14}$", name="summary<=15 words"),
        json_schema(SCHEMA),               # whole output validates
    ],
    min_interval_s=8.0,                    # crude rate limit for free-tier APIs
)
```

```bash
evalbar run examples/feedbackos/eval.py:v2 --out runs/v2.json   # run + terminal report + JSON
evalbar show runs/v2.json                                        # re-print a saved report
evalbar diff runs/v1.json runs/v2.json --fail-on-drop 0.02       # M2: compare and gate
```

Building blocks:

| Module | Responsibility |
|---|---|
| `dataset` | Load JSONL/YAML into `Case(id, input, expected, metadata)`; reject duplicate or missing ids with the line number. |
| `trace` | A per-case `Trace` held in a context variable (a value that follows the current call stack, so nested code finds the active trace without passing it around). `record_llm_call()` and `tool_call()` append spans. |
| `llm` | `completion()` wraps `litellm.completion`, times it, reads token usage and asks LiteLLM for the cost, then records a span. Non-LiteLLM clients can call `record_llm_call()` directly. |
| `scorers` | Scorer factories returning `Score(name, value 0..1, passed, reason)`. A scorer returns `None` when the case has no expectation for it, so per-scorer counts are honest. |
| `runner` | Runs the pipeline per case inside a fresh trace; captures exceptions as case errors (an error fails every scorer rather than disappearing); applies rate limiting. Returns a `RunResult`. |
| `report` | Aggregates: mean score and pass rate per scorer, error count, total tokens and cost, latency p50/p95. Renders a `rich` terminal table; writes and reads the JSON run file. |
| `cli` | `run`, `show` (M1), `diff` (M2), `html` (M3). |
| `html` | One self-contained page per run: summary, scores, every case with input, expected, output, scores and trace. Everything escaped, no scripts, no external assets. |

The JSON run file is versioned (`"schema_version": 1`) because `diff` and CI depend on it staying readable across releases.

## Design decisions

1. **Explicit tracing wrapper over LiteLLM callbacks.** LiteLLM can call a global logger after each request, but in sync mode that logger may run on another thread, where the per-case context variable is not visible. That would attach spans to the wrong case under concurrency. A thin `completion()` wrapper is explicit and correct; `record_llm_call()` covers other SDKs.
2. **Scores are numbers in 0..1 plus a pass flag.** Deterministic scorers return 0 or 1; a judge can return a fraction. Aggregation is the same for both.
3. **Errors are first-class.** A crashed case counts as a failure for every scorer and is listed in the report. Silently dropping failed cases inflates accuracy, which is the most common way eval numbers lie.
4. **Cost comes from the price LiteLLM computed at call time.** `litellm.completion_cost(response)` re-derives the provider from the response, and that can go wrong. Found on 2026-10-22: Groq returns `model="openai/gpt-oss-120b"` with no provider hint, so the lookup goes to OpenAI and raises `ModelNotMappedError`, even when the full `groq/...` model name is passed. evalbar reads `response._hidden_params["response_cost"]` first (set while LiteLLM still knew the provider) and only then falls back to `completion_cost()`. This is a candidate upstream LiteLLM issue next to the planned `custom_pricing` PR.
5. **Judge calls are tagged, not mixed in.** An LLM-as-judge scorer makes LLM calls during scoring. The runner re-opens the case trace with role `judge` while scorers run, so those calls are kept and reported (judge cost, judge calls) but never added to the pipeline's cost or tokens. Otherwise a stricter judge would make the system under test look more expensive.
6. **Cost is "list price".** Cost uses LiteLLM's price table even when the account is on a free tier, so two runs are comparable in money terms. The report says when a model has no known price instead of printing 0.
7. **No config language.** The eval is a Python file. Python already has variables, imports and functions; YAML configs end up reinventing them.

## How it compares

| | evalbar | promptfoo | DeepEval | Ragas |
|---|---|---|---|---|
| Language / config | Python file | Node.js CLI, YAML config | Python, pytest-style | Python library |
| System under test | any Python callable | prompts × providers matrix, or custom provider scripts | any callable via test cases | RAG pipelines (question, contexts, answer) |
| Main metrics | deterministic first; one rubric judge; recall@k | many assertions incl. LLM-graded; red-teaming | large catalogue of LLM-judged metrics | RAG metrics (faithfulness, context precision/recall), mostly LLM-judged |
| Per-case trace (tokens, latency, cost, tool calls) | core feature, saved in the run file | latency/cost per test, no tool-call trace | via its tracing + hosted platform | not the focus |
| Run diff + CI gate | core feature (M2) | yes, plus a hosted web viewer | yes, plus Confident AI platform | no built-in gate |
| Platform | none | optional cloud | Confident AI (optional) | optional |
| Size | target < 2,000 lines | large | large | medium |

**Why evalbar is not a clone.** promptfoo is the closest in spirit (CLI, assertions, CI), but it is YAML-and-Node-first, so a Python team ends up writing glue scripts. DeepEval and Ragas are metric libraries whose headline metrics are themselves LLM calls, which adds cost and noise to every run. evalbar's bet is the opposite: start from cheap deterministic checks on structured output, make the trace and cost of every case visible, and make "did this change make things worse?" a single CI exit code. It is small enough to read in an afternoon, which is the feature.

## Risks

- **LiteLLM supply chain.** LiteLLM releases 1.82.7 and 1.82.8 on PyPI were compromised in March 2026. evalbar excludes them in its dependency range and keeps the lower bound on a known-good release. Revisit on every bump.
- **Price table drift.** Provider prices change; LiteLLM updates its table, but an old install reports old prices. The run file records the LiteLLM version so numbers can be explained later.
- **Small datasets overstate differences.** With 50 cases one flipped case is 2 points. The diff report (M2) will show flipped case ids, not only percentages, and README numbers state the sample size.
