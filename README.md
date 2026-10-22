# evalbar

Small eval and tracing harness for LLM pipelines. Point it at any Python function, give it a file of labelled test cases, and get a scored run with a trace per case: LLM calls, prompt and completion tokens, latency and cost. Save the run as JSON so the next change can be compared against it.

- **Any callable**: an LLM call, a RAG (retrieval-augmented generation) chain or an agent; evalbar only sees `pipeline(input) -> output`.
- **Plain datasets**: JSONL (one JSON object per line) or YAML.
- **Traces**: tokens, latency and list-price cost per case through [LiteLLM](https://github.com/BerriAI/litellm), so any provider works; tool calls via a decorator.
- **Deterministic scorers first**: exact match, contains, regex, JSON-schema validity. LLM-as-judge and retrieval recall@k come in M3.
- **Diffs and CI gates**: `evalbar diff base.json head.json --fail-on-drop 0.02` names the regressed cases and exits 1 when quality drops (or cost rises past `--max-cost-increase`).
- **Honest numbers**: a crashed case fails every scorer it applies to instead of disappearing; an unknown price is reported as unknown, never as $0.

Status: M1 (core harness) and M2 (diff + CI gate) of the [roadmap](docs/roadmap.md) are done; not yet on PyPI. Design and comparison with promptfoo, DeepEval and Ragas: [docs/design.md](docs/design.md).

## Quickstart

```bash
git clone https://github.com/MilindDevX/evalbar && cd evalbar
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/evalbar run examples/quickstart/eval.py
```

The quickstart needs no API key (LiteLLM's `mock_response` stands in for the model) and prints a score table, a usage table and the failing case with its reason. The run is saved under `runs/`; re-print it with `evalbar show runs/<file>.json`.

## Write an eval

```python
# my_eval.py
import json

from evalbar import Eval, completion, load_dataset, traced_tool
from evalbar.scorers import exact, json_schema, regex

@traced_tool                       # recorded in the case trace: args, result, latency, errors
def search(query: str) -> list[str]:
    ...

def pipeline(case_input: dict) -> dict:
    docs = search(case_input["question"])
    reply = completion(            # litellm.completion + tokens/latency/cost in the trace
        model="groq/openai/gpt-oss-120b",
        messages=[{"role": "user", "content": f"{docs}\n\n{case_input['question']}"}],
        response_format={"type": "json_object"},
    )
    return json.loads(reply.choices[0].message.content)

my_eval = Eval(
    name="support-bot",
    pipeline=pipeline,
    dataset=load_dataset("cases.jsonl"),
    scorers=[
        exact("label"),                                      # output["label"] == expected["label"]
        json_schema(SCHEMA),                                 # output validates
        regex("answer", r"^The user", must_match=False),     # style rule
    ],
    min_interval_s=2.0,                                      # simple rate limit
)
```

```bash
evalbar run my_eval.py --out runs/baseline.json   # add :name when the file defines several Evals
evalbar run my_eval.py --limit 3                  # smoke test before spending quota
evalbar show runs/baseline.json
```

A case in `cases.jsonl`:

```json
{"id": "refund-01", "input": {"question": "How do I get a refund?"}, "expected": {"label": "billing"}, "metadata": {"split": "dev"}}
```

A scorer is any `(output, case) -> Score | None` function. Built-ins return `None` when the case has no expectation for them (for example `contains("secondary_themes")` on a case without `expected.secondary_themes`), so every scorer is averaged only over the cases it applies to.

`evalbar run` loads `.env` from the current folder if present (variables already set in the shell win); see `.env.example`.

## Compare runs and gate CI

```bash
evalbar diff runs/baseline.json runs/head.json                       # report only, exit 0
evalbar diff runs/baseline.json runs/head.json --fail-on-drop 0.02   # exit 1 if any pass rate drops > 2 points
evalbar diff runs/baseline.json runs/head.json --fail-on-drop 0 --scorer exact:label --max-cost-increase 0.2
evalbar diff runs/baseline.json runs/head.json --format markdown >> "$GITHUB_STEP_SUMMARY"
```

The diff compares only the cases both runs share, so adding or removing cases cannot fake a gain. For each scorer it shows the pass rate before and after and the ids of cases that regressed or got fixed (a case that newly crashes counts as a regression). It also shows cost, tokens and LLM latency per case.

Reversing the FeedbackOS comparison below (production prompt as the baseline, old prompt as the change) is a realistic regression, and the gate catches it:

```text
$ evalbar diff runs/feedbackos-v2.json runs/feedbackos-v1.json --fail-on-drop 0.02
FAIL: exact:sentiment dropped 6.0 pts (100.0% -> 94.0%), allowed 2.0; regressed: mixed-01, mixed-02, mixed-03
FAIL: theme+sentiment dropped 4.0 pts (96.0% -> 92.0%), allowed 2.0; regressed: mixed-01, mixed-02
FAIL: contains:secondary_themes dropped 100.0 pts (100.0% -> 0.0%), allowed 2.0; regressed: mixed-01, ..., mixed-08
```

Forward (v1 to v2), the quality gate passes, but `--max-cost-increase 0.10` fails because v2 costs 13% more per item. The job surfaces the trade-off instead of hiding it.

GitHub Actions:

- [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs this repo's tests on Python 3.10–3.13. It also dogfoods the gate: it runs the offline quickstart and diffs it against the committed [`examples/quickstart/baseline.json`](examples/quickstart/baseline.json), so no API key is needed.
- [`examples/ci/evalbar-gate.yml`](examples/ci/evalbar-gate.yml) is a template to copy into your project. It runs your eval against a real model on pull requests, compares it with a committed baseline, writes the table to the job summary and uploads the run file.

## First real user: FeedbackOS's feedback classifier

FeedbackOS (Milind's feedback-triage SaaS) classifies customer feedback into a theme (`bug_problem`, `feature_request`, `usability_issue`, `question_support`, `praise`, `other`), a sentiment and a product area with `openai/gpt-oss-120b` on Groq. On 2026-10-12 its prompt changed: v2 added `secondary_themes` (praise inside a complaint is kept) and a stricter "mixed" sentiment rule.

[`examples/feedbackos/`](examples/feedbackos) reruns both prompt versions with the production settings:

- `prompts/v1.txt`, `prompts/v2.txt`: system prompts copied verbatim from FeedbackOS's git history (v1 = `9a896d9`, v2 = `59e4902`, the production prompt).
- `adapter.py`: a Python port of `classifyFeedback` (same user prompt, model, `temperature=0`, 512 max tokens, JSON mode, `reasoning_effort="low"`, short-text guard, product-area fallback).
- `cases.jsonl`: 50 synthetic, hand-labelled cases (8 bug, 8 feature request, 7 usability, 7 question, 7 praise, 5 other, 8 mixed praise+problem). No real user data.

```bash
cp .env.example .env   # then put your GROQ_API_KEY in .env with an editor
evalbar run examples/feedbackos/eval.py:v1 --out runs/feedbackos-v1.json
evalbar run examples/feedbackos/eval.py:v2 --out runs/feedbackos-v2.json
```

### Results

Measured 2026-10-22 on Groq, one run per prompt, 50 cases each, 49 LLM calls each (one case is under 5 words and skips the model, as in production). evalbar 0.1.0.dev0, LiteLLM 1.106.2. Raw run files are not committed; rerun both evals, then `evalbar diff runs/feedbackos-v1.json runs/feedbackos-v2.json --format markdown` regenerates the comparison.

| metric | v1 (single theme) | v2 (production) | change |
|---|---|---|---|
| theme accuracy (n=50) | 48/50 (96%) | 48/50 (96%) | 0 |
| sentiment accuracy (n=50) | 47/50 (94%) | **50/50 (100%)** | +6 pts |
| theme and sentiment both right (n=50) | 46/50 (92%) | **48/50 (96%)** | +4 pts |
| product-area accuracy (n=50) | 48/50 (96%) | 48/50 (96%) | 0 |
| praise kept in mixed feedback (n=8) | 0/8 | **8/8** | +8 cases |
| valid output schema | 50/50 | 50/50 | 0 |
| summary at most 15 words | 50/50 | 50/50 | 0 |
| prompt tokens per case | 778 | 929 | +19% |
| completion tokens per case | 107 | 118 | +10% |
| cost per 1,000 items (Groq list price) | $0.16 | $0.18 | +13% |
| LLM latency p50 / p95 | 0.90s / 1.83s | 0.80s / 1.63s | within noise |

What changed and what did not:

- **All of v2's gain is on mixed feedback** (praise next to a problem or request). On the 42 single-theme cases both prompts score the same. On the 8 mixed cases, v1 never keeps the praise (it has no `secondary_themes` field) and gets the sentiment wrong 3 times: `mixed-01` negative, `mixed-02` positive, `mixed-03` negative, all expected mixed. v2 gets all 8 right.
- **The price is a longer prompt**: +19% prompt tokens, +13% cost per item. On Groq's free tier the actual spend is $0; the list price for both runs together was $0.017 (96,591 tokens).
- **Shared misses** (both prompts, so not caused by the change):
  - `other-01`: a SOC 2 / vendor-questionnaire request is classified `question_support`, although the prompt's own boundary rule says administrative requests are `other`. A real prompt-following gap.
  - `other-05`: "ok see attached" (under 5 words) skips the model and gets product area `Dashboard`, because FeedbackOS's short-text guard uses the first product area instead of the `Other` fallback. A real FeedbackOS bug, found by this eval.
  - `feat-03` (webhooks: `API` vs `Integrations`) and `mixed-03` (a broken "Remember me" box: `bug_problem` vs `usability_issue`) are arguably label ambiguity rather than model errors.

Caveats: 50 synthetic cases written by the evalbar author and labelled to the v2 contract, so one case is 2 points and these numbers say nothing about production accuracy. `praise kept` measures a field v1 cannot produce, so it shows the contract change, not v1 failing at its own job. Each prompt was run once; `gpt-oss-120b` at temperature 0 is not guaranteed deterministic.

What the CLI prints for v2 (trimmed):

```text
evalbar run: feedbackos-v2  50 cases, 0 errors, started 2026-10-22T13:10:02+00:00
Scores
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┓
┃ scorer                    ┃ cases ┃ passed ┃ pass rate ┃  mean ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━┩
│ exact:theme               │    50 │     48 │     96.0% │ 0.960 │
│ exact:sentiment           │    50 │     50 │    100.0% │ 1.000 │
│ exact:product_area        │    50 │     48 │     96.0% │ 0.960 │
│ theme+sentiment           │    50 │     48 │     96.0% │ 0.960 │
│ contains:secondary_themes │     8 │      8 │    100.0% │ 1.000 │
│ valid_output              │    50 │     50 │    100.0% │ 1.000 │
│ summary<=15 words         │    50 │     50 │    100.0% │ 1.000 │
└───────────────────────────┴───────┴────────┴───────────┴───────┘
Usage
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ metric                 ┃         value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ LLM calls              │            49 │
│ prompt tokens          │        46,454 │
│ completion tokens      │         5,887 │
│ cost (list price)      │     $0.009233 │
│ cost per case          │     $0.000185 │
│ case latency p50 / p95 │ 0.80s / 1.62s │
│ LLM latency p50 / p95  │ 0.80s / 1.63s │
└────────────────────────┴───────────────┘
```

## Development

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

Costs come from the price LiteLLM computes during the call (`response_cost`), because `completion_cost()` on a Groq response looks the model up under OpenAI and fails; see [design decision 4](docs/design.md#design-decisions).

Tests run offline: LiteLLM's `mock_response` exercises the real `completion()` path, and LiteLLM's bundled price table is used (`LITELLM_LOCAL_MODEL_COST_MAP=True`) so costs are reproducible and tied to the LiteLLM version recorded in each run file.

LiteLLM 1.82.7 and 1.82.8 were compromised releases on PyPI (March 2026); the dependency range excludes them.

## License

MIT
