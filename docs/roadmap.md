# evalbar roadmap (M1–M4)

Budget: 15–20 h/week, alongside mid-sem recovery (which wins any clash). Goal: résumé-ready and public before mid-November 2026, ahead of placement-cell companies (Nov–Dec).

"Résumé-ready" means: public repo, green CI, a README with real measured numbers, and a released 0.1.0 on PyPI. M2 alone is already enough to put on a résumé; M3 and M4 make it stronger.

| Milestone | Window | Hours | Done when |
|---|---|---|---|
| M1 Core harness ✅ 2026-10-22 | Oct 22 – Oct 28 | ~12 (built 2026-10-22; Milind reads and owns it) | Dataset loader, runner with tracing, 4 scorers, terminal report, JSON run file, CLI `run`/`show`; FeedbackOS adapter run end to end; README has real numbers; pytest green. |
| M2 Diff + CI gate ✅ 2026-10-22 | Oct 29 – Nov 4 | ~15 | `evalbar diff a.json b.json` shows per-scorer deltas, cost/latency deltas and flipped case ids; `--fail-on-drop` exits 1; GitHub Action example (`examples/ci/evalbar-gate.yml`) runs a committed baseline vs a fresh run; evalbar's own CI (pytest + ruff) on GitHub Actions. |
| M3 Judge + RAG + HTML ✅ 2026-10-23 | Nov 5 – Nov 11 | ~18 | `llm_judge(rubric=...)` scorer (LiteLLM, JSON verdict, judge cost traced separately); `recall_at_k(retrieved_field, relevant_field, k)`; `evalbar html run.json` writes a single self-contained HTML report; a tiny RAG example. |
| M4 Release ✅ 0.1.0 on PyPI and case study 2026-10-23 (LiteLLM PR in progress) | Nov 12 – Nov 16 | ~12 | 0.1.0 on PyPI (trusted publishing from GitHub Actions); docs pass; FeedbackOS before/after written up as a case study; LiteLLM `custom_pricing` PR opened (rerank/embedding cost paths), linked from the README since evalbar depends on that cost path. |

## Cut order if exams squeeze time

1. HTML report (M3) moves after release; the JSON + terminal report is enough.
2. RAG example shrinks to unit tests for `recall_at_k` only.
3. M4 release slips by one week. Do not cut M2: the CI gate is the résumé line ("blocks PRs that drop accuracy").

## Résumé line

> **evalbar** — open-source Python LLM eval and tracing harness ([PyPI](https://pypi.org/project/evalbar/), [GitHub](https://github.com/MilindDevX/evalbar)). Runs any pipeline over a labelled test set, traces tokens, latency and cost per case via LiteLLM, scores outputs (exact, regex, JSON schema, recall@k, LLM-as-judge), and fails CI when quality drops or cost rises. Used it to evaluate FeedbackOS's production feedback classifier: the new prompt raised sentiment accuracy from 94% to 100% and kept praise in 8/8 mixed-feedback cases (was 0/8), at +13% cost per item, on 50 hand-labelled synthetic cases; it also surfaced a product-area bug in FeedbackOS.

Shorter, for a one-line bullet:

> Built and published evalbar (Python, LiteLLM, PyPI): an LLM eval harness with per-case cost/latency tracing and a CI quality gate; measured a production prompt change at +6 pts sentiment accuracy for +13% cost.

All numbers come from the measured runs in the [case study](case-study-feedbackos.md). Keep "50 synthetic cases" when quoting them; they are not production accuracy.

## Explicitly later (not in M1–M4)

Async execution, caching of LLM responses between runs, pytest plugin, dataset versioning, multi-turn conversation cases.
