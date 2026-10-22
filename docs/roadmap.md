# evalbar roadmap (M1–M4)

Budget: 15–20 h/week, alongside mid-sem recovery (which wins any clash). Goal: résumé-ready and public before mid-November 2026, ahead of placement-cell companies (Nov–Dec).

"Résumé-ready" means: public repo, green CI, a README with real measured numbers, and a released 0.1.0 on PyPI. M2 alone is already enough to put on a résumé; M3 and M4 make it stronger.

| Milestone | Window | Hours | Done when |
|---|---|---|---|
| M1 Core harness ✅ 2026-10-22 | Oct 22 – Oct 28 | ~12 (built 2026-10-22; Milind reads and owns it) | Dataset loader, runner with tracing, 4 scorers, terminal report, JSON run file, CLI `run`/`show`; FeedbackOS adapter run end to end; README has real numbers; pytest green. |
| M2 Diff + CI gate ✅ 2026-10-22 (CI not yet run on GitHub: repo not pushed) | Oct 29 – Nov 4 | ~15 | `evalbar diff a.json b.json` shows per-scorer deltas, cost/latency deltas and flipped case ids; `--fail-on-drop` exits 1; GitHub Action example (`examples/ci/evalbar-gate.yml`) runs a committed baseline vs a fresh run; evalbar's own CI (pytest + ruff) on GitHub Actions. |
| M3 Judge + RAG + HTML | Nov 5 – Nov 11 | ~18 | `llm_judge(rubric=...)` scorer (LiteLLM, JSON verdict, judge cost traced separately); `recall_at_k(retrieved_field, relevant_field, k)`; `evalbar html run.json` writes a single self-contained HTML report; a tiny RAG example. |
| M4 Release | Nov 12 – Nov 16 | ~12 | 0.1.0 on PyPI (trusted publishing from GitHub Actions); docs pass; FeedbackOS before/after written up as a case study; LiteLLM `custom_pricing` PR opened (rerank/embedding cost paths), linked from the README since evalbar depends on that cost path. |

## Cut order if exams squeeze time

1. HTML report (M3) moves after release; the JSON + terminal report is enough.
2. RAG example shrinks to unit tests for `recall_at_k` only.
3. M4 release slips by one week. Do not cut M2: the CI gate is the résumé line ("blocks PRs that drop accuracy").

## Résumé line (after M2)

> evalbar — open-source Python harness that runs LLM pipelines over labelled test sets, traces tokens/latency/cost per case via LiteLLM, and fails CI when quality drops. Used it to evaluate FeedbackOS's feedback classifier: prompt v2 vs v1 on 50 labelled cases (see README for measured accuracy, cost and latency).

Fill the numbers from the README; never quote numbers that were not measured.

## Explicitly later (not in M1–M4)

Async execution, caching of LLM responses between runs, pytest plugin, dataset versioning, multi-turn conversation cases.
