# evalbar roadmap

## Shipped in 0.1.0

| Milestone | Shipped | What it added |
|---|---|---|
| M1 Core harness | 2026-10-22 | Dataset loader (JSONL, YAML), runner with per-case tracing, the exact, contains, regex and JSON-schema scorers, terminal report, JSON run file, `evalbar run` and `evalbar show`. |
| M2 Diff and CI gate | 2026-10-22 | `evalbar diff a.json b.json` with per-scorer, cost and latency changes and the ids of cases that flipped; `--fail-on-drop` and `--max-cost-increase` exit 1; a GitHub Actions template in `examples/ci/evalbar-gate.yml`; this repo's own CI. |
| M3 Judge, retrieval, HTML | 2026-10-23 | `llm_judge` (JSON verdict, judge cost traced apart from the pipeline's), `recall_at_k`, `evalbar html` for a single self-contained report, and the RAG example. |
| M4 Release | 2026-10-23 | 0.1.0 on PyPI through trusted publishing, and the [FeedbackOS case study](case-study-feedbackos.md). |

## Later

Not scheduled:

- Async execution
- Caching LLM responses between runs
- A pytest plugin
- Dataset versioning
- Multi-turn conversation cases
