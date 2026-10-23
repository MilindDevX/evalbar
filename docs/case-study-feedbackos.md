# Case study: did FeedbackOS's prompt change make the classifier better?

Measured 2026-10-22 with evalbar 0.1.0.dev0 and LiteLLM 1.106.2. Everything here can be rerun from [`examples/feedbackos/`](../examples/feedbackos).

## The question

FeedbackOS is a feedback-triage app. Every piece of customer feedback goes through an LLM classifier that returns:

- a **theme**, one of six: `bug_problem`, `feature_request`, `usability_issue`, `question_support`, `praise`, `other`;
- a **sentiment**: `positive`, `negative`, `neutral` or `mixed`;
- a **product area** from the workspace's list;
- a one-sentence **summary** and a human-review flag.

The model is `openai/gpt-oss-120b` on Groq, at temperature 0 (the most repeatable setting), in JSON mode (the API forces a JSON object), with low reasoning effort.

On 2026-10-12 the system prompt changed. Users had reported that mixed feedback such as "the UI is good but checkout needs work" lost its praise. The old prompt allowed only one theme and pushed factual problems towards `negative`. The new prompt (v2):

- keeps the most actionable part as the theme and adds `secondary_themes`, so the praise survives;
- calls any positive remark next to a problem `mixed`;
- asks the summary to mention both halves.

The change shipped on the strength of a small spot check. This study asks what it actually bought, and what it cost.

## Method

1. **Same call as production.** [`adapter.py`](../examples/feedbackos/adapter.py) is a Python port of FeedbackOS's `classifyFeedback`. It uses the same user prompt, model, temperature, 512-token limit, JSON mode and reasoning effort, plus the same short-text guard (feedback under 5 words skips the model) and the same product-area fallback.
2. **Same prompts as production.** Both system prompts were copied word for word from FeedbackOS's git history: v1 at commit `9a896d9` (the last one before the change) and v2 at `59e4902` (the production prompt).
3. **A labelled test set.** [`cases.jsonl`](../examples/feedbackos/cases.jsonl) has 50 synthetic, hand-labelled cases: 8 bugs, 8 feature requests, 7 usability issues, 7 questions, 7 praise, 5 other, and 8 "mixed" cases that pair praise with a problem or request. Labels follow the v2 contract (the rules in the v2 prompt). No real user data is used.
4. **Scorers** (checks that turn each output into pass/fail):
   - exact match on theme, sentiment and product area;
   - theme and sentiment both right;
   - `praise` present in `secondary_themes` (only the 8 mixed cases carry this expectation);
   - output matches FeedbackOS's JSON schema;
   - summary at most 15 words.
5. **One run per prompt** on Groq's free tier, with cases spaced 9 seconds apart to stay under the 8,000-tokens-per-minute limit. Cost is Groq's list price from LiteLLM's price table.

## Results

| metric | v1 (single theme) | v2 (production) | change |
|---|---|---|---|
| theme accuracy (n=50) | 48/50 (96%) | 48/50 (96%) | 0 |
| sentiment accuracy (n=50) | 47/50 (94%) | 50/50 (100%) | +6 pts |
| theme and sentiment both right (n=50) | 46/50 (92%) | 48/50 (96%) | +4 pts |
| product-area accuracy (n=50) | 48/50 (96%) | 48/50 (96%) | 0 |
| praise kept in mixed feedback (n=8) | 0/8 | 8/8 | +8 cases |
| valid output schema | 50/50 | 50/50 | 0 |
| summary at most 15 words | 50/50 | 50/50 | 0 |
| prompt tokens per case | 778 | 929 | +19% |
| completion tokens per case | 107 | 118 | +10% |
| cost per 1,000 items (list price) | $0.16 | $0.18 | +13% |
| LLM latency p50 / p95 | 0.90s / 1.83s | 0.80s / 1.63s | within noise |

p50 and p95 are the median and the 95th-percentile response time.

**All of v2's gain is on mixed feedback.** On the 42 single-theme cases both prompts score the same. On the 8 mixed cases:

- v1 never keeps the praise: its output has no `secondary_themes` field.
- v1 also gets the sentiment wrong on 3 of them: `mixed-01` comes out `negative`, `mixed-02` `positive` and `mixed-03` `negative`, where all three should be `mixed`.
- v1's summaries drop the praise. For `mixed-01` ("The dashboard looks great, but the weekly chart has shown the wrong totals since Monday") v1 writes "Weekly chart shows incorrect totals since Monday." v2 writes "Dashboard praised; weekly chart totals are incorrect."
- v2 gets all 8 right.

**The price is a longer prompt.** v2's system prompt is 23% longer in bytes, which shows up as +19% prompt tokens and +13% cost per item. At FeedbackOS's scale this is about two cents more per 1,000 items. That's trivially worth it, but it is a number, not a feeling.

**The CI gate sees both sides.** `evalbar diff` compares the two runs case by case:

- v1 → v2: the quality gate (`--fail-on-drop 0.02`) passes, but a 10% cost limit (`--max-cost-increase 0.10`) fails at +13%. The trade-off is surfaced, not hidden.
- v2 → v1, i.e. treating a revert as a regression: the gate fails and names the cases. Sentiment drops 6 points on `mixed-01`–`mixed-03`, and praise is lost on all eight mixed cases.

## What else the eval found

Both prompts miss the same two cases. That rules out the prompt change as the cause, and each points to a real fix:

- **A FeedbackOS bug.** "ok see attached" is under 5 words, so the short-text guard skips the model. It then assigns the *first* product area in the list ("Dashboard") instead of the `Other` fallback. The guard should use the same fallback as the main path.
- **A prompt-following gap.** "Please send your SOC 2 report and a completed vendor security questionnaire" comes back as `question_support`, although the prompt's own boundary rule says administrative and procurement requests are `other`. A worked example in the prompt would probably fix it, and evalbar can show whether it does.

Two more misses are arguably label ambiguity rather than model errors:

- `feat-03`: webhooks, `API` or `Integrations`?
- `mixed-03`: a broken "Remember me" box, `bug_problem` or `usability_issue`?

Running against real responses also exposed two bugs in evalbar itself before its first release:

- **Cost lookup.** LiteLLM's `completion_cost()` cannot price Groq responses that name an OpenAI-style model. evalbar now uses the cost LiteLLM records during the call.
- **Lost scorer.** A custom scorer could silently drop out of the summary.

Neither would have shown up with mocked responses alone.

## Limits

- **Small and synthetic.** 50 cases written and labelled by the evalbar author, so one case is worth 2 points. These numbers do not measure production accuracy.
- **Labelled to the v2 contract.** "Praise kept" measures a field v1 cannot produce. It shows the contract change, not v1 failing at its own job; theme and sentiment accuracy are the fair comparison.
- **One run per prompt.** `gpt-oss-120b` at temperature 0 is not guaranteed to be deterministic, so small differences (like the latency gap) can be noise.

## Cost of the study

Two runs used 96,591 tokens in 98 model calls. The list price was $0.017, and the actual spend on Groq's free tier was $0. Each run took about 8 minutes, mostly the deliberate spacing for the rate limit.

## Next steps

1. Fix the short-text guard's product area and add a worked SOC 2 example to the prompt. Rerun to confirm both cases flip without breaking others.
2. Add the evalbar gate to FeedbackOS's CI (template: [`examples/ci/evalbar-gate.yml`](../examples/ci/evalbar-gate.yml)), so prompt changes are measured before they ship rather than after.
3. Repeat the comparison on a larger, independently labelled set of real public reviews to get numbers that say something about production.
