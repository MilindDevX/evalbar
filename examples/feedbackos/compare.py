"""Before/after table for two FeedbackOS run files (a stand-in until `evalbar diff` lands in M2).

    python examples/feedbackos/compare.py runs/feedbackos-v1.json runs/feedbackos-v2.json
"""

import sys
from collections import defaultdict

from evalbar import load_run, summarize
from evalbar.report import format_cost

HEADLINE = ["exact:theme", "exact:sentiment", "theme+sentiment", "exact:product_area",
            "contains:secondary_themes", "valid_output", "summary<=15 words"]


def group_accuracy(run, scorer):
    groups = defaultdict(lambda: [0, 0])
    for case in run.cases:
        group = case.case_id.rsplit("-", 1)[0]  # ids are <group>-<n>
        for score in case.scores:
            if score.name == scorer:
                groups[group][0] += score.passed
                groups[group][1] += 1
    return groups


def main(before_path, after_path):
    before, after = load_run(before_path), load_run(after_path)
    sb, sa = summarize(before), summarize(after)

    print(f"| metric | {before.name} | {after.name} | change |")
    print("|---|---|---|---|")
    for name in HEADLINE:
        b, a = sb.scorers.get(name), sa.scorers.get(name)
        if not b or not a:
            continue
        delta = (a.pass_rate - b.pass_rate) * 100
        print(f"| {name} (n={a.n}) | {b.passed}/{b.n} ({b.pass_rate:.0%}) | {a.passed}/{a.n} ({a.pass_rate:.0%}) "
              f"| {delta:+.0f} pts |")
    print(f"| errors | {sb.n_errors} | {sa.n_errors} | |")
    print(f"| prompt tokens / case | {sb.prompt_tokens / sb.n_cases:,.0f} | {sa.prompt_tokens / sa.n_cases:,.0f} | "
          f"{(sa.prompt_tokens / sb.prompt_tokens - 1):+.0%} |")
    print(f"| completion tokens / case | {sb.completion_tokens / sb.n_cases:,.0f} | "
          f"{sa.completion_tokens / sa.n_cases:,.0f} | {(sa.completion_tokens / sb.completion_tokens - 1):+.0%} |")
    print(f"| cost / 1,000 items (list price) | {format_cost(sb.cost_usd / sb.n_cases * 1000)} | "
          f"{format_cost(sa.cost_usd / sa.n_cases * 1000)} | {(sa.cost_usd / sb.cost_usd - 1):+.0%} |")
    print(f"| LLM latency p50 / p95 | {sb.llm_latency_p50_s:.2f}s / {sb.llm_latency_p95_s:.2f}s | "
          f"{sa.llm_latency_p50_s:.2f}s / {sa.llm_latency_p95_s:.2f}s | |")

    print("\nTheme accuracy by group:")
    gb, ga = group_accuracy(before, "exact:theme"), group_accuracy(after, "exact:theme")
    for group in ga:
        print(f"  {group:10s} {gb[group][0]}/{gb[group][1]} -> {ga[group][0]}/{ga[group][1]}")

    print("\nCases whose theme+sentiment result changed:")
    before_cases = {case.case_id: case for case in before.cases}
    for case in after.cases:
        old = before_cases.get(case.case_id)
        if old is None:
            continue
        was = next((s.passed for s in old.scores if s.name == "theme+sentiment"), None)
        now = next((s.passed for s in case.scores if s.name == "theme+sentiment"), None)
        if was != now:
            got = lambda c: f"{(c.output or {}).get('theme')}/{(c.output or {}).get('sentiment')}"  # noqa: E731
            print(f"  {case.case_id}: {'pass' if was else 'fail'} -> {'pass' if now else 'fail'} "
                  f"({got(old)} -> {got(case)}; expected {case.expected['theme']}/{case.expected['sentiment']})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
