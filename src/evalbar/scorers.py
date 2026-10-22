"""Scorers turn (output, case) into a Score.

A scorer returns None when a case carries no expectation for it, so each scorer is
averaged only over the cases it applies to.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match

from .dataset import Case

_MISSING = object()


@dataclass(frozen=True)
class Score:
    name: str
    value: float
    passed: bool
    reason: str = ""


ScoreFn = Callable[[Any, Case], "Score | None"]


class Scorer:
    """A named scoring function. `expects` names the case.expected key it needs, if any."""

    def __init__(self, name: str, fn: ScoreFn, expects: str | None = None):
        self.name = name
        self.expects = expects
        self._fn = fn

    def applies(self, case: Case) -> bool:
        return self.expects is None or self.expects in case.expected

    def __call__(self, output: Any, case: Case) -> Score | None:
        if not self.applies(case):
            return None
        return self._fn(output, case)


def as_scorer(scorer: Scorer | ScoreFn) -> Scorer:
    if isinstance(scorer, Scorer):
        return scorer
    return Scorer(getattr(scorer, "__name__", "custom"), scorer)


def exact(field: str | None = None, *, expected_key: str | None = None, case_sensitive: bool = False,
          name: str | None = None) -> Scorer:
    """Output (or output[field]) equals case.expected[expected_key or field or 'output']."""
    key = expected_key or field or "output"
    scorer_name = name or (f"exact:{field}" if field else "exact")

    def score(output: Any, case: Case) -> Score:
        actual = _pick(output, field)
        if actual is _MISSING:
            return _fail(scorer_name, f"field '{field}' missing in output")
        wanted = case.expected[key]
        if _normalise(actual, case_sensitive) == _normalise(wanted, case_sensitive):
            return _pass(scorer_name)
        return _fail(scorer_name, f"expected {wanted!r}, got {actual!r}")

    return Scorer(scorer_name, score, expects=key)


def contains(field: str | None = None, *, expected_key: str | None = None, name: str | None = None) -> Scorer:
    """Every expected item (or the expected substring) appears in the output value."""
    key = expected_key or field or "output"
    scorer_name = name or (f"contains:{field}" if field else "contains")

    def score(output: Any, case: Case) -> Score:
        actual = _pick(output, field)
        if actual is _MISSING:
            return _fail(scorer_name, f"field '{field}' missing in output")
        wanted = case.expected[key]
        items = wanted if isinstance(wanted, list) else [wanted]
        absent = [item for item in items if not _contains(actual, item)]
        if absent:
            return _fail(scorer_name, f"missing {absent!r} in {actual!r}")
        return _pass(scorer_name)

    return Scorer(scorer_name, score, expects=key)


def regex(field: str | None, pattern: str, *, must_match: bool = True, flags: int = 0,
          name: str | None = None) -> Scorer:
    """Output (or output[field]) matches `pattern` (re.search), or must not when must_match=False."""
    compiled = re.compile(pattern, flags)
    scorer_name = name or (f"regex:{field}" if field else "regex")

    def score(output: Any, case: Case) -> Score:
        actual = _pick(output, field)
        if actual is _MISSING:
            return _fail(scorer_name, f"field '{field}' missing in output")
        matched = compiled.search(str(actual)) is not None
        if matched == must_match:
            return _pass(scorer_name)
        verb = "does not match" if must_match else "matches"
        return _fail(scorer_name, f"{actual!r} {verb} /{pattern}/")

    return Scorer(scorer_name, score)


def json_schema(schema: dict[str, Any], *, field: str | None = None, name: str = "json_schema") -> Scorer:
    """Output (parsed from a string if needed) validates against a JSON Schema."""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)

    def score(output: Any, case: Case) -> Score:
        actual = _pick(output, field)
        if actual is _MISSING:
            return _fail(name, f"field '{field}' missing in output")
        if isinstance(actual, str):
            try:
                actual = json.loads(actual)
            except json.JSONDecodeError as exc:
                return _fail(name, f"not valid JSON ({exc.msg})")
        error = best_match(validator.iter_errors(actual))
        if error is None:
            return _pass(name)
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        return _fail(name, f"{location}: {error.message}")

    return Scorer(name, score)


def _pick(output: Any, field: str | None) -> Any:
    if field is None:
        return output
    if isinstance(output, dict) and field in output:
        return output[field]
    return _MISSING


def _normalise(value: Any, case_sensitive: bool) -> Any:
    if isinstance(value, str) and not case_sensitive:
        return value.strip().casefold()
    return value


def _contains(actual: Any, item: Any) -> bool:
    if isinstance(actual, str) and isinstance(item, str):
        return item.casefold() in actual.casefold()
    if isinstance(actual, (list, tuple, set)):
        return any(_normalise(element, False) == _normalise(item, False) for element in actual)
    return False


def _pass(name: str) -> Score:
    return Score(name=name, value=1.0, passed=True)


def _fail(name: str, reason: str) -> Score:
    return Score(name=name, value=0.0, passed=False, reason=reason)
