from evalbar.dataset import Case
from evalbar.scorers import Score, contains, exact, json_schema, regex


def case(expected=None):
    return Case(id="c1", input="x", expected=expected or {}, metadata={})


def test_exact_compares_a_field_case_insensitively_by_default():
    scorer = exact("theme")

    hit = scorer({"theme": "Praise"}, case({"theme": "praise"}))
    miss = scorer({"theme": "bug"}, case({"theme": "praise"}))

    assert hit == Score(name="exact:theme", value=1.0, passed=True, reason="")
    assert miss.passed is False and miss.value == 0.0
    assert "expected 'praise', got 'bug'" in miss.reason


def test_exact_without_field_compares_whole_output_to_expected_output():
    scorer = exact()

    assert scorer("42", case({"output": "42"})).passed
    assert not scorer("41", case({"output": "42"})).passed


def test_scorer_skips_case_without_that_expectation():
    assert exact("theme")({"theme": "x"}, case({"sentiment": "neutral"})) is None


def test_exact_missing_output_field_fails_with_reason():
    score = exact("theme")({"other": 1}, case({"theme": "praise"}))

    assert not score.passed
    assert "missing" in score.reason


def test_contains_checks_every_expected_item_is_in_output_list():
    scorer = contains("secondary_themes")

    assert scorer({"secondary_themes": ["praise", "bug"]}, case({"secondary_themes": ["praise"]})).passed
    miss = scorer({"secondary_themes": []}, case({"secondary_themes": ["praise"]}))
    assert not miss.passed and "praise" in miss.reason


def test_contains_works_on_substrings():
    scorer = contains("answer")

    assert scorer({"answer": "The capital is Paris."}, case({"answer": "paris"})).passed


def test_regex_matches_output_field_without_needing_expectation():
    words_15 = regex("summary", r"^\S+(\s+\S+){0,14}$", name="summary<=15w")

    short = words_15({"summary": "CSV export drops rows after 1,000."}, case())
    long = words_15({"summary": " ".join(["word"] * 16)}, case())

    assert short.name == "summary<=15w" and short.passed
    assert not long.passed


def test_regex_can_require_no_match():
    scorer = regex("summary", r"^The user", must_match=False)

    assert scorer({"summary": "Export is slow."}, case()).passed
    assert not scorer({"summary": "The user hates export."}, case()).passed


def test_json_schema_validates_output():
    schema = {
        "type": "object",
        "properties": {"theme": {"enum": ["bug", "praise"]}},
        "required": ["theme"],
    }
    scorer = json_schema(schema)

    assert scorer({"theme": "bug"}, case()).passed
    bad = scorer({"theme": "nope"}, case())
    assert not bad.passed and "nope" in bad.reason


def test_json_schema_parses_string_output():
    scorer = json_schema({"type": "object", "required": ["a"]})

    assert scorer('{"a": 1}', case()).passed
    assert "not valid JSON" in scorer("{oops", case()).reason
