import pytest

from evalbar.dataset import Case, DatasetError, load_dataset


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_loads_jsonl_cases_in_order(tmp_path):
    path = write(
        tmp_path,
        "cases.jsonl",
        '{"id": "a", "input": {"text": "hi"}, "expected": {"label": "x"}}\n'
        "\n"
        '{"id": "b", "input": "plain", "metadata": {"split": "dev"}}\n',
    )

    cases = load_dataset(path)

    assert cases == [
        Case(id="a", input={"text": "hi"}, expected={"label": "x"}, metadata={}),
        Case(id="b", input="plain", expected={}, metadata={"split": "dev"}),
    ]


def test_loads_yaml_list_or_cases_key(tmp_path):
    as_list = write(tmp_path, "a.yaml", "- id: one\n  input: q1\n  expected: {answer: '1'}\n")
    as_key = write(tmp_path, "b.yml", "cases:\n  - id: two\n    input: q2\n")

    assert load_dataset(as_list)[0].expected == {"answer": "1"}
    assert load_dataset(as_key)[0].id == "two"


def test_invalid_json_reports_line_number(tmp_path):
    path = write(tmp_path, "bad.jsonl", '{"id": "a", "input": 1}\n{"id": "b",\n')

    with pytest.raises(DatasetError, match="line 2"):
        load_dataset(path)


def test_duplicate_and_missing_ids_are_rejected(tmp_path):
    dup = write(tmp_path, "dup.jsonl", '{"id": "a", "input": 1}\n{"id": "a", "input": 2}\n')
    missing = write(tmp_path, "missing.jsonl", '{"input": 1}\n')

    with pytest.raises(DatasetError, match="duplicate id 'a'"):
        load_dataset(dup)
    with pytest.raises(DatasetError, match="missing 'id'"):
        load_dataset(missing)


def test_case_requires_input(tmp_path):
    path = write(tmp_path, "noinput.jsonl", '{"id": "a"}\n')

    with pytest.raises(DatasetError, match="missing 'input'"):
        load_dataset(path)


def test_unknown_extension_is_rejected(tmp_path):
    path = write(tmp_path, "cases.csv", "id,input\n")

    with pytest.raises(DatasetError, match="unsupported"):
        load_dataset(path)
