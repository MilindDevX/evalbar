"""Load test cases from JSONL or YAML files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class DatasetError(ValueError):
    """A dataset file is malformed. The message names the file and, for JSONL, the line."""


@dataclass(frozen=True)
class Case:
    id: str
    input: Any
    expected: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_dataset(path: str | Path) -> list[Case]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records = _read_jsonl(path)
    elif suffix in (".yaml", ".yml"):
        records = _read_yaml(path)
    else:
        raise DatasetError(f"{path}: unsupported dataset format '{suffix}' (use .jsonl, .yaml or .yml)")

    cases: list[Case] = []
    seen: set[str] = set()
    for where, record in records:
        case = _to_case(path, where, record)
        if case.id in seen:
            raise DatasetError(f"{path} {where}: duplicate id '{case.id}'")
        seen.add(case.id)
        cases.append(case)
    return cases


def _read_jsonl(path: Path) -> list[tuple[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append((f"line {number}", json.loads(line)))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path} line {number}: invalid JSON ({exc.msg})") from exc
    return records


def _read_yaml(path: Path) -> list[tuple[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("cases")
    if not isinstance(data, list):
        raise DatasetError(f"{path}: expected a list of cases or a mapping with a 'cases' list")
    return [(f"case {index}", record) for index, record in enumerate(data, start=1)]


def _to_case(path: Path, where: str, record: Any) -> Case:
    if not isinstance(record, dict):
        raise DatasetError(f"{path} {where}: each case must be an object")
    if "id" not in record:
        raise DatasetError(f"{path} {where}: missing 'id'")
    if "input" not in record:
        raise DatasetError(f"{path} {where}: missing 'input'")
    expected = record.get("expected") or {}
    metadata = record.get("metadata") or {}
    if not isinstance(expected, dict) or not isinstance(metadata, dict):
        raise DatasetError(f"{path} {where}: 'expected' and 'metadata' must be objects")
    return Case(id=str(record["id"]), input=record["input"], expected=expected, metadata=metadata)
