import pytest

from conftest import QUESTION, make_logs
from distill.logs.ingest import options, read_jsonl


def test_reads_valid_rows(write_jsonl):
    records, errors = read_jsonl(write_jsonl(make_logs(5)))
    assert len(records) == 5 and errors == []
    assert records[0].ts.tzinfo is not None and records[0].question == QUESTION


def test_bad_lines_are_reported_with_line_numbers(write_jsonl):
    good = make_logs(3)
    rows = [
        good[0],
        "{not json",
        {**good[1], "label": "sales"},                 # not an option
        {**good[1], "id": "x", "ts": "2026-01-01T00:00:00"},  # naive timestamp
        {k: v for k, v in good[2].items() if k != "task"},
        good[0],                                       # duplicate id
        {**good[2], "teacher_confidence": 1.5},
    ]
    records, errors = read_jsonl(write_jsonl(rows))
    assert [r.id for r in records] == ["r0"]
    assert [e.split(":")[0] for e in errors] == ["line 2", "line 3", "line 4", "line 5", "line 6", "line 7"]
    assert "not in options" in errors[1] and "timezone" in errors[2] and "duplicate id" in errors[4]


def test_strict_raises(write_jsonl):
    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(write_jsonl(["{"]), strict=True)


def test_options_follow_d1_schema():
    assert options({"type": "noul", "instructions": "?"}) == ["yes", "no"]
    assert options({"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}) == ["0", "1", "2"]
    assert options(QUESTION) == ["billing", "technical", "fraud"]
    with pytest.raises(ValueError):
        options({"type": "score", "instructions": "?", "criteria": ["only one"]})
