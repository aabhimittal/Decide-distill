"""M0: the decision-log schema and JSONL ingestion.

One line per teacher decision:

    {"id": "t-1", "ts": "2026-03-01T12:00:00Z", "task": "ticket_team",
     "state": "I was charged twice", "question": {"type": "choice", "instructions": "...",
     "criteria": {"billing": "...", "technical": "..."}},
     "label": "billing", "teacher_confidence": 0.93, "human_label": "billing", "cost_usd": 0.0004}

`question` follows the d1 / Decision Index schema (`type` noul | choice | score), so the logged question can
be replayed against d1 verbatim. `label` is the teacher's answer as an option name: "yes"/"no" for noul,
a criteria key for choice, the level index as a string ("0".."9") for score.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Decision:
    id: str
    ts: datetime
    task: str
    state: Any
    question: dict
    label: str
    teacher_confidence: float | None = None
    human_label: str | None = None
    cost_usd: float | None = None


def options(question: dict) -> list[str]:
    """The option names a question's answer is one of, in d1's readout order."""
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "noul":
        return ["yes", "no"]
    if kind == "choice" and isinstance(criteria, dict) and criteria:
        return list(criteria)
    if kind == "score" and isinstance(criteria, list) and 2 <= len(criteria) <= 10:
        return [str(i) for i in range(len(criteria))]
    raise ValueError(f"bad question: type must be noul|choice|score with valid criteria, got {question!r}")


def parse(row: dict) -> Decision:
    """One validated record; raises ValueError naming the problem."""
    for key in ("id", "ts", "task", "state", "question", "label"):
        if key not in row:
            raise ValueError(f"missing field {key!r}")
    ts = datetime.fromisoformat(row["ts"])
    if ts.tzinfo is None:  # mixing naive and aware times would make the time split wrong or crash
        raise ValueError(f"ts {row['ts']!r} has no timezone")
    if not isinstance(row["question"], dict):
        raise ValueError("question must be an object")
    opts = options(row["question"])
    label, human = str(row["label"]), row.get("human_label")
    if label not in opts:
        raise ValueError(f"label {label!r} not in options {opts}")
    if human is not None and str(human) not in opts:
        raise ValueError(f"human_label {human!r} not in options {opts}")
    conf = row.get("teacher_confidence")
    if conf is not None and not 0.0 <= float(conf) <= 1.0:
        raise ValueError(f"teacher_confidence {conf!r} outside [0, 1]")
    return Decision(
        id=str(row["id"]), ts=ts, task=str(row["task"]), state=row["state"], question=row["question"],
        label=label, teacher_confidence=None if conf is None else float(conf),
        human_label=None if human is None else str(human),
        cost_usd=None if row.get("cost_usd") is None else float(row["cost_usd"]),
    )


def read_jsonl(path: str | Path, strict: bool = False) -> tuple[list[Decision], list[str]]:
    """Records and `"line N: reason"` errors. Bad lines are skipped, or raise when `strict`."""
    records: list[Decision] = []
    errors: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                d = parse(json.loads(line))
                if d.id in seen:
                    raise ValueError(f"duplicate id {d.id!r}")
            except (ValueError, TypeError) as e:  # JSONDecodeError is a ValueError
                if strict:
                    raise ValueError(f"line {n}: {e}") from e
                errors.append(f"line {n}: {e}")
                continue
            seen.add(d.id)
            records.append(d)
    return records, errors
