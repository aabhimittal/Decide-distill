"""M0: dedupe and time-based split.

Split by time, not randomly: production repeats itself (the same ticket re-sent, a retried request), and a
random split puts one copy in train and its twin in test, so offline accuracy overstates what the model does
on next month's traffic.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from distill.logs.ingest import Decision, options


def _canonical_state(state) -> str:
    if isinstance(state, str):
        return " ".join(state.lower().split())
    return json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_key(d: Decision) -> tuple[str, str, str]:
    return d.task, _canonical_state(d.state), json.dumps(d.question, sort_keys=True)


def chronological(records: list[Decision]) -> list[Decision]:
    """Sorted by time, ties broken by id, so the order does not depend on the input order."""
    return sorted(records, key=lambda d: (d.ts, d.id))


def dedupe(records: list[Decision]) -> list[Decision]:
    """Keep the earliest record per (task, normalized state, question).

    Keeping the earliest means a repeat can never land in a later split than its original.
    shortcut: exact match after case/whitespace/key-order normalization only; add MinHash near-dup
    detection if logs carry templated inputs that differ by an id or a timestamp.
    """
    seen: set = set()
    out = []
    for d in chronological(records):
        key = content_key(d)
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


@dataclass(frozen=True)
class Split:
    train: list[Decision]
    calib: list[Decision]  # held-out slice for temperature scaling
    test: list[Decision]


def time_split(records: list[Decision], fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Split:
    """Oldest `fractions[0]` to train, the next slice to calibration, the newest to test."""
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-9:
        raise ValueError(f"fractions must be three non-negative numbers summing to 1, got {fractions}")
    ordered = chronological(records)
    a = round(len(ordered) * fractions[0])
    b = round(len(ordered) * (fractions[0] + fractions[1]))
    return Split(ordered[:a], ordered[a:b], ordered[b:])


def by_task(records: list[Decision]) -> dict[str, list[Decision]]:
    """Records grouped by task family; every record of a task must share one option list."""
    groups: dict[str, list[Decision]] = defaultdict(list)
    for d in records:
        groups[d.task].append(d)
    for task, rows in groups.items():
        first = options(rows[0].question)
        bad = next((d for d in rows if options(d.question) != first), None)
        if bad is not None:
            raise ValueError(f"task {task!r}: record {bad.id!r} has options {options(bad.question)}, "
                             f"expected {first}")
    return dict(groups)
