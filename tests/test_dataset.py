import random
from dataclasses import replace
from datetime import timedelta

import pytest

from conftest import make_logs
from distill.dataset.build import by_task, dedupe, time_split
from distill.logs.ingest import parse


def records(n=100):
    return [parse(r) for r in make_logs(n)]


def test_dedupe_normalizes_case_whitespace_and_key_order():
    a = parse({**make_logs(1)[0], "state": "Charged  TWICE\n please"})
    b = replace(a, id="b", ts=a.ts + timedelta(hours=1), state="charged twice please")
    c = replace(a, id="c", ts=a.ts + timedelta(hours=2), state={"x": 1, "y": 2})
    d = replace(a, id="d", ts=a.ts + timedelta(hours=3), state={"y": 2, "x": 1})
    other_task = replace(a, id="e", task="other")
    assert [r.id for r in dedupe([d, c, b, a, other_task])] == ["e", "r0", "c"]


def test_dedupe_keeps_earliest_so_repeats_cannot_leak_forward():
    a = parse(make_logs(1)[0])
    late_twin = replace(a, id="late", ts=a.ts + timedelta(days=30))
    assert [r.id for r in dedupe([late_twin, a])] == ["r0"]


def test_split_is_deterministic_and_independent_of_input_order():
    rows = records()
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    s1, s2 = time_split(rows), time_split(shuffled)
    assert [[d.id for d in p] for p in (s1.train, s1.calib, s1.test)] == \
           [[d.id for d in p] for p in (s2.train, s2.calib, s2.test)]
    assert (len(s1.train), len(s1.calib), len(s1.test)) == (70, 15, 15)


def test_split_is_by_time():
    s = time_split(records())
    assert max(d.ts for d in s.train) <= min(d.ts for d in s.calib)
    assert max(d.ts for d in s.calib) <= min(d.ts for d in s.test)


def test_split_ties_break_on_id():
    a = parse(make_logs(1)[0])
    same_time = [replace(a, id=i) for i in "dcba"]
    assert [d.id for d in time_split(same_time, (0.5, 0.0, 0.5)).train] == ["a", "b"]


def test_bad_fractions_raise():
    with pytest.raises(ValueError):
        time_split(records(), (0.5, 0.5, 0.5))


def test_by_task_rejects_mixed_option_lists():
    rows = records(3)
    bad = replace(rows[2], question={"type": "noul", "instructions": "?"}, label="yes")
    assert list(by_task(rows)) == ["ticket_team"]
    with pytest.raises(ValueError, match="options"):
        by_task(rows[:2] + [bad])
