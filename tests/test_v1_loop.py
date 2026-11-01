"""The v1 loop: gate default and curve, retraining on logs + queue, active-learning queue, cost dashboard, and
the parts of LoRA serving that don't need the 6 GB model."""

import json
import math

import numpy as np
import pytest

from conftest import make_logs
from distill.logs.ingest import read_jsonl
from distill.monitor import active
from distill.serve.gate import Gateway
from distill.serve.student import Student, extractor_for, load_students
from distill.train.baseline import DEFAULT_TARGET, format_curve, run
from distill.train.features import HashingFeatures


@pytest.fixture
def trained(write_jsonl, tmp_path):
    logs = write_jsonl(make_logs(600), "logs.jsonl")
    report = run(str(logs), HashingFeatures(), save_dir=tmp_path / "students")
    return logs, tmp_path / "students", report["tasks"]["ticket_team"]


def labels_of(rows):
    by_state = {r["state"]: r["label"] for r in rows}
    return lambda task, state, question: by_state[state]


def test_default_gate_target_is_085_and_the_curve_marks_it(trained):
    _, _, t = trained
    assert DEFAULT_TARGET == 0.85 and t["gate"]["target_agreement"] == 0.85
    text = format_curve("ticket_team", t["gate"])
    marked = [line for line in text.splitlines() if line.startswith(" *")]
    assert len(marked) == 1 and marked[0].split()[1] == "0.85"


def test_retraining_reads_logs_and_queue_together(trained, tmp_path, write_jsonl):
    logs, students_dir, _ = trained
    students = load_students(students_dir)
    students["ticket_team"].tau = math.inf  # everything falls back, so everything is queued
    fresh = make_logs(50, start=7000, seed=9)
    queue = tmp_path / "queue.jsonl"
    gw = Gateway(students, HashingFeatures(), labels_of(fresh), queue)
    for r in fresh:
        gw.decide("ticket_team", r["state"])
    report = run([str(logs), str(queue)], HashingFeatures())
    assert report["ingest"]["records"] == 650 and report["ingest"]["rejected"] == 0


def test_gateway_counts_every_teacher_call_in_the_live_cost(trained):
    _, students_dir, _ = trained
    students = load_students(students_dir)
    s = students["ticket_team"]
    s.tau = 0.0  # the student answers everything; the only teacher calls are shadow checks
    rows = make_logs(40, start=8000, seed=10)
    gw = Gateway(students, HashingFeatures(), labels_of(rows), shadow_rate=0.5)
    for r in rows:
        gw.decide("ticket_team", r["state"])
    st = gw.status()["ticket_team"]
    assert st["fallbacks"] == 0 and st["teacher_calls"] == st["n_shadow"] > 0
    unit = s.meta["teacher_cost_usd"]
    assert st["cost_per_1k_teacher_only"] == pytest.approx(1000 * unit)
    assert st["cost_per_1k_observed"] == pytest.approx(1000 * unit * st["teacher_calls"] / 40)


def test_dashboard_shows_cost_and_alarms_and_escapes_task_names(trained):
    from fastapi.testclient import TestClient

    from distill.serve.app import create_app

    _, students_dir, _ = trained
    students = load_students(students_dir)
    s = students.pop("ticket_team")
    s.task = "<script>alert(1)</script>"
    students[s.task] = s
    rows = make_logs(5, start=9000)
    gw = Gateway(students, HashingFeatures(), labels_of(rows))
    gw.monitors[s.task].observe_shadow(False)  # a tiny window can't alarm; force one to render the badge
    gw.monitors[s.task].min_shadow, gw.monitors[s.task].floors["agreement"] = 1, 0.9
    client = TestClient(create_app(gw))
    client.post("/decide", json={"task": s.task, "state": rows[0]["state"]})
    html = client.get("/dashboard").text
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "$ / 1k, gated" in html and "⚠ agreement" in html


def test_active_learning_queue_ranks_the_pool_and_labels_it_into_the_queue(trained, tmp_path, write_jsonl):
    logs, students_dir, _ = trained
    students = load_students(students_dir)
    pool_path = tmp_path / "pool.jsonl"
    gw = Gateway(students, HashingFeatures(), labels_of([]), pool_path=pool_path)
    students["ticket_team"].tau = 0.0
    for r in make_logs(30, start=9500, seed=11) * 2:  # every request twice: the pool repeats itself
        gw.decide("ticket_team", r["state"])
    pool = [json.loads(line) for line in pool_path.read_text().splitlines()]
    picked = active.select(students, HashingFeatures(), pool, read_jsonl(logs)[0], budget=10)
    assert len(picked) == 10 and len({p["state"] for p in picked}) == 10  # repeats count once

    queue = tmp_path / "queue.jsonl"
    active.main([str(students_dir), str(pool_path), str(logs), "--budget", "5",
                 "--teacher", "conftest:always_billing", "--queue", str(queue)])
    queued, errors = read_jsonl(queue)
    assert errors == [] and len(queued) == 5 and {q.label for q in queued} == {"billing"}


def test_lora_student_adapter_name_cannot_leave_its_directory(trained):
    _, students_dir, _ = trained
    s = next(iter(load_students(students_dir).values()))
    bad = Student(**{**s.__dict__, "meta": {"features": "d1", "model_id": "x", "dtype": "bfloat16",
                                            "adapter": "../../elsewhere.pt", "lora_config": {}}})
    with pytest.raises(ValueError, match="file name next to the student"):
        extractor_for({"t": bad})  # refused before any model is loaded


def test_lora_readout_head_is_d1s_own_linear_readout():
    from distill.train.lora import readout_head

    P = np.random.default_rng(0).normal(size=(16, 3))

    class Fake:
        def readout_weights(self, question):
            return P

    X = np.random.default_rng(1).normal(size=(5, 16))
    assert np.allclose(readout_head(Fake(), {}).logits(X), X @ P)
