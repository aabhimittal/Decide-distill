import json
import math

import numpy as np
import pytest

from conftest import QUESTION, make_logs
from distill.logs.ingest import parse, read_jsonl
from distill.monitor.active import prioritize
from distill.monitor.drift import DriftMonitor, psi
from distill.serve.gate import Gateway, choose_tau, gate_metrics
from distill.serve.replay import replay
from distill.serve.student import Student, extractor_for, load_students
from distill.train.baseline import run
from distill.train.features import HashingFeatures

FRAUD_SURGE = [0.2, 0.2, 0.6]  # billing, technical, fraud


@pytest.fixture
def students(write_jsonl, tmp_path):
    """Students trained and saved by the real pipeline, plus the report it wrote."""
    report = run(str(write_jsonl(make_logs(600))), HashingFeatures(), save_dir=tmp_path / "students")
    return load_students(tmp_path / "students"), report["tasks"]["ticket_team"]


def labels_of(rows):
    by_state = {r["state"]: r["label"] for r in rows}
    return lambda task, state, question: by_state[state]


# ------------------------------------------------------------------ drift

def test_psi():
    assert psi([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0)
    a, e = np.array([0.5, 0.5]), np.array([0.9, 0.1])
    assert psi(e, a) == pytest.approx(float(np.sum((a - e) * np.log(a / e))))
    assert math.isfinite(psi([50, 50, 0], [10, 10, 80]))  # an empty reference bin doesn't blow up


def test_drift_monitor_waits_for_enough_rows_then_alarms():
    m = DriftMonitor([100, 100, 100], min_n=50, agreement_floor=0.8, min_shadow=10)
    for _ in range(49):
        m.observe_prediction(2)
    assert m.status()["psi"] > 0.2 and not m.status()["psi_alarm"]
    m.observe_prediction(2)
    assert m.status()["psi_alarm"]
    for agrees in [True] * 5 + [False] * 5:
        m.observe_shadow(agrees)
    assert m.status()["agreement"] == 0.5 and m.status()["agreement_alarm"]


# ------------------------------------------------------------------ gate

def test_choose_tau_maximizes_coverage_at_target():
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
    assert choose_tau(conf, correct, 0.8) == 0.7   # top 5: 4/5 = 0.8
    assert choose_tau(conf, correct, 1.0) == 0.9   # top 3 only
    assert choose_tau(conf, ~correct, 0.9) == math.inf


def test_gate_metrics_counts_fallbacks_as_teacher_answers():
    m = gate_metrics(np.array([0.9, 0.4]), np.array([0, 0]), np.array([1, 1]), 0.5,
                     human=np.array([1, -1]), teacher_cost=0.002)
    assert m["fallback_rate"] == 0.5 and m["accepted_agreement"] == 0.0 and m["system_agreement"] == 0.5
    assert m["system_accuracy_vs_human"] == 0.0  # only row 0 has a human label, and the student got it wrong
    assert m["cost_per_1k_teacher_only"] == pytest.approx(2.0)
    assert m["cost_per_1k_gated"] == pytest.approx(1.0)


def test_student_round_trips_and_task_names_never_become_paths(tmp_path, students):
    students, _ = students
    s = next(iter(students.values()))
    s.task = "../../etc/passwd"
    path = s.save(tmp_path / "out")
    assert path.parent == tmp_path / "out"
    back = Student.load(path)
    X = HashingFeatures()(["refund charged twice", "app crash"])
    assert back.task == s.task and np.allclose(back.probs(X), s.probs(X)) and back.tau == s.tau


def test_extractor_for_refuses_mixed_features(students):
    students, _ = students
    s = next(iter(students.values()))
    assert isinstance(extractor_for(students), HashingFeatures)
    other = Student(**{**s.__dict__, "task": "x", "meta": {**s.meta, "dim": 8}})
    with pytest.raises(ValueError, match="different features"):
        extractor_for({"a": s, "x": other})


def test_fallbacks_and_shadow_labels_land_in_the_queue_as_ingestible_logs(tmp_path, students):
    students, _ = students
    s = students["ticket_team"]
    s.tau = 0.6
    rows = make_logs(100, start=1000, seed=3)
    queue, pool = tmp_path / "queue.jsonl", tmp_path / "pool.jsonl"
    gw = Gateway(students, HashingFeatures(), labels_of(rows), queue, pool, shadow_rate=0.5)
    out = [gw.decide("ticket_team", r["state"]) for r in rows]
    n_fallback = sum(o["source"] == "teacher" for o in out)
    assert 0 < n_fallback < len(rows)
    queued, errors = read_jsonl(queue)
    assert errors == [] and len(queued) == n_fallback + gw.status()["ticket_team"]["n_shadow"]
    assert all(q.question == QUESTION for q in queued)
    assert len(pool.read_text().splitlines()) == len(rows) - n_fallback
    assert gw.status()["ticket_team"]["fallbacks"] == n_fallback


def test_teacher_answer_outside_the_options_is_rejected_and_not_queued(tmp_path, students):
    students, _ = students
    students["ticket_team"].tau = math.inf  # always fall back
    gw = Gateway(students, HashingFeatures(), lambda *a: "sales", tmp_path / "q.jsonl")
    with pytest.raises(ValueError, match="sales"):
        gw.decide("ticket_team", "hello")
    assert not (tmp_path / "q.jsonl").exists()


def test_replay_through_the_served_gate_matches_the_offline_estimate(students):
    """M3 acceptance in miniature: the test slice replayed one request at a time through students loaded from
    disk gives exactly the fallback rate and agreement the training report predicted for it."""
    students, report = students
    test_rows = [parse(r) for r in make_logs(600)][510:]  # the newest 15%: the report's test slice
    got = replay(students, HashingFeatures(), test_rows)["ticket_team"]
    for key in ("fallback_rate", "accepted_agreement", "system_agreement", "system_accuracy_vs_human", "tau"):
        assert got[key] == pytest.approx(report["gate"]["test"][key]), key


def test_http_app(students):
    from fastapi.testclient import TestClient

    from distill.serve.app import create_app

    students, _ = students
    rows = make_logs(5, start=2000)
    client = TestClient(create_app(Gateway(students, HashingFeatures(), labels_of(rows))))
    r = client.post("/decide", json={"task": "ticket_team", "state": rows[0]["state"]})
    assert r.status_code == 200 and r.json()["answer"] in QUESTION["criteria"]
    assert client.post("/decide", json={"task": "nope", "state": "x"}).status_code == 404
    assert client.get("/monitor").json()["ticket_team"]["requests"] == 1


# ------------------------------------------------------------------ M4: simulated drift

def test_drift_scenario_psi_and_agreement_alarms(students):
    """Stable traffic stays quiet; a fraud surge trips PSI; a teacher policy change trips shadow agreement."""
    students, _ = students
    s = students["ticket_team"]

    def stream(rows, teacher):
        gw = Gateway(students, HashingFeatures(), teacher, shadow_rate=1.0)
        for r in rows:
            gw.decide("ticket_team", r["state"])
        return gw.status()["ticket_team"]

    stable = make_logs(300, start=3000, seed=4)
    surge = make_logs(300, start=4000, seed=5, mix=FRAUD_SURGE)
    assert not stream(stable, labels_of(stable))["psi_alarm"]
    assert stream(surge, labels_of(surge))["psi_alarm"]

    # Concept drift: the teacher now routes all billing to fraud. Inputs look the same, so PSI stays quiet.
    relabel = {r["state"]: ("fraud" if r["label"] == "billing" else r["label"]) for r in stable}
    shifted = stream(stable, lambda task, state, q: relabel[state])
    assert not shifted["psi_alarm"] and shifted["agreement_alarm"]
    assert s.expected_agreement is not None
    # Clean traffic must not trip either agreement alarm.
    clean = stream(stable, labels_of(stable))
    assert not clean["agreement_alarm"] and not clean["fallback_agreement_alarm"]


def test_concept_drift_is_caught_on_fallbacks_when_the_gate_rejects_most_traffic(students):
    """The d1 run's blind spot: with ~96% fallback, shadow checks of accepted rows are too few. Fallbacks are
    teacher-labeled anyway; the student's guess vs the teacher there is the signal, against its own baseline."""
    students, _ = students
    s = students["ticket_team"]
    s.tau = math.inf  # reject everything: no shadow samples at all
    stable = make_logs(300, start=3000, seed=4)
    s.expected_fallback_agreement = float(np.mean(
        [s.classes[int(p.argmax())] == r["label"] for p, r in zip(s.probs(HashingFeatures()([r["state"] for r in stable])), stable)]))
    relabel = {r["state"]: ("fraud" if r["label"] == "billing" else r["label"]) for r in stable}
    for teacher, alarm in [(labels_of(stable), False), (lambda task, state, q: relabel[state], True)]:
        gw = Gateway(students, HashingFeatures(), teacher, shadow_rate=1.0)
        for r in stable:
            gw.decide("ticket_team", r["state"])
        st = gw.status()["ticket_team"]
        assert st["n_shadow"] == 0 and st["n_fallback"] == 300 and st["fallback_agreement_alarm"] is alarm


def test_active_learning_prefers_regions_training_never_saw():
    """Three well-separated regions; training has only two. With no entropy signal, the cluster term alone must
    put the unseen region first. (Whether real features cluster by topic is measured in simulate_drift.py.)"""
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 10, (3, 16))
    blob = lambda c, n: centres[c] + rng.normal(0, 1, (n, 16))
    Xtr = np.vstack([blob(0, 150), blob(1, 150)])
    Xp = np.vstack([blob(0, 100), blob(1, 100), blob(2, 40)])
    new = np.r_[np.zeros(200, bool), np.ones(40, bool)]
    picked = prioritize(np.full((len(Xp), 3), 1 / 3), Xp, Xtr, budget=40, k=3)
    assert new[picked].mean() > 0.9


def test_active_learning_breaks_ties_by_entropy():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (50, 4))
    P = np.tile([0.9, 0.05, 0.05], (50, 1))
    P[7] = [0.34, 0.33, 0.33]  # the one row the student is unsure about
    assert prioritize(P, X, X, budget=1, k=1)[0] == 7
