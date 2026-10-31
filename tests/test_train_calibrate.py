import numpy as np

from conftest import make_logs
from distill.calibrate.temperature import ece, fit_temperature, softmax
from distill.train.baseline import run
from distill.train.features import HashingFeatures
from distill.train.linear_head import LinearHead


def test_fit_temperature_recovers_true_temperature():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, (20000, 4))
    p = softmax(logits / 2.0)  # the truth is twice as uncertain as the logits claim
    y = np.array([rng.choice(4, p=row) for row in p])
    assert abs(fit_temperature(logits, y) - 2.0) < 0.1
    assert ece(softmax(logits / fit_temperature(logits, y)), y) < ece(softmax(logits), y)


def test_ece_is_zero_when_confidence_matches_accuracy():
    probs = np.tile([0.75, 0.25], (4, 1))
    assert ece(probs, np.array([0, 0, 0, 1])) == 0.0
    assert abs(ece(probs, np.array([1, 1, 1, 1])) - 0.75) < 1e-12


def test_linear_head_learns_separable_classes():
    rng = np.random.default_rng(0)
    y = rng.integers(3, size=300)
    X = rng.normal(size=(300, 20)) + 3 * np.eye(20)[y]
    assert (LinearHead().fit(X, y, 3).logits(X).argmax(1) == y).mean() > 0.95


def test_hashing_features_are_deterministic():
    f = HashingFeatures(256)
    assert np.array_equal(f(["a b c", {"k": 1}]), f(["a b c", {"k": 1}]))


def test_baseline_reports_accuracy_and_ece(write_jsonl):
    rows = make_logs(600, teacher_noise=0.1)
    rows += [{**rows[0], "id": "dupe"}, "garbage"]
    report = run(str(write_jsonl(rows)), HashingFeatures())
    assert report["ingest"] == {"records": 601, "rejected": 1, "after_dedupe": 600, "errors": report["ingest"]["errors"]}
    t = report["tasks"]["ticket_team"]
    assert t["n"] == {"train": 420, "calib": 90, "test": 90}
    assert t["accuracy_vs_teacher"] > 0.6  # chance is 1/3
    assert t["accuracy_vs_human"] > 0.6 and t["n_human_labels"] == 90
    assert 0.8 < t["teacher_accuracy_vs_human"] <= 1.0
    assert 0 <= t["ece_after"] <= 1 and 0 <= t["ece_before"] <= 1 and t["temperature"] > 0


def test_baseline_skips_task_without_enough_data(write_jsonl):
    rows = [{**r, "label": "billing"} for r in make_logs(20)]
    assert "skipped" in run(str(write_jsonl(rows)), HashingFeatures())["tasks"]["ticket_team"]
