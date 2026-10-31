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


def test_linear_head_prior_is_start_point_and_shrinkage_centre():
    rng = np.random.default_rng(0)
    X = rng.normal(2, 3, size=(300, 20))
    prior = rng.normal(size=(20, 3))
    y = rng.integers(3, size=300)  # labels unrelated to the prior
    assert np.allclose(LinearHead(steps=0).fit(X, y, 3, prior).logits(X), X @ prior)
    held = LinearHead(l2=1e3).fit(X, y, 3, prior).logits(X)
    assert (held.argmax(1) == (X @ prior).argmax(1)).mean() > 0.95  # strong shrinkage: stays at the prior
    Xs = rng.normal(size=(300, 20)) + 3 * np.eye(20)[y]  # separable labels the prior knows nothing about
    assert (LinearHead().fit(Xs, y, 3, prior).logits(Xs).argmax(1) == y).mean() > 0.95  # evidence wins


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


def test_baseline_adds_zero_shot_row_when_extractor_has_one(write_jsonl):
    class WithZeroShot(HashingFeatures):
        def zero_shot(self, X, question):  # a model that always says "billing", confidently
            return np.log(np.tile([0.97, 0.02, 0.01], (len(X), 1)))

    path = str(write_jsonl(make_logs(600)))
    assert "zero_shot" not in run(path, HashingFeatures())["tasks"]["ticket_team"]
    t = run(path, WithZeroShot())["tasks"]["ticket_team"]
    zs = t["zero_shot"]
    assert set(zs) == {"accuracy_vs_teacher", "accuracy_vs_human", "temperature", "ece_before", "ece_after"}
    assert zs["accuracy_vs_teacher"] < 0.5 < t["accuracy_vs_teacher"]  # constant guess ≈ 1/3; the head learns
    assert zs["ece_after"] < zs["ece_before"]  # scaling tames the overconfident constant


def test_baseline_uses_readout_prior_only_when_asked(write_jsonl):
    class WithPrior(HashingFeatures):
        calls = 0

        def readout_weights(self, question):
            WithPrior.calls += 1
            return np.zeros((self.dim, 3))

    path = str(write_jsonl(make_logs(600)))
    run(path, WithPrior())
    assert WithPrior.calls == 0
    assert run(path, WithPrior(), use_prior=True)["tasks"]["ticket_team"]["accuracy_vs_teacher"] > 0.6
    assert WithPrior.calls == 1


def test_l2_is_chosen_on_calib_and_never_sees_test_labels(write_jsonl):
    rows = make_logs(600)
    t = run(str(write_jsonl(rows, "a.jsonl")), HashingFeatures())["tasks"]["ticket_team"]
    assert f"{t['l2']:g}" == min(t["calib_nll_by_l2"], key=t["calib_nll_by_l2"].get)
    # Rewrite every test-slice label (the newest 15%): the choice and the calib NLLs must not move.
    names = ["billing", "technical", "fraud"]
    flipped = rows[:510] + [{**r, "label": names[(names.index(r["label"]) + 1) % 3]} for r in rows[510:]]
    t2 = run(str(write_jsonl(flipped, "b.jsonl")), HashingFeatures())["tasks"]["ticket_team"]
    assert t2["l2"] == t["l2"] and t2["calib_nll_by_l2"] == t["calib_nll_by_l2"]
    assert t2["accuracy_vs_teacher"] < t["accuracy_vs_teacher"]  # the flip did reach the test slice


def test_baseline_skips_task_without_enough_data(write_jsonl):
    rows = [{**r, "label": "billing"} for r in make_logs(20)]
    assert "skipped" in run(str(write_jsonl(rows)), HashingFeatures())["tasks"]["ticket_team"]
