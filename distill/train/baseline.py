"""M1: linear-head baseline, per task family, with accuracy and ECE before/after temperature scaling.

    python -m distill.train.baseline logs.jsonl --features d1      # frozen d1-3B features
    python -m distill.train.baseline logs.jsonl --features hashing # model-free floor
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from distill.calibrate.temperature import ece, fit_temperature, nll, softmax
from distill.dataset.build import by_task, dedupe, time_split
from distill.logs.ingest import Decision, options, read_many
from distill.serve.gate import choose_tau, choose_tau_vs_human, gate_metrics
from distill.serve.student import Student
from distill.train.linear_head import LinearHead

L2_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
GATE_CURVE = (0.8, 0.85, 0.9, 0.95, 0.99)
# "auto": τ that makes the gated system most accurate vs human labels on calib (one-SE rule, see
# choose_tau_vs_human). Without enough human labels it falls back to an agreement target: agreement with a noisy
# teacher is capped by its noise (at 0.95 a 10%-noise teacher got ~96% of AG News traffic back, though the
# student was more accurate), so the fallback is 0.85. Read the printed curve either way.
DEFAULT_TARGET = "auto"
AGREEMENT_FALLBACK = 0.85


def _acc(pred: np.ndarray, labels: list[str | None], classes: list[str]) -> tuple[float | None, int]:
    """Accuracy of `pred` against the labels that exist, and how many exist."""
    idx = [i for i, l in enumerate(labels) if l is not None]
    if not idx:
        return None, 0
    return float(np.mean([classes[pred[i]] == labels[i] for i in idx])), len(idx)


def evaluate_task(rows: list[Decision], extract, fractions=(0.7, 0.15, 0.15), target_agreement=DEFAULT_TARGET,
                  save_dir=None) -> dict:
    classes = options(rows[0].question)
    split = time_split(rows, fractions)
    sizes = {"train": len(split.train), "calib": len(split.calib), "test": len(split.test)}
    if not split.calib or not split.test or len({d.label for d in split.train}) < 2:
        return {"skipped": "needs ≥2 teacher classes in train and non-empty calib/test", "n": sizes}

    def xy(part):
        return extract([d.state for d in part], rows[0].question), np.array([classes.index(d.label) for d in part])

    (Xtr, ytr), (Xc, yc), (Xte, yte) = xy(split.train), xy(split.calib), xy(split.test)
    humans = [d.human_label for d in split.test]

    def scores(logits_calib, logits_test) -> dict:  # reports on test; calib only fits T
        T = fit_temperature(logits_calib, yc)
        raw, cal = softmax(logits_test), softmax(logits_test / T)
        pred = raw.argmax(1)  # temperature never changes the argmax
        return {"accuracy_vs_teacher": float((pred == yte).mean()),
                "accuracy_vs_human": _acc(pred, humans, classes)[0],
                "temperature": T, "ece_before": ece(raw, yte), "ece_after": ece(cal, yte)}

    q = rows[0].question
    prior = extract.readout_weights(q) if hasattr(extract, "readout_weights") else None
    head, init, calib_nll = select_head(Xtr, ytr, Xc, yc, len(classes), prior)
    Lc, Lt = head.logits(Xc), head.logits(Xte)
    gate = gate_and_student(rows, head, Lc, yc, Lt, yte, humans, target_agreement,
                            {**getattr(extract, "meta", {}), "l2": head.l2, "init": init}, save_dir,
                            [d.human_label for d in split.calib])
    teacher_vs_human, n_human = _acc(yte, humans, classes)
    report = {
        "n": sizes,
        "classes": classes,
        **scores(Lc, Lt),
        "init": init,
        "l2": head.l2,
        "calib_nll": calib_nll,
        "teacher_accuracy_vs_human": teacher_vs_human,  # the ceiling the student is learning towards
        "n_human_labels": n_human,
        "gate": gate,
    }
    if hasattr(extract, "zero_shot"):  # the untrained model's own answer: the bar a student must clear
        report["zero_shot"] = scores(extract.zero_shot(Xc, q), extract.zero_shot(Xte, q))
    return report


def gate_and_student(rows, head, Lc, yc, Lt, yte, humans, target_agreement, meta, save_dir=None,
                     calib_humans=None) -> dict:
    """Temperature, τ and the gate report from a student's calib/test logits; saves the student if asked.
    Shared by the linear head and LoRA so both are gated, reported and served the same way.
    `target_agreement`: a float, or "auto" to choose τ on calib human labels (`calib_humans`)."""
    q, task = rows[0].question, rows[0].task
    classes = options(q)
    T = fit_temperature(Lc, yc)
    Pc, Pt = softmax(Lc / T), softmax(Lt / T)
    # τ from calib (yet another use of that slice, after l2, init and T); the test block is the honest estimate.
    to_idx = lambda hs: np.array([classes.index(h) if h is not None else -1 for h in hs], dtype=int)
    human_idx, calib_human = to_idx(humans), to_idx(calib_humans or [None] * len(yc))
    chosen, tau = target_agreement, None
    if target_agreement == "auto":
        tau = choose_tau_vs_human(Pc.max(1), Pc.argmax(1), yc, calib_human)
        if tau is None:
            chosen = AGREEMENT_FALLBACK  # too few human labels on calib to optimise accuracy against
    if tau is None:
        tau = choose_tau(Pc.max(1), Pc.argmax(1) == yc, chosen)
    costs = [d.cost_usd for d in rows if d.cost_usd is not None]
    cost = float(np.mean(costs)) if costs else None

    def test_gate(t):
        return gate_metrics(Pt.max(1), Pt.argmax(1), yte, t, human_idx, cost)

    gate = {"target_agreement": chosen,  # "auto", or the agreement target actually used
            "calib": gate_metrics(Pc.max(1), Pc.argmax(1), yc, tau, calib_human),
            "test": test_gate(tau),
            # The target is a business choice, not a model parameter: show what each one buys on test. With a
            # noisy teacher, high agreement targets fall back to it even where the student is more accurate.
            "curve": [{"target": g, **test_gate(choose_tau(Pc.max(1), Pc.argmax(1) == yc, g))}
                      for g in GATE_CURVE]}
    if save_dir is not None:
        ref = np.bincount(np.concatenate([Pc.argmax(1), Pt.argmax(1)]), minlength=len(classes))
        Student(task, q, classes, head, T, tau, ref.tolist(), gate["calib"]["accepted_agreement"],
                {**meta, "teacher_cost_usd": cost}, gate["calib"]["rejected_agreement"]).save(save_dir)
    return gate


def select_head(Xtr, ytr, Xc, yc, n_classes: int, prior=None) -> tuple[LinearHead, str, dict]:
    """The head with the lowest calib NLL after temperature scaling, over {zero, prior} × L2_GRID.

    NLL after scaling, so confidence scale (which T fixes anyway) doesn't decide it. Calib also fits T later:
    calib-side numbers are mildly optimistic, test stays untouched.
    """
    best, table = None, {}
    for init, P in [("zero", None)] + ([("prior", prior)] if prior is not None else []):
        for l2 in L2_GRID:
            h = LinearHead(l2=l2).fit(Xtr, ytr, n_classes, P)
            Lc = h.logits(Xc)
            table[f"{init}/{l2:g}"] = v = nll(Lc, yc, fit_temperature(Lc, yc))
            if best is None or v < best[2]:
                best = (h, init, v)
    return best[0], best[1], table


def run(paths, extract, target_agreement: float = DEFAULT_TARGET, save_dir=None) -> dict:
    """`paths`: one log file or several (e.g. logs + the serving queue)."""
    records, errors = read_many(paths)
    unique = dedupe(records)
    return {
        "ingest": {"records": len(records), "rejected": len(errors), "after_dedupe": len(unique),
                   "errors": errors[:20]},
        "tasks": {task: evaluate_task(rows, extract, target_agreement=target_agreement, save_dir=save_dir)
                  for task, rows in sorted(by_task(unique).items())},
    }


def parse_target(s: str):
    """`auto`, or an agreement target in (0, 1]."""
    if s == "auto":
        return s
    v = float(s)
    if not 0 < v <= 1:
        raise argparse.ArgumentTypeError("must be 'auto' or a number in (0, 1]")
    return v


def make_extractor(features: str, model: str = "LiquidAI/d1-3B", device=None, dtype=None):
    if features == "d1":
        import torch

        from distill.train.features import D1Features

        return D1Features(model, device, dtype and getattr(torch, dtype))
    from distill.train.features import HashingFeatures

    return HashingFeatures()


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs", nargs="+", help="log files, e.g. logs.jsonl queue.jsonl")
    p.add_argument("--features", choices=["d1", "hashing"], default="d1")
    p.add_argument("--model", default="LiquidAI/d1-3B")
    p.add_argument("--device")
    # fp32 (the card's CPU default) needs ~12.5 GB for d1-3B; use bfloat16 on smaller machines.
    p.add_argument("--dtype", choices=["float32", "bfloat16"])
    p.add_argument("--target-agreement", type=parse_target, default=DEFAULT_TARGET,
                   help="gate: the student answers only where calib says it agrees with the teacher this often")
    p.add_argument("--save", metavar="DIR", help="write one deployable student per task here")
    args = p.parse_args(argv)
    report = run(args.logs, make_extractor(args.features, args.model, args.device, args.dtype),
                 args.target_agreement, args.save)
    for task, t in report["tasks"].items():
        if "gate" in t:
            print(format_curve(task, t["gate"]), file=sys.stderr)
    json.dump(report, sys.stdout, indent=2)
    print()


def format_curve(task: str, gate: dict) -> str:
    """The gate's trade-off on test, one line per agreement target, the chosen one marked."""
    fmt = lambda v, spec: "   –" if v is None else format(v, spec)
    chosen = gate["target_agreement"]
    t = gate["test"]
    lines = [f"[{task}] gate on test: target {chosen}, τ={t['tau']:.3f}, fallback {t['fallback_rate']:.1%}, "
             f"acc vs human {fmt(t.get('system_accuracy_vs_human'), '.3f').strip()}",
             "  curve over fixed agreement targets (* = the one used):",
             "  target  fallback  agree(accepted)  acc vs human  cost/1k gated"]
    for c in gate["curve"]:
        mark = " *" if c["target"] == gate["target_agreement"] else "  "
        lines.append(f"{mark}{c['target']:5.2f}    {c['fallback_rate']:6.1%}   {fmt(c['accepted_agreement'], '13.3f')}"
                     f"  {fmt(c.get('system_accuracy_vs_human'), '12.3f')}  {fmt(c.get('cost_per_1k_gated'), '13.4f')}")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
