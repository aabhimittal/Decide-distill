"""M1: linear-head baseline, per task family, with accuracy and ECE before/after temperature scaling.

    python -m distill.train.baseline logs.jsonl --features d1      # frozen d1-3B features
    python -m distill.train.baseline logs.jsonl --features hashing # model-free floor
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from distill.calibrate.temperature import ece, fit_temperature, softmax
from distill.dataset.build import by_task, dedupe, time_split
from distill.logs.ingest import Decision, options, read_jsonl
from distill.train.linear_head import LinearHead


def _acc(pred: np.ndarray, labels: list[str | None], classes: list[str]) -> tuple[float | None, int]:
    """Accuracy of `pred` against the labels that exist, and how many exist."""
    idx = [i for i, l in enumerate(labels) if l is not None]
    if not idx:
        return None, 0
    return float(np.mean([classes[pred[i]] == labels[i] for i in idx])), len(idx)


def evaluate_task(rows: list[Decision], extract, fractions=(0.7, 0.15, 0.15), use_prior: bool = False) -> dict:
    classes = options(rows[0].question)
    split = time_split(rows, fractions)
    sizes = {"train": len(split.train), "calib": len(split.calib), "test": len(split.test)}
    if not split.calib or not split.test or len({d.label for d in split.train}) < 2:
        return {"skipped": "needs ≥2 teacher classes in train and non-empty calib/test", "n": sizes}

    def xy(part):
        return extract([d.state for d in part], rows[0].question), np.array([classes.index(d.label) for d in part])

    (Xtr, ytr), (Xc, yc), (Xte, yte) = xy(split.train), xy(split.calib), xy(split.test)
    humans = [d.human_label for d in split.test]

    def scores(logits_calib, logits_test) -> dict:
        T = fit_temperature(logits_calib, yc)
        raw, cal = softmax(logits_test), softmax(logits_test / T)
        pred = raw.argmax(1)  # temperature never changes the argmax
        return {"accuracy_vs_teacher": float((pred == yte).mean()),
                "accuracy_vs_human": _acc(pred, humans, classes)[0],
                "temperature": T, "ece_before": ece(raw, yte), "ece_after": ece(cal, yte)}

    prior = extract.readout_weights(rows[0].question) if use_prior else None
    head = LinearHead().fit(Xtr, ytr, len(classes), prior)
    teacher_vs_human, n_human = _acc(yte, humans, classes)
    report = {
        "n": sizes,
        "classes": classes,
        **scores(head.logits(Xc), head.logits(Xte)),
        "teacher_accuracy_vs_human": teacher_vs_human,  # the ceiling the student is learning towards
        "n_human_labels": n_human,
    }
    if hasattr(extract, "zero_shot"):  # the untrained model's own answer: the bar a student must clear
        q = rows[0].question
        report["zero_shot"] = scores(extract.zero_shot(Xc, q), extract.zero_shot(Xte, q))
    return report


def run(path: str, extract, use_prior: bool = False) -> dict:
    records, errors = read_jsonl(path)
    unique = dedupe(records)
    return {
        "ingest": {"records": len(records), "rejected": len(errors), "after_dedupe": len(unique),
                   "errors": errors[:20]},
        "tasks": {task: evaluate_task(rows, extract, use_prior=use_prior)
                  for task, rows in sorted(by_task(unique).items())},
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs")
    p.add_argument("--features", choices=["d1", "hashing"], default="d1")
    p.add_argument("--model", default="LiquidAI/d1-3B")
    p.add_argument("--device")
    # fp32 (the card's CPU default) needs ~12.5 GB for d1-3B; use bfloat16 on smaller machines.
    p.add_argument("--dtype", choices=["float32", "bfloat16"])
    # Off by default: on AG News it matched zero-init accuracy and calibrated worse (see README).
    p.add_argument("--prior", action="store_true", help="start the d1 head from d1's readout and shrink towards it")
    args = p.parse_args(argv)
    if args.prior and args.features != "d1":
        p.error("--prior needs --features d1")
    if args.features == "d1":
        import torch

        from distill.train.features import D1Features

        extract = D1Features(args.model, args.device, args.dtype and getattr(torch, args.dtype))
    else:
        from distill.train.features import HashingFeatures

        extract = HashingFeatures()
    json.dump(run(args.logs, extract, args.prior), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
