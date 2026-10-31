"""M3 acceptance: replay logged decisions through the real gate and compare with the offline estimate.

    python -m distill.serve.replay students/ new_logs.jsonl

Each row goes through `Gateway.decide` one at a time, loaded from disk exactly as served; the "teacher" answers
with the logged label, so fallback rate and agreement are measured, not assumed.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from distill.logs.ingest import read_jsonl
from distill.serve.gate import Gateway, gate_metrics
from distill.serve.student import extractor_for, load_students


def replay(students, extract, records) -> dict:
    by_key = {(d.task, json.dumps(d.state, sort_keys=True)): d for d in records}
    gw = Gateway(students, extract, lambda task, state, q: by_key[(task, json.dumps(state, sort_keys=True))].label)
    out = {}
    for task, s in students.items():
        rows = [d for d in records if d.task == task]
        if not rows:
            continue
        res = [gw.decide(task, d.state) for d in rows]
        idx = {c: i for i, c in enumerate(s.classes)}
        costs = [d.cost_usd for d in rows if d.cost_usd is not None]
        out[task] = {"n": len(rows), **gate_metrics(
            np.array([r["confidence"] for r in res]), np.array([idx[r["student_answer"]] for r in res]),
            np.array([idx[d.label] for d in rows]), s.tau,
            np.array([idx[d.human_label] if d.human_label is not None else -1 for d in rows]),
            float(np.mean(costs)) if costs else None),
            "monitor": gw.status()[task]}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("students")
    p.add_argument("logs")
    p.add_argument("--device")
    args = p.parse_args(argv)
    students = load_students(args.students)
    records, errors = read_jsonl(args.logs)
    if errors:
        print(f"{len(errors)} rejected log lines, first: {errors[0]}", file=sys.stderr)
    json.dump(replay(students, extractor_for(students, args.device), records), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
