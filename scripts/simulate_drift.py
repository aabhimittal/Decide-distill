"""M3 + M4 acceptance on AG News with a simulated teacher (SPEC §7, §8).

    pip install pandas pyarrow
    python scripts/simulate_drift.py --features d1 --dtype bfloat16 > drift_report.json

Era A (training): science/tech is rare (5%). Then four streams through the real Gateway:
  stable   same mix as era A, fresh rows         → M3: replayed gate ≈ offline estimate; no alarms
  surge    science/tech at 55%                   → PSI alarm (covariate drift)
  concept  stable inputs, teacher now files business as world → shadow-agreement alarm, PSI quiet
  active   label B surge rows chosen by prioritize() vs at random, retrain, score on held-out surge rows

The teacher is AG News' gold label with 10% uniform noise. shortcut: real LLM errors cluster on hard inputs.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from distill.dataset.build import by_task, time_split
from distill.logs.ingest import parse
from distill.monitor.active import prioritize
from distill.serve.gate import Gateway
from distill.serve.replay import replay
from distill.serve.student import load_students
from distill.train.baseline import evaluate_task, make_extractor
from distill.train.linear_head import LinearHead

URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet"
QUESTION = {
    "type": "choice",
    "instructions": "Which section of a news site does this article belong in?",
    "criteria": {"world": "World news", "sports": "Sports", "business": "Business", "scitech": "Science and technology"},
}
NAMES = list(QUESTION["criteria"])
RARE = [0.3167, 0.3167, 0.3166, 0.05]
SURGE = [0.15, 0.15, 0.15, 0.55]


class Cached:
    """Extract each distinct state once; d1 on CPU is ~0.5 s a row and the streams overlap."""

    def __init__(self, f):
        self.f, self.cache = f, {}

    def __call__(self, states, question):
        key = [json.dumps(s, sort_keys=True) for s in states]
        todo = [s for s, k in zip(states, key) if k not in self.cache]
        if todo:
            for k, x in zip([json.dumps(s, sort_keys=True) for s in todo], self.f(todo, question)):
                self.cache[k] = x
        return np.stack([self.cache[k] for k in key])

    def __getattr__(self, name):  # zero_shot, readout_weights, meta
        return getattr(self.f, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", choices=["d1", "hashing"], default="d1")
    p.add_argument("--dtype")
    p.add_argument("--budgets", default="30,60")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_parquet(URL)
    pools = {c: list(rng.permutation(np.nonzero(df["label"].to_numpy() == c)[0])) for c in range(4)}
    t = [datetime(2026, 1, 1, tzinfo=timezone.utc)]

    def rows(n, mix, prefix):
        out = []
        for c in rng.choice(4, size=n, p=mix):
            i = int(pools[c].pop())
            human = NAMES[c]
            teacher = human if rng.random() >= 0.1 else str(rng.choice([x for x in NAMES if x != human]))
            t[0] += timedelta(minutes=1)
            out.append(parse({"id": f"{prefix}-{i}", "ts": t[0].isoformat(), "task": "news_section",
                              "state": df["text"][i], "question": QUESTION, "label": teacher,
                              "human_label": human, "cost_usd": 0.0004}))
        return out

    era_a, stable, surge = rows(800, RARE, "a"), rows(300, RARE, "s"), rows(600, SURGE, "g")
    extract = Cached(make_extractor(args.features, dtype=args.dtype))
    report = {}

    with TemporaryDirectory() as tmp:
        report["train"] = evaluate_task(by_task(era_a)["news_section"], extract, save_dir=tmp)
        students = load_students(tmp)
    s = students["news_section"]
    report["train"].pop("calib_nll")

    # M3: the offline gate estimate (era A test slice) vs fresh traffic through the served gate.
    report["m3_stable_replay"] = replay(students, extract, stable)["news_section"]

    def stream(rs, teacher):
        gw = Gateway(students, extract, teacher, shadow_rate=1.0)  # shadow everything: small streams
        for d in rs:
            gw.decide(d.task, d.state)
        return gw.status()["news_section"]

    label = {d.state: d.label for d in stable + surge}
    report["m4_surge"] = stream(surge, lambda task, state, q: label[state])
    report["m4_concept"] = stream(stable, lambda task, state, q: "world" if label[state] == "business" else label[state])

    # Active learning: B teacher labels from the surge pool, chosen vs random, then retrain and score.
    split = time_split(by_task(era_a)["news_section"])
    pool, held = surge[:300], surge[300:]
    Xtr, Xp, Xh = (extract([d.state for d in part], QUESTION) for part in (split.train, pool, held))
    ytr, yp, yh = (np.array([NAMES.index(d.label) for d in part]) for part in (split.train, pool, held))
    hh = np.array([NAMES.index(d.human_label) for d in held])
    prior = extract.readout_weights(QUESTION) if s.meta.get("init") == "prior" else None
    P_pool = s.probs(Xp)

    def retrained(idx):
        h = LinearHead(l2=s.head.l2).fit(np.vstack([Xtr, Xp[idx]]), np.r_[ytr, yp[idx]], 4, prior)
        pred = h.logits(Xh).argmax(1)
        return {"vs_teacher": float((pred == yh).mean()), "vs_human": float((pred == hh).mean())}

    al = {"no_new_labels": retrained(np.array([], dtype=int)),
          "pool_scitech_share": float(np.mean([d.human_label == "scitech" for d in pool]))}
    for b in map(int, args.budgets.split(",")):
        picks = {"active": prioritize(P_pool, Xp, Xtr, b), "entropy_only": prioritize(P_pool, Xp, Xtr, b, weight=0.0)}
        al[f"budget_{b}"] = {
            **{name: {**retrained(i), "scitech_share": float(np.mean([pool[j].human_label == "scitech" for j in i]))}
               for name, i in picks.items()},
            "random_mean_of_5": {k: float(np.mean([retrained(np.random.default_rng(r).choice(len(pool), b, replace=False))[k]
                                                   for r in range(5)])) for k in ("vs_teacher", "vs_human")},
        }
    report["m4_active_learning"] = al
    json.dump(report, sys.stdout, indent=2, default=float)
    print()


if __name__ == "__main__":
    main()
