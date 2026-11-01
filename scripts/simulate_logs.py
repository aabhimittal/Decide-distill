"""Turn AG News into simulated teacher decision logs (SPEC §7).

    pip install pandas pyarrow
    python scripts/simulate_logs.py --n 600 --noise 0.1 > agnews_logs.jsonl
    python scripts/simulate_logs.py --policy house --n 800 > house_logs.jsonl
    python scripts/simulate_logs.py --policy house --offset 800 --n 300 > house_fresh.jsonl

--policy topic: the label is AG News' topic, which d1 reads well zero-shot.
--policy house: a routing policy d1 cannot know. Options are opaque ("Queue 1".."Queue 3") and the rule is
  hidden: world → q1, sports → q3, science/tech → q2, business → q2 if the article contains a digit, else q1.
  Zero-shot d1 is near chance; the business-digit split is an interaction a student has to learn.

The human label is the policy applied to the gold topic; the teacher agrees except on a `noise` fraction, where
it picks another option at random. `--offset` skips rows, for fresh traffic that never overlaps training.
shortcut: uniform label noise, not a real LLM teacher; a real teacher's errors cluster on hard inputs, so
this overstates how learnable the teacher is. Replace with logged calls to a real LLM for real numbers.
"""

import argparse
import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet"
TOPICS = ["world", "sports", "business", "scitech"]
POLICIES = {
    "topic": ("news_section", {
        "type": "choice",
        "instructions": "Which section of a news site does this article belong in?",
        "criteria": {"world": "World news", "sports": "Sports", "business": "Business",
                     "scitech": "Science and technology"}}),
    "house": ("house_routing", {
        "type": "choice",
        "instructions": "Which queue does this article go to under our routing policy?",
        "criteria": {"q1": "Queue 1", "q2": "Queue 2", "q3": "Queue 3"}}),
}


def house(topic: str, text: str) -> str:
    if topic == "business":
        return "q2" if re.search(r"\d", text) else "q1"
    return {"world": "q1", "sports": "q3", "scitech": "q2"}[topic]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", choices=list(POLICIES), default="topic")
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--cost", type=float, default=0.0004, help="teacher cost per call, USD")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    task, question = POLICIES[args.policy]
    names = list(question["criteria"])
    rng = np.random.default_rng(args.seed + args.offset)  # fresh traffic gets its own noise draw
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    df = pd.read_parquet(URL).iloc[args.offset: args.offset + args.n].reset_index(drop=True)
    for i, row in df.iterrows():
        topic = TOPICS[int(row["label"])]
        human = topic if args.policy == "topic" else house(topic, row["text"])
        teacher = human if rng.random() >= args.noise else str(rng.choice([c for c in names if c != human]))
        print(json.dumps({
            "id": f"{task}-{args.offset + i}", "ts": (t0 + timedelta(minutes=args.offset + i)).isoformat(),
            "task": task, "state": row["text"], "question": question, "label": teacher, "human_label": human,
            "cost_usd": args.cost,
        }))


if __name__ == "__main__":
    main()
