"""Turn AG News into simulated teacher decision logs (SPEC §7).

    pip install pandas pyarrow
    python scripts/simulate_logs.py --n 600 --noise 0.1 > agnews_logs.jsonl

The human label is AG News' gold label. The "teacher" agrees with it except on a `noise` fraction, where it
picks a different class at random. Timestamps follow dataset order, one minute apart.
shortcut: uniform label noise, not a real LLM teacher; a real teacher's errors cluster on hard inputs, so
this overstates how learnable the teacher is. Replace with logged calls to a real LLM for M3 numbers.
"""

import argparse
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

URL = "https://huggingface.co/datasets/fancyzhx/ag_news/resolve/main/data/test-00000-of-00001.parquet"
QUESTION = {
    "type": "choice",
    "instructions": "Which section of a news site does this article belong in?",
    "criteria": {"world": "World news", "sports": "Sports", "business": "Business", "scitech": "Science and technology"},
}

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=600)
p.add_argument("--noise", type=float, default=0.1)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
names = list(QUESTION["criteria"])
t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
for i, row in pd.read_parquet(URL).head(args.n).reset_index(drop=True).iterrows():
    human = names[int(row["label"])]
    teacher = human if rng.random() >= args.noise else str(rng.choice([c for c in names if c != human]))
    print(json.dumps({
        "id": f"agnews-{i}", "ts": (t0 + timedelta(minutes=i)).isoformat(), "task": "news_section",
        "state": row["text"], "question": QUESTION, "label": teacher, "human_label": human,
    }))
