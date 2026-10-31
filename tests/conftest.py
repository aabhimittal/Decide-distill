import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

QUESTION = {
    "type": "choice",
    "instructions": "Which team should handle this?",
    "criteria": {"billing": "Charges, refunds", "technical": "App faults", "fraud": "Unauthorised use"},
}
VOCAB = {
    "billing": ["charged", "refund", "invoice", "twice", "payment"],
    "technical": ["crash", "error", "login", "settings", "slow"],
    "fraud": ["stolen", "unknown", "hacked", "suspicious", "unauthorised"],
}
FILLER = ["please", "help", "my", "account", "today", "the", "app", "i", "was", "hello"]


def make_logs(n=600, teacher_noise=0.1, seed=0):
    """Simulated teacher logs: the human label is the truth, the teacher flips `teacher_noise` of them."""
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    names = list(VOCAB)
    rows = []
    for i in range(n):
        truth = names[rng.integers(3)]
        words = list(rng.choice(VOCAB[truth], 2)) + list(rng.choice(FILLER, 4)) + [f"ref{i}"]
        rng.shuffle(words)
        teacher = names[rng.integers(3)] if rng.random() < teacher_noise else truth
        rows.append({
            "id": f"r{i}", "ts": (t0 + timedelta(minutes=i)).isoformat(), "task": "ticket_team",
            "state": " ".join(words), "question": QUESTION, "label": teacher, "human_label": truth,
            "teacher_confidence": 0.9, "cost_usd": 0.0004,
        })
    return rows


@pytest.fixture
def write_jsonl(tmp_path):
    def write(rows, name="logs.jsonl"):
        path = tmp_path / name
        path.write_text("".join((r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
        return path
    return write
