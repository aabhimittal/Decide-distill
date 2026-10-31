"""Real d1-3B check (6 GB download). Run with DISTILL_D1=1 pytest tests/test_d1_features.py."""

import os

import numpy as np
import pytest

from conftest import QUESTION

pytestmark = pytest.mark.skipif(not os.environ.get("DISTILL_D1"), reason="set DISTILL_D1=1 to load d1-3B")


def test_d1_features_are_what_d1_decides_from():
    import torch

    from distill.train.features import D1Features

    f = D1Features(dtype=torch.bfloat16)
    states = ["I was charged twice this month, please refund one.", "The app crashes when I open settings."]
    X = f(states, QUESTION)
    assert X.shape == (2, 2048) and np.isfinite(X).all()

    # lm_head over our features, read at the option tokens, must reproduce d1's own system_one answer.
    engine = f.engine
    prompt = __import__(type(engine).__module__.rsplit(".", 1)[0] + ".prompt", fromlist=["readout"])
    for state, x in zip(states, X):
        logits = f.model.lm_head(torch.tensor(x, dtype=f.model.lm_head.weight.dtype)).float()
        ours = prompt.readout(engine.tokenizer, f.as_question(QUESTION), logits - torch.logsumexp(logits, -1))
        theirs = f.model.system_one(state, {"q": QUESTION})["answers"]["q"]["probabilities"]
        assert np.allclose(ours, list(theirs.values()), atol=0.02)
