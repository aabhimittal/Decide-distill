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

    # zero_shot over our features (lm_head option rows + readout) must reproduce d1's own system_one answer.
    ours = np.exp(f.zero_shot(X, QUESTION))
    for state, p in zip(states, ours):
        theirs = f.model.system_one(state, {"q": QUESTION})["answers"]["q"]["probabilities"]
        assert np.allclose(p, list(theirs.values()), atol=0.02)

    # The linear readout prior (primary option tokens) picks the same answers as max-pooled zero-shot.
    assert ((X @ f.readout_weights(QUESTION)).argmax(1) == ours.argmax(1)).all()
