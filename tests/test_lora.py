from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from distill.train.lora import option_logits  # noqa: E402


def test_training_readout_max_pools_each_option_like_zero_shot():
    """LoRA trains on the same readout zero_shot scores: max over each option's token forms."""
    rng = np.random.default_rng(0)
    h, W = torch.tensor(rng.normal(size=(3, 8))), torch.tensor(rng.normal(size=(5, 8)))
    groups = [[10, 11], [12], [13, 14]]  # option 0 has two forms ("A", " A"), option 1 one, option 2 two
    got = option_logits(SimpleNamespace(torch=torch), h, groups, W).numpy()
    full = (h @ W.T).numpy()
    expected = np.stack([full[:, 0:2].max(1), full[:, 2], full[:, 3:5].max(1)], axis=1)
    assert np.allclose(got, expected)


def test_readout_gradients_reach_only_the_winning_form():
    h = torch.zeros(1, 2, requires_grad=True)
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    out = option_logits(SimpleNamespace(torch=torch), h + torch.tensor([[2.0, 0.0]]), [[0, 1], [2]], W)
    out[0, 0].backward()  # option 0's best form is row 0 (2 > 0), so the gradient is W[0]
    assert torch.allclose(h.grad, torch.tensor([[1.0, 0.0]]))
