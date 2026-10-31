"""Frozen feature extractors: `extractor(states, question) -> (N, D) float32`.

D1Features. The d1-3B card (LiquidAI/d1-3B, transformers>=5.14, trust_remote_code) exposes only answers,
`system_one` / `system_one_batch`, with no feature API. Its remote code (`runner.py`, `prompt.py`) renders
`state + question + "<|im_start|>assistant\\n"` and reads the decision off `lm_head` at the last token, the
"answer slot". So the feature here is the backbone's final-norm hidden state at that slot: the vector d1's
own decision is a linear readout of. `hybrid.LanguageModel.forward` ignores `output_hidden_states` and
returns `last_hidden_state` (already final-normed), which is what we take.

HashingFeatures. A model-free bag of n-grams: deterministic, runs anywhere, used in tests and as the floor
d1 features must beat.
"""

from __future__ import annotations

import json
import sys
import zlib

import numpy as np


class HashingFeatures:
    def __init__(self, dim: int = 4096):
        self.dim = dim

    def __call__(self, states, question=None) -> np.ndarray:
        X = np.zeros((len(states), self.dim), dtype=np.float32)
        for i, s in enumerate(states):
            words = (s if isinstance(s, str) else json.dumps(s, sort_keys=True)).lower().split()
            for g in words + [a + " " + b for a, b in zip(words, words[1:])]:
                X[i, zlib.crc32(g.encode()) % self.dim] += 1  # crc32, not hash(): hash() is salted per process
        X = np.log1p(X)
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


class D1Features:
    def __init__(self, model_id: str = "LiquidAI/d1-3B", device: str | None = None, dtype=None):
        import torch
        from transformers import AutoModel

        self.torch = torch
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or (torch.float32 if device == "cpu" else torch.bfloat16)  # the card's recommendation
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).to(device).eval()
        self.engine = self.model.engine  # d1's SystemOne: its render() is the exact prompt d1 decides from
        prompt = sys.modules[type(self.engine).__module__.rsplit(".", 1)[0] + ".prompt"]
        self.as_question = prompt.as_question
        self.device = device

    def __call__(self, states, question) -> np.ndarray:
        q = self.as_question(question)
        lm = self.model.model.language_model
        out = []
        with self.torch.inference_mode():
            # shortcut: one prompt per pass; pack rows with `lm.answer(...)` like runner.py for GPU throughput.
            for s in states:
                ids = self.engine.tokenizer.encode(self.engine.render(s, q), add_special_tokens=False)
                h = lm(input_ids=self.torch.tensor([ids], device=self.device)).last_hidden_state[0, -1]
                out.append(h.float().cpu().numpy())
        return np.stack(out)
