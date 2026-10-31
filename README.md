# Decide-distill

Turn your LLM decisions into a local decision model. See [SPEC.md](SPEC.md) for the full design.

Status: **M0** (log schema, ingestion, dedupe, time split) and **M1** (linear head on frozen d1-3B
features, temperature scaling, accuracy + ECE report). M2–M4 (LoRA, gated serving, active learning, drift)
are not built yet.

## Install

```bash
pip install -e ".[dev]"        # core: numpy only
pip install -e ".[d1]"         # + torch, transformers>=5.14 for d1-3B features
```

## Log format (JSONL, one teacher decision per line)

```json
{"id": "t-1", "ts": "2026-03-01T12:00:00Z", "task": "ticket_team",
 "state": "I was charged twice",
 "question": {"type": "choice", "instructions": "Which team?",
              "criteria": {"billing": "Charges", "technical": "Faults"}},
 "label": "billing", "human_label": "billing", "teacher_confidence": 0.93, "cost_usd": 0.0004}
```

`question` uses d1's Decision Index schema (`noul` | `choice` | `score`), so logged questions replay
against d1 unchanged. `label` is an option name: `yes`/`no`, a `criteria` key, or a score level `"0"`…`"9"`.
Timestamps must carry a timezone. Bad lines are skipped and reported with their line numbers.

## Run the M1 baseline

```bash
python -m distill.train.baseline logs.jsonl --features d1       # frozen d1-3B features
python -m distill.train.baseline logs.jsonl --features hashing  # model-free floor
```

For each task family: dedupe (keeps the earliest copy), a 70/15/15 time split, a logistic-regression head on
train, a temperature fitted on calib, and on test: accuracy vs teacher, accuracy vs human labels (when
present, next to the teacher's own accuracy vs human), and ECE before/after scaling.

Simulated logs from AG News for a quick end-to-end run: `python scripts/simulate_logs.py > logs.jsonl`.

## How d1 features are extracted

The d1-3B model card exposes answers only (`system_one`), with no feature API. Its remote code renders
`state + question + "<|im_start|>assistant\n"` and reads the decision off `lm_head` at the last token. We take
the backbone's final hidden state at that token, the vector d1's own answer is a linear readout of.
`tests/test_d1_features.py` (run with `DISTILL_D1=1`) checks that `lm_head` over our features reproduces
`system_one`'s probabilities.

## License note on d1 weights

d1-3B is under the **LFM Open License v1.0**: derivatives are allowed (keep the license, mark changes), but
commercial use is licensed only for entities under **$10M annual revenue**. Check this before deploying a
fine-tuned d1 commercially. This repository's code is MIT ([LICENSE](LICENSE)).

## Tests

```bash
pytest                                       # fast, no model download
DISTILL_D1=1 pytest tests/test_d1_features.py  # loads d1-3B (~6 GB)
```
