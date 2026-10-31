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

On CPU, pass `--dtype bfloat16` unless you have well over 12.5 GB of free RAM. fp32 d1-3B thrashed a
15 GB machine (600 rows did not finish in an hour; bf16 took 5 minutes).

For each task family: dedupe (keeps the earliest copy), a 70/15/15 time split, a logistic-regression head on
train, a temperature fitted on calib, and on test: accuracy vs teacher, accuracy vs human labels (when
present, next to the teacher's own accuracy vs human), and ECE before/after scaling. With `--features d1`
the report adds a `zero_shot` row: untrained d1's own answer on the same rows, recomputed from the same
features (no extra forward pass). A student that doesn't beat it isn't worth training.

`--prior` starts the d1 head from d1's own readout (the `lm_head` rows of each option's token) and shrinks
towards it instead of towards zero. With no training it reproduces zero-shot exactly.

The head's L2 strength is picked per task from a grid, by NLL after temperature scaling on the calibration
slice; the test slice plays no part (`l2` and `calib_nll_by_l2` in the report). On AG News (600 logs, 90 test
rows):

| | chosen l2 | calib NLL | test acc vs human | test ECE after |
|---|---|---|---|---|
| d1 zero-shot | – | – | 0.822 | 0.086 |
| head, zero init | 0.1 | 0.480 | 0.867 | 0.039 |
| head, `--prior` | 1.0 | 0.493 | 0.911 | 0.050 |

`--prior` stays off by default: calibration NLL slightly prefers zero init, and its 4-row test lead is
inside the noise of 90 rows.

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
