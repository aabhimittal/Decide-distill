# Decide-distill

Turn your LLM decisions into a local decision model. See [SPEC.md](SPEC.md) for the full design.

Status: **M0** (log schema, ingestion, dedupe, time split), **M1** (linear head on frozen d1-3B features,
temperature scaling, accuracy + ECE), **M3** (confidence-gated serving with teacher fallback) and **M4**
(drift alarms, active learning, simulated drift scenario). **M2** (LoRA) is not built yet.

```
logs.jsonl ─▶ baseline --save ─▶ students/ ─▶ serve.app (gate) ─▶ answer
                    ▲                               │ conf < τ ─▶ teacher ─▶ queue.jsonl (log format)
                    └───────── retrain on logs + queue ◀────────┘
                                                    └ /monitor: PSI, shadow agreement, alarms
```

## Install

```bash
pip install -e ".[dev]"        # core: numpy only (+ pytest, fastapi, httpx for the tests)
pip install -e ".[d1]"         # + torch, transformers>=5.14 for d1-3B features
pip install -e ".[serve]"      # + fastapi, uvicorn for the HTTP gate
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

**Choosing the head.** The calibration slice picks the head from {zero init, d1 readout prior} × six L2
strengths, by NLL after temperature scaling; the test slice plays no part (`init`, `l2`, `calib_nll` in the
report). The prior starts the head at d1's `lm_head` rows for each option token and shrinks towards them, so
with no training it reproduces zero-shot exactly; it is only on offer with d1 features. On AG News (600 logs,
90 test rows):

| | init / l2 | calib NLL | test acc vs human | test ECE after |
|---|---|---|---|---|
| d1 zero-shot | – | – | 0.822 | 0.086 |
| **chosen**: zero init | 0.1 | **0.480** | 0.867 | 0.039 |
| readout prior | 1.0 | 0.493 | 0.911 | 0.050 |

Calibration picks zero init. The prior's 4-row test lead is inside the noise of 90 rows, and overruling the
calibration slice because of it would be picking on test.

**Gate.** τ is chosen on calib as the threshold that accepts the most rows while those rows agree with the
teacher at ≥ `--target-agreement` (default 0.95). `gate.calib` is that choice; `gate.test` is the honest
estimate: fallback rate, agreement on accepted rows, agreement and human accuracy of the whole gated system,
and cost per 1k decisions (teacher only vs gated, from the logs' `cost_usd`; the local student's cost isn't
counted). `--save students/` writes one deployable student per task (head, T, τ, drift reference).

Simulated logs from AG News for a quick end-to-end run: `python scripts/simulate_logs.py > logs.jsonl`.

## Serve (M3)

```bash
python -m distill.serve.app --students students/ --teacher mypkg.llm:ask \
    --queue queue.jsonl --pool pool.jsonl --shadow 0.05
curl -s localhost:8000/decide -d '{"task": "ticket_team", "state": "I was charged twice"}'
curl -s localhost:8000/monitor
```

`--teacher` is your existing LLM call as `ask(task, state, question) -> label`. Every teacher answer
(fallbacks, and the random `--shadow` sample of student answers) is appended to `--queue` as a log row, so
the next `baseline` run trains on it directly. A teacher answer outside the question's options is a 502 and
is never queued.

`python -m distill.serve.replay students/ new_logs.jsonl` sends logged rows through the same gate, one at a
time, with the logged label as the teacher: the check that served fallback rate and agreement match the
offline estimate.

## Monitor and active learning (M4)

- **PSI** over the student's predicted classes in a rolling window vs the pre-deployment reference; alarm
  at > 0.2 once 100 predictions are in. Needs no labels, so it sees every request; it catches inputs moving.
- **Shadow agreement**: the random shadow sample's agreement with the teacher; alarm below the offline
  agreement − 0.1 once 30 shadow labels are in. It catches the right answer moving, which PSI can't see.
- **Active learning** (`distill.monitor.active.prioritize`): which unlabeled requests (`--pool`) to send to
  the teacher. Normalized entropy plus how under-represented the request's cluster (cosine k-means) is in
  training. Never use it for the shadow sample: agreement needs a random sample.

`scripts/simulate_drift.py` runs all of it on AG News; results are in the next section.

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
