# Decide-distill

Turn your LLM decisions into a local decision model. See [SPEC.md](SPEC.md) for the full design.

Status: every SPEC milestone and the v1 scope. **M0** log schema, ingestion, dedupe, time split. **M1** linear
head on frozen d1-3B features, temperature scaling, accuracy + ECE. **M2** LoRA fine-tune, compared with the
head and servable. **M3** confidence-gated serving with teacher fallback. **M4** drift alarms, active learning,
simulated drift. **v1** active-learning queue CLI and a cost dashboard. Gate target chosen on calib human
labels (`auto`), and results against a real LLM teacher.

```
logs.jsonl ─▶ baseline | lora --save ─▶ students/ ─▶ serve.app (gate) ─▶ answer
                    ▲                                     │ conf < τ ─▶ teacher ─▶ queue.jsonl (log format)
                    │                                     ├ --pool ─▶ monitor.active ─▶ teacher ─▶ queue.jsonl
                    └──── baseline logs.jsonl queue.jsonl ◀┘
                                                          └ /monitor, /dashboard: cost, PSI, agreement, alarms
```

## Install

```bash
pip install -e ".[dev]"        # core: numpy only (+ pytest, fastapi, httpx for the tests)
pip install -e ".[d1]"         # + torch, transformers>=5.14 for d1-3B features
pip install -e ".[serve]"      # + fastapi, uvicorn for the HTTP gate
pip install -e ".[lora]"       # + peft for the LoRA fine-tune
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
python -m distill.train.baseline logs.jsonl queue.jsonl --save students/  # retrain on logs + serving queue
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

**Gate.** By default (`--target-agreement auto`) τ is chosen on calib to make the gated system most accurate
against the human labels there, taking the lowest τ (most student coverage) within one standard error of the
best. With fewer than 20 calib human labels it falls back to an agreement target of 0.85. A number instead
(`--target-agreement 0.9`) picks the τ that accepts the most rows while those rows agree with the teacher at
≥ that rate. `gate.calib` is that choice; `gate.test` is the honest
estimate: fallback rate, agreement on accepted rows, agreement and human accuracy of the whole gated system,
and cost per 1k decisions (teacher only vs gated, from the logs' `cost_usd`; the local student's cost isn't
counted). Every run prints the trade-off curve over targets to stderr, the chosen one marked `*`: the target
is a business choice, and agreement with a noisy teacher is capped by its noise (at 0.95 a 10%-noise teacher
got ~96% of AG News traffic back, though the student was more accurate). `--save students/` writes one
deployable student per task (head, T, τ, drift reference, agreement baselines, teacher cost).

Simulated logs from AG News: `python scripts/simulate_logs.py > logs.jsonl` (topics, which d1 reads well),
or `--policy house` (opaque queue names and a hidden routing rule, which d1 can't know; see Results).

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

**`/dashboard`** (SPEC v1 cost dashboard) shows per task: requests, fallback rate, teacher calls, cost per
1k decisions gated vs teacher-only and the saving, PSI, both agreement streams, and alarms; it refreshes
every 10 s. Live cost counts every teacher call, shadow checks included, which the offline estimate doesn't.

**LoRA students** (`distill.train.lora --save students/`) serve through the same gate. LoRA only changes the
backbone, so the student is d1's own option-token rows as a fixed head over adapted features. The adapter is
saved next to it and loaded in fp32. One process serves one adapter; students trained on different features
are refused, not mixed.

## Monitor and active learning (M4)

- **PSI** over the student's predicted classes in a rolling window vs the pre-deployment reference; alarm
  at > 0.2 once 100 predictions are in. Needs no labels, so it sees every request; it catches inputs moving.
- **Shadow agreement**: the random shadow sample's agreement with the teacher; alarm below the offline
  agreement − 0.1 once 30 shadow labels are in. It catches the right answer moving, which PSI can't see.
- **Active learning** (`distill.monitor.active`): which unlabeled requests (`--pool`) to send to the teacher.
  Normalized entropy plus how under-represented the request's cluster (cosine k-means) is in training. Never
  use it for the shadow sample: agreement needs a random sample.

  ```bash
  python -m distill.monitor.active students/ pool.jsonl logs.jsonl --budget 50 \
      --teacher mypkg.llm:ask --queue queue.jsonl   # omit --teacher/--queue to just print the ranking
  ```

- **Fallback agreement**: every fallback is teacher-labeled anyway, so the student's own guess vs the teacher's
  answer there is a second agreement stream, against its own calib baseline (`rejected_agreement`). When the
  gate rejects most traffic, it is most of the concept-drift signal.

`scripts/simulate_drift.py` runs all of it on AG News; results are in the next section.

## Results: simulated drift on AG News (`scripts/simulate_drift.py`, d1-3B, CPU bf16)

Era A (training, 800 logs): science/tech is 5% of traffic. Teacher = gold label with 10% uniform noise. Calib
picked the readout prior at l2=10: 0.917 vs human on the 120 test rows (zero-shot d1: 0.908).

**M3: served gate vs offline estimate**, at the default target 0.85; 300 fresh era-A rows through the gate
loaded from disk:

| | offline (test slice) | served (fresh rows) |
|---|---|---|
| fallback rate | 0.108 | 0.157 |
| agreement on accepted rows | 0.850 | 0.877 |
| system accuracy vs human | 0.958 | 0.953 |
| cost / 1k decisions (teacher only: $0.40) | $0.043 | $0.063 |

The fallback gap is 5 points, about 1.4 combined standard errors on these row counts.

**The target is the knob that matters.** Test-slice curve for this task:

| target agreement | fallback | system acc vs human | cost / 1k decisions |
|---|---|---|---|
| 0.80 | 0.8% | 0.917 | $0.003 |
| **0.85** (the default before `auto`) | 10.8% | **0.958** | $0.043 |
| 0.90 | 93.3% | 0.917 | $0.373 |
| 0.95 | 96.7% | 0.908 | $0.387 |

A 10%-noise teacher caps how often anyone can agree with it, so high targets reject almost everything and
hand rows to a teacher that is *less* accurate than the student. At 0.85 the gate sends only the student's
shaky rows to the teacher and beats both (student 0.917, teacher ≈ 0.90). On the house task below the best
target is 0.80, not 0.85: no fixed target is right everywhere, which is why the default is now `auto`.

**M4: drift alarms** (window 500; PSI alarm at 0.2; agreement alarms at their calib baseline − 0.1; this
scenario shadows every accepted answer):

| stream | PSI | shadow agreement | alarms |
|---|---|---|---|
| stable, fresh era-A rows | 0.018 | – | none |
| surge: science/tech 55% | **0.856** | 0.686 | PSI, shadow agreement |
| concept: teacher files business as world | 0.018 | **0.621** | shadow agreement only |

At a 0.95 target the gate accepted so little that shadow agreement had 8–12 samples and concept drift showed
up only in the fallback stream (0.569 vs a 0.798 baseline). Both streams exist for that reason.

**M4: active learning, the fair test.** The new region (science/tech, 5% of training) is a small share of a
400-row pool (15% as sampled), so random labels rarely reach it. Retrain on era A + B pool labels, score on a
balanced held-out set (300 rows, 66 science/tech) vs human; random is 10 seeds (mean, range):

| budget | | prioritize() | entropy only | random | no new labels |
|---|---|---|---|---|---|
| 30 | overall | **0.886** | 0.876 | 0.858 (0.853–0.866) | 0.850 |
| 30 | science/tech | **0.681** | 0.636 | 0.553 (0.530–0.590) | 0.515 |
| 60 | overall | **0.890** | 0.886 | 0.868 (0.860–0.876) | 0.850 |
| 60 | science/tech | **0.696** | 0.681 | 0.595 (0.560–0.636) | 0.515 |

`prioritize()` beats every random seed at both budgets, and the cluster term adds to entropy alone. When the
new region is already the majority of the pool (the surge pool, 56% science/tech), random finds it too and
there is no difference (60 labels: 0.833 vs 0.840). One pool draw and 66 science/tech rows: a clear direction,
not a precise size.

## Results: a task d1 can't know (`scripts/simulate_logs.py --policy house`)

Opaque options ("Queue 1".."Queue 3") and a hidden routing rule over AG News: world → q1, sports → q3,
science/tech → q2, business → q2 if the article contains a digit, else q1. 800 logs, teacher with 10% noise,
120 time-split test rows:

| | calib NLL | test acc vs human | test ECE after scaling |
|---|---|---|---|
| d1 zero-shot | 1.096 | 0.258 | 0.089 |
| linear head, frozen d1 features (zero init, l2=1) | **0.673** | 0.883 | 0.113 |
| LoRA, epoch 2 (chosen; epoch 1: 0.754) | 0.692 | 0.883 | 0.110 |
| the teacher, for reference | – | 0.933 | – |

**This is the case the project exists for:** zero-shot d1 is below chance, and distilling 560 teacher
decisions takes it to 0.883. Gated at target 0.80, the head falls back on 26% of traffic and the system
reaches **0.942** vs human, above the teacher, at $0.103 per 1k against the teacher's $0.40. The default 0.85
would have sent 88% back: pick the target from the curve. Temperature scaling made test ECE worse on this
task (0.064 → 0.113 for the head): T is fitted on 120 calibration rows.

**LoRA still doesn't beat the head**, even where d1 knows nothing: the same test accuracy, a slightly worse
calib NLL, at ~28 minutes of CPU against seconds. Calib NLL was still falling at epoch 2; four epochs with
the real teacher (below) peaked at epoch 3 and still tied the head on test.

**Served LoRA student (M3 for LoRA).** `lora --save` then `serve.replay` on 300 fresh house rows, target 0.85:

| | offline (test slice) | served (fresh rows) |
|---|---|---|
| fallback rate | 0.392 | 0.400 |
| system accuracy vs human | 0.925 | 0.913 |
| agreement on accepted rows | 0.904 | 0.817 |
| cost / 1k decisions | $0.157 | $0.160 |

Fallback, accuracy and cost carry over. Agreement on accepted rows drops 9 points, about 2 combined standard
errors (calib had 0.851). The served readout (primary option tokens) picks the same answer as d1's
max-pooled readout on 100% of test rows.

## Results: a real LLM teacher (`scripts/llm_teacher.py`)

The runs above use a simulated teacher with uniform 10% noise. Here the teacher is a real LLM,
Qwen2.5-3B-Instruct, run locally. It decides by option-letter logits:
`python scripts/llm_teacher.py logs.jsonl > logs_llm.jsonl` relabels any log. On the house task its own prompt
spells out the routing policy, while the logged question stays opaque ("Queue 1".."Queue 3"). Its errors are
systematic, not random:

- house: 0.805 vs human on 800 rows, 116 of its 155 errors are q2→q1 (business with digits);
- AG News: 0.795 vs human, mostly science/tech filed as business or world;
- it is about 0.97–0.99 confident when wrong, so its own confidence is no signal.

Qwen's license is fine for generating evaluation logs, not for shipping. Same splits as above (120 / 90 test
rows), d1 bf16 on CPU, gate `auto`:

| test, vs human | d1 zero-shot | head | LoRA (epoch 3 of 4) | teacher | gated system (`auto`) |
|---|---|---|---|---|---|
| house | 0.258 | 0.825 | 0.825 | 0.867 | 0.842 (head, 7.5% fallback) |
| AG News | **0.822** | 0.700 | – | 0.700 | 0.700 (4.4% fallback) |

- **The student learns the teacher, mistakes included.** On AG News the head agrees with the teacher on 94%
  of test rows and sits at exactly the teacher's 0.700 vs human, below untrained d1 (0.822). Distillation
  from a worse teacher made d1 worse, and no gate can fix it: every fallback goes to that same teacher.
  Compare zero-shot against the teacher on human labels *before* distilling.
- **`auto` behaved sensibly but has little room.** House: 7.5% fallback, 0.842 vs human; the most accurate
  point on the curve was 0.867 at 73% fallback, which the one-SE rule correctly turns down as within noise.
  On AG News, falling back can't beat the teacher, so `auto` keeps almost everything.
- **Fresh traffic:** the head student replayed on 300 new house rows gets 0.783 vs human at 4.3% fallback,
  against 0.842 offline. The teacher scored 0.82 on those rows, so the fresh slice is harder. Fallback
  agreement is 0.46: the rows the gate sends back are ones the student and the teacher disagree on.
- **LoRA, 4 epochs:** calib NLL 0.447 → 0.398 → **0.384** → 0.492 (epoch 4 overfits; epoch 3 kept). That
  is its first calib win over the head (0.414), but test accuracy ties at 0.825, and `auto` gave LoRA 0%
  fallback. About 65 minutes of CPU against 6 for the head.

With 120 test rows one standard error is about 0.035: every head-to-head gap here except zero-shot vs the
student on AG News is inside it.

## LoRA fine-tune (M2)

```bash
pip install -e ".[d1,lora]"
python -m distill.train.lora logs.jsonl --dtype bfloat16 --epochs 2 [--task NAME] > lora_report.json
```

LoRA (r=8 on the attention and short-conv projections; adapters kept in fp32 on the bf16 base, since bf16's
~3 significant digits would swallow Adam's small updates) trains d1's own decision: cross-entropy over the
max-pooled option logits at the answer slot, as `system_one` reads them. The epoch is chosen on calib NLL
after temperature scaling, with untrained d1 (epoch 0) as a candidate, so harmful fine-tuning is not kept.
The report puts the frozen-feature linear head (the full baseline report) next to LoRA on the same split.

**Result, same 600 AG News logs and 90 test rows as above, CPU bf16 (about 2.5–5 s a row, 25 min an epoch):**

| | calib NLL (selection metric) | test acc vs human | test ECE after scaling |
|---|---|---|---|
| d1 zero-shot (epoch 0) | 0.608 | 0.822 | 0.086 |
| linear head, frozen features | **0.480** | 0.867 | **0.039** |
| LoRA, epoch 1 (epoch 2 overfit: 0.547) | 0.522 | 0.900 | 0.084 |

On AG News too, calibration prefers the head; LoRA's test lead is 3 rows of 90, and it calibrates worse after
scaling (SPEC §10's risk, measured). On both tasks the cheap head is the better default; LoRA is there to try
when a head plateaus, and the report shows whether it helped.

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
