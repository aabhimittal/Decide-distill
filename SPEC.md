# DECIDE-DISTILL — turn your LLM decisions into a local decision model

## 1. Thesis

Many production LLM calls are really decisions (route, label, approve, score). DECIDE-DISTILL shadow-logs
those calls, fine-tunes a local decision model on them, deploys it with a confidence gate that falls back to
the LLM, and keeps it honest with active learning and drift detection.

Analogy: an apprentice learning from a master. At first the apprentice watches; then handles routine cases
while the master takes anything unusual; and the master periodically checks the apprentice's work to catch
bad habits.

## 2. Why now

- d1-3B is open-weight and can be downloaded, fine-tuned and run locally (Liquid model card).
- Decision models output calibrated typed answers, so a confidence-gated fallback is natural rather than
  bolted on.
- Teams now have months of LLM decision logs sitting in observability tools.

## 3. Prior art & differentiation

Distillation platforms (fine-tune a small generative model on a big one's outputs) target text generation.
DECIDE-DISTILL targets typed decisions, preserves calibration after fine-tuning (post-hoc temperature
scaling), and ships the operational loop: gate, fallback, active learning, drift alarms.

## 4. Scope

- MVP: log ingestion (JSONL), LoRA fine-tune of d1-3B (or a linear head on frozen features as a cheaper
  baseline), confidence-gated serving with LLM fallback.
- v1: active-learning queue, drift monitor, cost dashboard.
- Non-goals: distilling free-form generation.

## 5. Architecture

```
prod traffic ─▶ LLM (teacher) ─▶ decision + confidence ─▶ log store
                                                    │
                                    dataset builder (dedupe, split by time)
                                                    ▼
                                   fine-tune student (LoRA | linear head)
                                                    ▼
                                   temperature scaling on held-out slice
                                                    ▼
serving: student ── conf ≥ τ ──▶ answer
            └─ else ─▶ teacher ─▶ answer + appended to training queue
monitor: shadow sample k% to teacher → agreement rate; PSI on student output distribution
```

## 6. Core method

- Split by time, not randomly. Random splits leak near-duplicates and overstate accuracy.
- Active learning: prioritize teacher labels for samples with highest student uncertainty
  (entropy H(p) = −Σ p_i log p_i) and for clusters under-represented in training.
- Drift: Population Stability Index over predicted-class distribution, PSI = Σ (a_i − e_i)·ln(a_i/e_i);
  alarm at PSI > 0.2 (conventional threshold, tune per task). Plus a rolling teacher-agreement rate on the
  shadow sample.
- Teacher noise: the student can't be better than the teacher's labels on average; where human labels
  exist, report both teacher-agreement and human accuracy.

## 7. Evaluation

Use public labeled datasets with a simulated "teacher" LLM to generate logs, then compare student vs teacher
vs human labels. Metrics: cost per 1k decisions, fallback rate, accuracy vs human labels, calibration (ECE)
before/after temperature scaling.

## 8. Milestones

- M0 Log schema + ingestion + time-based split. Acceptance: deterministic splits, dedupe tests.
- M1 Linear-head baseline on frozen d1 features. Acceptance: reported accuracy and ECE.
- M2 LoRA fine-tune path. Acceptance: beats the linear head or the report says it doesn't.
- M3 Gated serving with fallback. Acceptance: fallback rate and accuracy at chosen τ match offline
  estimates within tolerance.
- M4 Active learning + drift monitor with a simulated drift scenario.

## 9. Repo layout & stack

`distill/ {logs/, dataset/, train/, calibrate/, serve/, monitor/}`; PyTorch, PEFT, transformers, FastAPI.

## 10. Risks & honest weaknesses

- Fine-tuning a decision model may degrade calibration; always re-calibrate and report.
- License terms for derivatives of d1 weights must be checked.
- Many teams' "decisions" hide multi-step reasoning a 3B model can't match; report per-task-family results,
  not one average.

## 11. Verify before building

d1 fine-tuning support and license for derivatives; GPU memory needs for LoRA on a 3B model on your
hardware.

## 12. Claude Code kickoff

"Read SPEC.md. Implement M0 and the M1 linear-head baseline with tests. Confirm the d1 feature-extraction
API from the current model card before writing the extractor."
