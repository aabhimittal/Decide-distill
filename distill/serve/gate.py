"""M3: the confidence gate and the serving loop.

    student ── conf ≥ τ ──▶ answer
       └─ else ─▶ teacher ─▶ answer + appended to the training queue

Queue rows are decision-log rows (distill.logs.ingest), so the next training run ingests them as they are.
"""

from __future__ import annotations

import json
import math
import random
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from distill.monitor.drift import DriftMonitor
from distill.serve.student import Student

# teacher(task, state, question) -> label (an option name of `question`)
Teacher = Callable[[str, Any, dict], str]


def choose_tau(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """The threshold that accepts the most rows while the accepted rows agree with the teacher at ≥ `target`.
    Accuracy over the top-k most confident rows isn't monotone in k, so this scans every k. `inf` (always
    fall back) when no k reaches the target."""
    order = np.argsort(-conf, kind="stable")
    running = np.cumsum(correct[order]) / np.arange(1, len(conf) + 1)
    ok = np.nonzero(running >= target)[0]
    return float(conf[order][ok[-1]]) if len(ok) else math.inf


def gate_metrics(conf: np.ndarray, pred: np.ndarray, teacher: np.ndarray, tau: float,
                 human: np.ndarray | None = None, teacher_cost: float | None = None) -> dict:
    """What the gated system does on these rows. `human` uses -1 for a missing label."""
    accept = conf >= tau
    answer = np.where(accept, pred, teacher)  # a fallback answers with the teacher's label
    out = {
        "tau": tau,
        "fallback_rate": float(1 - accept.mean()),
        "accepted_agreement": float((pred[accept] == teacher[accept]).mean()) if accept.any() else None,
        "system_agreement": float((answer == teacher).mean()),
    }
    if human is not None and (human >= 0).any():
        m = human >= 0
        out["system_accuracy_vs_human"] = float((answer[m] == human[m]).mean())
    if teacher_cost is not None:
        out["cost_per_1k_teacher_only"] = 1000 * teacher_cost
        out["cost_per_1k_gated"] = 1000 * teacher_cost * out["fallback_rate"]  # local student cost not counted
    return out


class Gateway:
    def __init__(self, students: dict[str, Student], extract, teacher: Teacher,
                 queue_path: str | Path | None = None, pool_path: str | Path | None = None,
                 shadow_rate: float = 0.0, window: int = 500, agreement_margin: float = 0.1, seed: int = 0):
        """`queue_path`: teacher-labeled rows (fallbacks and shadow checks), in log format, for retraining.
        `pool_path`: student-answered requests with the student's probabilities, for active learning.
        `shadow_rate`: fraction of student answers also sent to the teacher, at random, to measure agreement."""
        if not 0 <= shadow_rate <= 1:
            raise ValueError(f"shadow_rate must be in [0, 1], got {shadow_rate}")
        self.students, self.extract, self.teacher = students, extract, teacher
        self.queue_path, self.pool_path = queue_path, pool_path
        self.shadow_rate, self.rng = shadow_rate, random.Random(seed)
        self.monitors = {
            t: DriftMonitor(s.reference_counts, window,
                            agreement_floor=None if s.expected_agreement is None
                            else s.expected_agreement - agreement_margin)
            for t, s in students.items()}
        self.counts = {t: {"requests": 0, "fallbacks": 0} for t in students}
        # shortcut: one lock serializes requests (file appends, monitors, the model); use a queue/worker for throughput.
        self.lock = threading.Lock()

    def decide(self, task: str, state: Any) -> dict:
        """Raises KeyError for an unknown task, ValueError if the teacher answers outside the options."""
        s = self.students[task]
        with self.lock:
            p = s.probs(self.extract([state], s.question))[0]
            k = int(p.argmax())
            self.monitors[task].observe_prediction(k)
            self.counts[task]["requests"] += 1
            out = {"task": task, "answer": s.classes[k], "student_answer": s.classes[k],
                   "confidence": float(p[k]), "source": "student"}
            if p[k] < s.tau:
                self.counts[task]["fallbacks"] += 1
                out.update(answer=self._ask(s, state, "fallback"), source="teacher")
                return out
            if self.pool_path:
                self._append(self.pool_path, {"task": task, "state": state,
                                              "probabilities": dict(zip(s.classes, p.tolist()))})
            if self.rng.random() < self.shadow_rate:
                self.monitors[task].observe_shadow(self._ask(s, state, "shadow") == s.classes[k])
            return out

    def status(self) -> dict:
        return {t: {**self.counts[t], **self.monitors[t].status()} for t in self.students}

    def _ask(self, s: Student, state: Any, reason: str) -> str:
        label = str(self.teacher(s.task, state, s.question))
        if label not in s.classes:  # never write a label the dataset builder would reject
            raise ValueError(f"teacher answered {label!r}, not one of {s.classes}")
        if self.queue_path:
            self._append(self.queue_path, {
                "id": f"q-{uuid.uuid4()}", "ts": datetime.now(timezone.utc).isoformat(), "task": s.task,
                "state": state, "question": s.question, "label": label, "reason": reason})
        return label

    @staticmethod
    def _append(path, row: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
