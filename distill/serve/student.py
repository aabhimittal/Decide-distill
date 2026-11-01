"""A trained student on disk: linear head + temperature + gate threshold + drift reference, per task."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from distill.calibrate.temperature import softmax
from distill.train.linear_head import LinearHead


@dataclass
class Student:
    task: str
    question: dict
    classes: list[str]
    head: LinearHead
    temperature: float
    tau: float                      # answer when calibrated confidence ≥ tau, else ask the teacher
    reference_counts: list[int]     # predicted-class counts on pre-deployment traffic, for PSI
    expected_agreement: float | None  # offline teacher agreement on the rows the gate accepts
    meta: dict = field(default_factory=dict)  # features, model id, l2, init: how to rebuild the extractor
    expected_fallback_agreement: float | None = None  # offline: student's guess vs teacher on rejected rows
    directory: Path | None = field(default=None, compare=False)  # set on load: where named files (an adapter) live

    def probs(self, X: np.ndarray) -> np.ndarray:
        return softmax(self.head.logits(X) / self.temperature)

    def save(self, directory: str | Path) -> Path:
        stem = student_stem(directory, self.task)
        stem.parent.mkdir(parents=True, exist_ok=True)
        np.savez(stem.with_suffix(".npz"), mean=self.head.mean, std=self.head.std, W=self.head.W)
        stem.with_suffix(".json").write_text(json.dumps({
            "task": self.task, "question": self.question, "classes": self.classes,
            "temperature": self.temperature, "tau": self.tau, "reference_counts": self.reference_counts,
            "expected_agreement": self.expected_agreement, "meta": self.meta,
            "expected_fallback_agreement": self.expected_fallback_agreement}, indent=2))
        return stem.with_suffix(".json")

    @classmethod
    def load(cls, path: str | Path) -> "Student":
        meta = json.loads(Path(path).read_text())
        arrays = np.load(Path(path).with_suffix(".npz"))
        head = LinearHead(l2=meta["meta"].get("l2", 0.0))
        head.mean, head.std, head.W = arrays["mean"], arrays["std"], arrays["W"]
        return cls(meta["task"], meta["question"], meta["classes"], head, meta["temperature"], meta["tau"],
                   meta["reference_counts"], meta["expected_agreement"], meta["meta"],
                   meta.get("expected_fallback_agreement"), Path(path).parent)


def student_stem(directory: str | Path, task: str) -> Path:
    """Where a task's student files go. Task names come from logs, so they are hashed, never used as paths."""
    return Path(directory) / hashlib.sha1(task.encode()).hexdigest()[:16]


# What decides the features a student was trained on; anything else in meta (l2, costs) may differ per task.
FEATURE_KEYS = ("features", "dim", "model_id", "dtype", "adapter", "lora_config")


def load_students(directory: str | Path) -> dict[str, Student]:
    students = {s.task: s for s in (Student.load(p) for p in sorted(Path(directory).glob("*.json")))}
    if not students:
        raise FileNotFoundError(f"no students in {directory}")
    return students


def extractor_for(students: dict[str, Student], device: str | None = None):
    """The feature extractor the students were trained on; one extractor serves all of them."""
    kinds = {json.dumps({k: v for k, v in s.meta.items() if k in FEATURE_KEYS}, sort_keys=True)
             for s in students.values()}
    if len(kinds) != 1:
        # shortcut: one adapter per process, so two LoRA students can't share a server; switch adapters per
        # request (peft's set_adapter) when serving several fine-tuned tasks from one machine matters.
        raise ValueError(f"students were trained on different features: {sorted(kinds)}")
    meta = json.loads(kinds.pop())
    if meta.get("features") == "d1":
        import torch

        from distill.train.features import D1Features

        name = meta.get("adapter")
        if name and Path(name).name != name:  # student files are input: an adapter name must not leave the dir
            raise ValueError(f"adapter must be a file name next to the student, got {name!r}")
        f = D1Features(meta["model_id"], device, getattr(torch, meta["dtype"]))
        if name:
            from distill.train.lora import load_adapter

            load_adapter(f, meta["lora_config"], next(iter(students.values())).directory / name)
        return f
    if meta.get("features") == "hashing":
        from distill.train.features import HashingFeatures

        return HashingFeatures(meta["dim"])
    raise ValueError(f"unknown features {meta!r}")
