"""HTTP serving for the gate.

    pip install -e ".[serve]"
    python -m distill.serve.app --students students/ --teacher mypkg.teacher:ask \\
        --queue queue.jsonl --pool pool.jsonl --shadow 0.05

    POST /decide   {"task": "ticket_team", "state": "I was charged twice"}
                -> {"answer": "billing", "source": "student" | "teacher", "confidence": 0.97, ...}
    GET  /monitor  per-task request/fallback counts, PSI, shadow agreement, alarms

`--teacher` names a function `ask(task, state, question) -> label`: your existing LLM call.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from distill.serve.gate import Gateway


class DecideRequest(BaseModel):  # module level: FastAPI can't resolve a model defined inside a function here
    task: str
    state: Any


def create_app(gateway: Gateway) -> FastAPI:
    app = FastAPI(title="decide-distill")

    @app.post("/decide")
    def decide(req: DecideRequest) -> dict:
        if req.task not in gateway.students:
            raise HTTPException(404, f"no student for task {req.task!r}")
        try:
            return gateway.decide(req.task, req.state)
        except ValueError as e:  # the teacher answered outside the options; nothing was queued
            raise HTTPException(502, str(e)) from e

    @app.get("/monitor")
    def monitor() -> dict:
        return gateway.status()

    return app


def load_teacher(spec: str):
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"--teacher must look like package.module:function, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def main(argv=None) -> None:
    from distill.serve.student import extractor_for, load_students

    p = argparse.ArgumentParser(description="Serve students behind a confidence gate with teacher fallback")
    p.add_argument("--students", required=True)
    p.add_argument("--teacher", required=True, help="package.module:function(task, state, question) -> label")
    p.add_argument("--queue", help="append teacher-labeled rows here (log format)")
    p.add_argument("--pool", help="append student-answered requests here, for active learning")
    p.add_argument("--shadow", type=float, default=0.05, help="fraction of student answers checked by the teacher")
    p.add_argument("--device")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    import uvicorn

    students = load_students(args.students)
    gateway = Gateway(students, extractor_for(students, args.device), load_teacher(args.teacher),
                      args.queue, args.pool, args.shadow)
    uvicorn.run(create_app(gateway), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
