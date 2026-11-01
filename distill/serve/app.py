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
from typing import Any

from html import escape

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from distill.serve.gate import Gateway, load_teacher

COLUMNS = [  # (status key, header, format)
    ("requests", "Requests", "{:,}"), ("fallback_rate", "Fallback", "{:.1%}"),
    ("teacher_calls", "Teacher calls", "{:,}"), ("cost_per_1k_teacher_only", "$ / 1k, teacher only", "${:.4f}"),
    ("cost_per_1k_observed", "$ / 1k, gated", "${:.4f}"), ("savings", "Saved", "{:.1%}"),
    ("psi", "PSI", "{:.3f}"), ("agreement", "Shadow agreement", "{:.3f}"),
    ("fallback_agreement", "Fallback agreement", "{:.3f}"),
]
ALARMS = ("psi_alarm", "agreement_alarm", "fallback_agreement_alarm")


def render_dashboard(status: dict) -> str:
    """Cost and drift per task (SPEC §4 v1 cost dashboard). Task names come from logs: everything is escaped."""
    rows = []
    for task, s in sorted(status.items()):
        s = {**s, "fallback_rate": s["fallbacks"] / s["requests"] if s["requests"] else None,
             "savings": None if not s.get("cost_per_1k_teacher_only") or s.get("cost_per_1k_observed") is None
             else 1 - s["cost_per_1k_observed"] / s["cost_per_1k_teacher_only"]}
        cells = "".join(f"<td>{'–' if s.get(k) is None else escape(fmt.format(s[k]))}</td>" for k, _, fmt in COLUMNS)
        alarms = [a.removesuffix("_alarm").replace("_", " ") for a in ALARMS if s.get(a)]
        badge = (f'<td class="alarm">⚠ {escape(", ".join(alarms))}</td>' if alarms else '<td class="ok">ok</td>')
        rows.append(f'<tr><th scope="row">{escape(task)}</th>{cells}{badge}</tr>')
    head = "".join(f'<th scope="col">{escape(h)}</th>' for _, h, _ in COLUMNS)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="10">
<title>decide-distill dashboard</title><style>
:root {{ color-scheme: light dark; --alarm: #b3261e; --ok: #1b6e3a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --alarm: #ff8a80; --ok: #7bd88f; }} }}
body {{ font: 14px/1.4 system-ui, sans-serif; margin: 16px; }} table {{ border-collapse: collapse; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #8884; text-align: right; }}
th[scope=row], caption {{ text-align: left; }} .alarm {{ color: var(--alarm); font-weight: 600; }}
.ok {{ color: var(--ok); }} .wrap {{ overflow-x: auto; }}
</style></head><body><div class="wrap"><table>
<caption>Per task, since the server started. Cost counts every teacher call (fallbacks and shadow checks) at the
training logs' mean cost; the local student isn't counted. Refreshes every 10 s.</caption>
<thead><tr><th scope="col">Task</th>{head}<th scope="col">Alarms</th></tr></thead>
<tbody>{"".join(rows) or '<tr><td colspan="99">no students loaded</td></tr>'}</tbody></table></div></body></html>"""


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

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> str:
        return render_dashboard(gateway.status())

    return app


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
