"""M4: which unlabeled requests to send to the teacher for labels.

    python -m distill.monitor.active students/ pool.jsonl logs.jsonl --budget 50 > to_label.jsonl
    python -m distill.monitor.active students/ pool.jsonl logs.jsonl --budget 50 \\
        --teacher mypkg.llm:ask --queue queue.jsonl          # label them now, as log rows for retraining

`pool.jsonl` is what `serve.app --pool` writes (student-answered requests with the student's probabilities);
`logs.jsonl` are the logs the students were trained on, for "under-represented in training".

Score = normalized entropy of the student's answer (it is unsure) + `weight` × under-representation of the
request's cluster in training (it is somewhere the student has barely been). Entropy alone keeps picking the
same ambiguous boundary cases; the cluster term pulls labels toward regions that drift has opened up.

Not for the shadow sample: agreement monitoring needs a random sample, and this one is biased on purpose.
"""

from __future__ import annotations

import numpy as np


def entropy(P: np.ndarray) -> np.ndarray:
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(axis=1)


def kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Lloyd's algorithm; returns centroids. Empty clusters keep their previous centroid."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=min(k, len(X)), replace=False)].copy()
    for _ in range(iters):
        assign = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for j in range(len(C)):
            if (assign == j).any():
                C[j] = X[assign == j].mean(0)
    return C


def prioritize(P_pool: np.ndarray, X_pool: np.ndarray, X_train: np.ndarray, budget: int,
               k: int = 8, weight: float = 1.0, seed: int = 0) -> np.ndarray:
    """Indices into the pool, best first, at most `budget` of them."""
    X = np.vstack([X_pool, X_train]).astype(np.float64)
    # Cosine k-means (centre, then unit rows). Per-dimension z-scores were worse: on sparse features a token
    # seen once gets a huge z and clusters form around noise.
    Z = X - X.mean(0)
    Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
    C = kmeans(Z, k, seed=seed)
    assign = ((Z[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
    a_pool, a_train = assign[: len(X_pool)], assign[len(X_pool):]
    pool_share = np.bincount(a_pool, minlength=len(C)) / len(a_pool)
    train_share = np.bincount(a_train, minlength=len(C)) / max(len(a_train), 1)
    under = pool_share / np.maximum(pool_share + train_share, 1e-12)  # 1: absent from training, 0.5: balanced
    score = entropy(P_pool) / np.log(P_pool.shape[1]) + weight * under[a_pool]
    return np.argsort(-score, kind="stable")[:budget]


def select(students, extract, pool_rows: list[dict], train_records, budget: int) -> list[dict]:
    """Per task, the `budget` pool requests most worth a teacher label. Repeated states count once."""
    import json

    picked = []
    for task, s in students.items():
        uniq = {json.dumps(r["state"], sort_keys=True): r for r in pool_rows if r.get("task") == task}
        rows = list(uniq.values())
        train = [d.state for d in train_records if d.task == task]
        if not rows or not train:
            continue
        P = np.array([[r["probabilities"][c] for c in s.classes] for r in rows])
        order = prioritize(P, extract([r["state"] for r in rows], s.question), extract(train, s.question), budget)
        picked += [{"task": task, "state": rows[i]["state"], "rank": n} for n, i in enumerate(order)]
    return picked


def main(argv=None) -> None:
    import argparse
    import json
    import sys

    from distill.logs.ingest import read_many
    from distill.serve.gate import Gateway, load_teacher
    from distill.serve.student import extractor_for, load_students

    p = argparse.ArgumentParser(description="Rank pooled requests for teacher labels")
    p.add_argument("students")
    p.add_argument("pool")
    p.add_argument("logs", nargs="+", help="the logs the students were trained on")
    p.add_argument("--budget", type=int, default=50, help="labels per task")
    p.add_argument("--teacher", help="package.module:function(task, state, question) -> label; asks it now")
    p.add_argument("--queue", help="with --teacher: append the labels here, as log rows")
    p.add_argument("--device")
    args = p.parse_args(argv)
    if args.budget < 1:
        p.error("--budget must be ≥ 1")
    if bool(args.teacher) != bool(args.queue):
        p.error("--teacher and --queue go together")
    students = load_students(args.students)
    extract = extractor_for(students, args.device)
    pool = []
    with open(args.pool, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    pool.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"{args.pool} line {n}: skipped, {e}", file=sys.stderr)
    records, _ = read_many(args.logs)
    picked = select(students, extract, pool, records, args.budget)
    if args.teacher:
        gw = Gateway(students, extract, load_teacher(args.teacher), queue_path=args.queue)
        for r in picked:
            r["label"] = gw._ask(students[r["task"]], r["state"], "active")  # validated and queued as a log row
    for r in picked:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
