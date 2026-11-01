"""M2: LoRA fine-tune of d1-3B on teacher decisions, against the frozen-feature linear head.

    pip install -e ".[d1,lora]"
    python -m distill.train.lora logs.jsonl --dtype bfloat16 --epochs 2 > lora_report.json

The objective is d1's own decision: cross-entropy over the option logits at the answer slot, max-pooled over
each option's token forms exactly as `system_one` reads them. LoRA teaches d1 to decide like the teacher; it
doesn't add a new head. The epoch is chosen on calib NLL after temperature scaling, and epoch 0 (untrained d1)
is a candidate, so fine-tuning that hurts is reported and not kept.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import numpy as np

from distill.calibrate.temperature import ece, fit_temperature, nll, softmax
from distill.dataset.build import by_task, dedupe, time_split
from distill.logs.ingest import options, read_jsonl
from distill.serve.gate import choose_tau, gate_metrics
from distill.train.baseline import GATE_CURVE, _acc, evaluate_task

TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj"]  # attention and short-conv projections


def option_logits(f, h, groups, W):
    """d1's readout from final hidden states `h` (N, D): option rows of lm_head, max over each option's forms."""
    logits = h @ W.T
    bounds = np.cumsum([0] + [len(g) for g in groups])
    return f.torch.stack([logits[:, a:b].max(1).values for a, b in zip(bounds, bounds[1:])], 1)


def fit_lora(f, train, calib, question, classes, r=8, alpha=16, lr=2e-4, epochs=2, accum=8, seed=0) -> list[dict]:
    """Inject LoRA into f's language model in place, train, keep the calib-best epoch. Returns per-epoch calib NLL."""
    from peft import LoraConfig, inject_adapter_in_model

    torch = f.torch
    torch.manual_seed(seed)
    lm = inject_adapter_in_model(LoraConfig(r=r, lora_alpha=alpha, target_modules=TARGETS), f.model.model.language_model)
    params = []
    for n, p in f.model.named_parameters():
        p.requires_grad = "lora_" in n
        if p.requires_grad:
            p.data = p.data.float()  # bf16 has ~3 significant digits: Adam's small updates would round to nothing
            params.append(p)
    q = f.as_question(question)
    groups = f.prompt.readout_ids(f.engine.tokenizer, q)
    W = f.model.lm_head.weight[[i for g in groups for i in g]].detach()
    yc = np.array([classes.index(d.label) for d in calib])
    opt = torch.optim.AdamW(params, lr=lr)
    encode = lambda d: torch.tensor([f.engine.tokenizer.encode(f.engine.render(d.state, q), add_special_tokens=False)])

    def calib_nll():
        Lc = f.zero_shot(f([d.state for d in calib], question), question)  # adapted model, same readout
        return nll(Lc, yc, fit_temperature(Lc, yc))

    history = [{"epoch": 0, "calib_nll": calib_nll()}]
    best = (history[0]["calib_nll"], {n: p.detach().clone() for n, p in f.model.named_parameters() if p.requires_grad})
    order = list(range(len(train)))
    for epoch in range(1, epochs + 1):
        random.Random(seed + epoch).shuffle(order)
        lm.train()
        total = 0.0
        for step, i in enumerate(order, 1):
            h = lm(input_ids=encode(train[i])).last_hidden_state[:, -1]
            loss = torch.nn.functional.cross_entropy(option_logits(f, h, groups, W).float(),
                                                     torch.tensor([classes.index(train[i].label)])) / accum
            loss.backward()
            total += loss.item() * accum
            if step % accum == 0 or step == len(order):
                opt.step()
                opt.zero_grad()
        lm.eval()
        history.append({"epoch": epoch, "train_loss": total / len(order), "calib_nll": calib_nll()})
        print(json.dumps(history[-1]), file=sys.stderr, flush=True)
        if history[-1]["calib_nll"] < best[0]:
            best = (history[-1]["calib_nll"], {n: p.detach().clone() for n, p in f.model.named_parameters() if p.requires_grad})
    with torch.no_grad():
        for n, p in f.model.named_parameters():
            if n in best[1]:
                p.copy_(best[1][n])
    return history


def compare_task(rows, f, **lora_kw) -> dict:
    """Linear head on frozen d1 features vs LoRA, on the same time split and the same metrics."""
    report = {"linear_head": evaluate_task(rows, f)}  # before any adapter exists: frozen features
    q, classes = rows[0].question, options(rows[0].question)
    split = time_split(rows)
    history = fit_lora(f, split.train, split.calib, q, classes, **lora_kw)
    Lc = f.zero_shot(f([d.state for d in split.calib], q), q)
    Lt = f.zero_shot(f([d.state for d in split.test], q), q)
    yc = np.array([classes.index(d.label) for d in split.calib])
    yt = np.array([classes.index(d.label) for d in split.test])
    humans = [d.human_label for d in split.test]
    hidx = np.array([classes.index(h) if h is not None else -1 for h in humans])
    T = fit_temperature(Lc, yc)
    Pc, Pt = softmax(Lc / T), softmax(Lt / T)
    report["lora"] = {
        "chosen_epoch": min(history, key=lambda e: e["calib_nll"])["epoch"], "history": history,
        "accuracy_vs_teacher": float((Lt.argmax(1) == yt).mean()), "accuracy_vs_human": _acc(Lt.argmax(1), humans, classes)[0],
        "temperature": T, "ece_before": ece(softmax(Lt), yt), "ece_after": ece(Pt, yt),
        "gate_curve": [{"target": g, **gate_metrics(Pt.max(1), Pt.argmax(1), yt,
                                                    choose_tau(Pc.max(1), Pc.argmax(1) == yc, g), hidx)}
                       for g in GATE_CURVE],
        "config": lora_kw,
    }
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("logs")
    p.add_argument("--model", default="LiquidAI/d1-3B")
    p.add_argument("--dtype", choices=["float32", "bfloat16"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--task", help="only this task (LoRA mutates the model, so one run trains one task)")
    args = p.parse_args(argv)
    from distill.train.baseline import make_extractor

    records, _ = read_jsonl(args.logs)
    tasks = by_task(dedupe(records))
    if args.task is None and len(tasks) > 1:
        p.error(f"several tasks in the logs; pick one with --task: {sorted(tasks)}")
    rows = tasks[args.task or next(iter(tasks))]
    f = make_extractor("d1", args.model, None, args.dtype)
    json.dump(compare_task(rows, f, r=args.r, lr=args.lr, epochs=args.epochs), sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
