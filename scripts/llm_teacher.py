"""Relabel decision logs with a real LLM as the teacher, instead of simulated uniform noise.

    pip install -e ".[d1]"
    python scripts/llm_teacher.py house_logs.jsonl --model Qwen/Qwen2.5-3B-Instruct > house_llm.jsonl

Each row's `label` becomes the LLM's decision and `teacher_confidence` its probability; `human_label`, `state`,
`question`, ids and timestamps are kept. The LLM decides the way production decision calls usually do: it reads
the logits of the option letters at the answer position (no free text, so no parsing failures).

Policies: a production prompt carries the policy the decision log doesn't. For `house_routing` the teacher gets
the hidden routing rule in its system prompt; the logged question stays opaque, so the student only ever sees
states and decisions, as it would in production. Your own logs need none of this: they already have labels.
"""

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# What the teacher's own prompt says each option means. A 3B teacher can't route through opaque labels with a
# separate rule (it answered "A" for everything); option descriptions that state the policy, as production
# prompts do, work. The logged question keeps the opaque descriptions.
POLICIES = {
    "house_routing": {
        "q1": "Queue 1: world news, and business news that contains no digits (0-9)",
        "q2": "Queue 2: science and technology news, and business news that contains digits (0-9)",
        "q3": "Queue 3: sports news",
    },
}


def prompt(tok, row: dict) -> tuple[str, list[str]]:
    q = row["question"]
    names = list(q["criteria"])
    letters = [chr(ord("A") + i) for i in range(len(names))]
    described = POLICIES.get(row["task"], q["criteria"])
    options = "\n".join(f"{l}) {described[n] or n}" for l, n in zip(letters, names))
    system = "You make routing and labeling decisions. Answer with one option letter only."
    state = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    user = f"{q['instructions']}\n\nInput:\n{state}\n\nOptions:\n{options}\n\nAnswer with the letter only."
    text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return text, letters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).eval()
    rows = [json.loads(line) for line in open(args.logs, encoding="utf-8") if line.strip()]
    for n, row in enumerate(rows, 1):
        text, letters = prompt(tok, row)
        ids = [tok.encode(l, add_special_tokens=False) for l in letters]
        if any(len(i) != 1 for i in ids):
            raise ValueError(f"option letters must be single tokens for {args.model}")
        with torch.inference_mode():
            logits = model(**tok(text, return_tensors="pt")).logits[0, -1].float()
        probs = torch.softmax(logits[[i[0] for i in ids]], 0)
        k = int(probs.argmax())
        print(json.dumps({**row, "label": list(row["question"]["criteria"])[k],
                          "teacher_confidence": round(float(probs[k]), 4), "teacher_model": args.model},
                         ensure_ascii=False), flush=True)
        if n % 100 == 0:
            print(f"{n}/{len(rows)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
