#!/usr/bin/env python3
# generators/math_long_cot_v2.py
"""
Math Long CoT v2 (short chains)
- Produces paired correct / wrong samples
- Chains of reasoning length configurable 2-4 steps (short CoT)
- Error modes: off-by-one, wrong intermediate, sign flip, carry miss, drift
- Designed for compact CoT training and evaluation (fits 256-512 token budgets)
"""

import random
import json
from typing import List, Dict, Tuple

SEED = 2026
random.seed(SEED)

def _rand(a: int, b: int) -> int:
    return random.randint(a, b)

def _off_by_one(x: int) -> int:
    return x + random.choice([-1, 1])

def _make_step_text(step_idx: int, desc: str, value: str) -> str:
    return f"Step {step_idx}: {desc} → {value}"

def _generate_base_problem():
    # choose problem family: addition chain, multi-term, algebra simple
    family = random.choice(["add_chain", "multi_term", "algebra_chain"])
    if family == "add_chain":
        a = _rand(10, 999)
        b = _rand(10, 999)
        return "add_chain", {"a": a, "b": b}
    if family == "multi_term":
        terms = [_rand(5, 199) for _ in range(random.randint(3, 5))]
        ops = [random.choice(["+", "-"]) for _ in range(len(terms)-1)]
        return "multi_term", {"terms": terms, "ops": ops}
    # algebra_chain
    x = _rand(2, 9)
    a = _rand(2, 9)
    c = _rand(1, 9)
    b_total = a * x + c
    return "algebra_chain", {"a": a, "c": c, "b_total": b_total, "x": x}

def _build_correct_chain(family: str, params: dict, n_steps: int) -> Tuple[str, str]:
    steps = []
    if family == "add_chain":
        a, b = params["a"], params["b"]
        running = a
        steps.append(_make_step_text(1, f"Start with {a}", str(running)))
        # distribute b across remaining steps
        for i in range(2, n_steps):
            add = max(1, b // max(1, (n_steps - i + 1)))
            running = running + add
            steps.append(_make_step_text(i, f"Add {add}", str(running)))
        steps.append(_make_step_text(n_steps, f"Final add remainder to reach {a+b}", str(a+b)))
        question = f"Compute {a} + {b} with short step-by-step reasoning."
        answer = str(a + b)
        return question + "\n\n" + "\n".join(steps), answer

    if family == "multi_term":
        terms, ops = params["terms"], params["ops"]
        running = terms[0]
        steps.append(_make_step_text(1, f"Start with {running}", str(running)))
        for i, op in enumerate(ops, start=2):
            t = terms[i-1]
            if op == "+":
                running = running + t
                steps.append(_make_step_text(i, f"Add {t}", str(running)))
            else:
                running = running - t
                steps.append(_make_step_text(i, f"Subtract {t}", str(running)))
        question = "Evaluate step-by-step: " + " ".join(
            f"{terms[i]} {ops[i]}" if i < len(ops) else str(terms[-1]) for i in range(len(terms)-1)
        )
        answer = str(running)
        return question + "\n\n" + "\n".join(steps), answer

    # algebra_chain
    a, c, b_total, x = params["a"], params["c"], params["b_total"], params["x"]
    steps.append(_make_step_text(1, f"Equation: {a}x + {c} = {b_total}", "Isolate term"))
    if n_steps >= 2:
        steps.append(_make_step_text(2, f"Subtract {c} from both sides", str(b_total - c)))
    if n_steps >= 3:
        steps.append(_make_step_text(3, f"Divide by {a}", str((b_total - c) // a)))
    steps.append(_make_step_text(n_steps, f"Solve for x", str(x)))
    question = f"Solve for x given {a}x + {c} = {b_total} with short steps."
    answer = str(x)
    return question + "\n\n" + "\n".join(steps), answer

def _build_wrong_chain(family: str, params: dict, n_steps: int) -> Tuple[str, str]:
    correct_text, correct_ans = _build_correct_chain(family, params, n_steps)
    lines = correct_text.splitlines()
    # pick an internal step to corrupt (prefer middle steps)
    candidate_idxs = [i for i in range(1, len(lines)-1) if lines[i].startswith("Step")]
    if not candidate_idxs:
        wrong_ans = str(_off_by_one(int(correct_ans)))
        wrong_text = correct_text + "\n\nNote: final step altered."
        return wrong_text, wrong_ans
    corrupt_idx = random.choice(candidate_idxs)
    parts = lines[corrupt_idx].split("→")
    if len(parts) >= 2:
        left, right = parts[0], parts[1]
        try:
            val = int(right.strip())
            mode = random.choice(["off_by_one", "wrong_sign", "carry_miss", "drift"])
            if mode == "off_by_one":
                new_val = val + random.choice([-1, 1])
            elif mode == "wrong_sign":
                new_val = -val
            elif mode == "carry_miss":
                new_val = val - random.randint(1, max(1, abs(val)//10))
            else:
                new_val = val + random.choice([-2, 2])
            lines[corrupt_idx] = left + "→ " + str(new_val)
            wrong_ans = str(_off_by_one(int(correct_ans)))
            wrong_text = "\n".join(lines)
            return wrong_text, wrong_ans
        except Exception:
            wrong_ans = str(_off_by_one(int(correct_ans)))
            wrong_text = correct_text + "\n\nNote: intermediate step altered textually."
            return wrong_text, wrong_ans
    wrong_ans = str(_off_by_one(int(correct_ans)))
    wrong_text = correct_text + "\n\nNote: final changed."
    return wrong_text, wrong_ans

def generate_math_long_cot_v2_short(n_pairs: int = 200, min_steps: int = 2, max_steps: int = 4) -> List[Dict]:
    """
    Generate paired short CoT samples (2-4 steps).
    Each pair: one correct (label 0), one wrong (label 1).
    """
    samples = []
    for _ in range(n_pairs):
        family, params = _generate_base_problem()
        n_steps = random.randint(min_steps, max_steps)
        q_correct, a_correct = _build_correct_chain(family, params, n_steps)
        q_wrong, a_wrong = _build_wrong_chain(family, params, n_steps)
        samples.append({
            "index": len(samples) + 1,
            "category": "Math-Long-CoT-v2-short",
            "question": q_correct,
            "answer": a_correct,
            "true_label": 0
        })
        samples.append({
            "index": len(samples) + 1,
            "category": "Math-Long-CoT-v2-short",
            "question": q_wrong,
            "answer": a_wrong,
            "true_label": 1
        })
    return samples

def save_math_long_cot_v2_short_jsonl(path: str, n_pairs: int = 200, min_steps: int = 2, max_steps: int = 4):
    samples = generate_math_long_cot_v2_short(n_pairs=n_pairs, min_steps=min_steps, max_steps=max_steps)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    save_math_long_cot_v2_short_jsonl("math_long_cot_v2_short_demo.jsonl", n_pairs=20, min_steps=2, max_steps=4)
    print("Wrote demo math_long_cot_v2_short_demo.jsonl")
