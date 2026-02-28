#!/usr/bin/env python3
# generators/math_adversarial_v2.py
"""
Math Adversarial v2
- Focus: trick questions, ambiguous phrasing, context traps
- Produces paired correct / wrong samples suitable for hallucination detection
- Error families: ambiguous wording, unit confusion, implicit assumptions,
  distractor numbers, phrasing that invites common heuristics, multi-interpretation
- API:
    generate_math_adversarial_v2(n_pairs=300) -> List[Dict]
    save_math_adversarial_v2_jsonl(path, n_pairs=300)
"""

import random
import json
from typing import List, Dict, Tuple

SEED = 777
random.seed(SEED)

# -----------------------
# Utilities
# -----------------------
def _rand(a: int, b: int) -> int:
    return random.randint(a, b)

def _off_by_one(x: int) -> int:
    return x + random.choice([-1, 1])

def _fmt_frac(n: int, d: int) -> str:
    return f"{n}/{d}"

# -----------------------
# Adversarial templates
# -----------------------

def ambiguous_order(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """
    Ambiguous order: "If you add A to B, what is the result?" vs ambiguous phrasing
    Wrong variant: misinterpret order or swap operands when phrasing invites it.
    """
    q = f"If you add {a} to {b}, what is the result?"
    true = str(a + b)
    if correct:
        return q, true
    # wrong: swap or subtract
    wrong = str(abs(b - a)) if random.random() < 0.5 else str(b + a + random.choice([-1, 1]))
    return q, wrong

def unit_confusion(value: int, unit: str, correct: bool = True) -> Tuple[str, str]:
    """
    Unit confusion: mixing units or omitting unit conversion.
    Example: "If a rope is 120 cm and cut into 3 equal pieces, how long is each piece?"
    Wrong variant: answer in meters without conversion or wrong denominator.
    """
    q = f"A rope is {value} {unit}. If cut into 3 equal pieces, how long is each piece?"
    if unit in ("cm", "mm", "m"):
        # compute in same unit
        true = f"{value/3:.3f} {unit}" if unit != "mm" else f"{value//3} {unit}"
    else:
        true = f"{value/3}"
    if correct:
        return q, true
    # wrong: convert incorrectly or give integer division
    if unit == "cm":
        wrong = f"{(value/100)/3:.3f} m"  # convert to meters then divide (incorrect if expected cm)
    else:
        wrong = str(value // 3)
    return q, wrong

def implicit_assumption(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """
    Implicit assumption: phrasing invites an assumption (e.g., 'between' inclusive/exclusive).
    Example: "How many integers are between 1 and 5?"
    Correct: 3 (2,3,4) if exclusive; wrong variant: 5 if inclusive or off-by-one.
    """
    q = f"How many integers are between {a} and {b}?"
    true = str(max(0, abs(b - a) - 1))
    if correct:
        return q, true
    # wrong: inclusive count or off-by-one
    wrong = str(abs(b - a) + 1)
    return q, wrong

def distractor_numbers(a: int, b: int, c: int, correct: bool = True) -> Tuple[str, str]:
    """
    Distractor numbers: include extra numbers in text that tempt heuristic answers.
    Example: "A has A apples, B gives B apples, C steals C apples. How many does A have now?"
    Wrong variant: sum all numbers or pick wrong actor.
    """
    q = f"A has {a} apples. B gives A {b} apples. C takes {c} apples from B. How many apples does A have now?"
    true = str(a + b)
    if correct:
        return q, true
    # wrong: sum all numbers or use wrong subtraction
    wrong = str(a + b - c) if random.random() < 0.5 else str(a + b + c)
    return q, wrong

def ambiguous_fraction_text(n: int, d: int, correct: bool = True) -> Tuple[str, str]:
    """
    Ambiguous fraction phrasing: "half of 1/3" vs "1/2 of 1/3" confusion.
    """
    q = f"What is half of {n}/{d}?"
    true = f"{n}/{d*2}"
    if correct:
        return q, true
    # wrong: interpret as 1/(2*{d}) or invert
    wrong = f"{n//2}/{d}" if n % 2 == 0 else f"{n}/{d//2 if d>1 else d}"
    return q, wrong

def ambiguous_wording(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """
    Ambiguous wording that invites common heuristics:
    "If you multiply A by B and then subtract A, what's the result?" vs mis-ordering.
    """
    q = f"If you multiply {a} by {b} and then subtract {a}, what's the result?"
    true = str(a * b - a)
    if correct:
        return q, true
    # wrong: subtract first then multiply or forget subtraction
    wrong = str((a - a) * b) if random.random() < 0.4 else str(a * b)
    return q, wrong

def ambiguous_percent(a: int, percent: int, correct: bool = True) -> Tuple[str, str]:
    """
    Percent phrasing: "What is X percent of Y?" vs "Y percent of X" confusion.
    """
    q = f"What is {percent}% of {a}?"
    true = f"{a * percent / 100:.3f}"
    if correct:
        return q, true
    wrong = f"{percent * a:.3f}"  # misinterpret as percent * number
    return q, wrong

def trick_sequence(a: int, correct: bool = True) -> Tuple[str, str]:
    """
    Sequence trick: ask about pattern where naive continuation is wrong.
    Example: "Sequence: 2, 4, 8, 16, what's next?" (naive: 32) vs trick: pattern resets.
    """
    seq_type = random.choice(["powers2", "alternating", "add_index"])
    if seq_type == "powers2":
        q = "Sequence: 2, 4, 8, 16. What is the next number?"
        true = "32"
        if correct:
            return q, true
        wrong = "16"  # repeat or trick
        return q, wrong
    if seq_type == "alternating":
        q = "Sequence: 3, 5, 7, 9, 11. What is the next number?"
        true = "13"
        if correct:
            return q, true
        wrong = "12"
        return q, wrong
    q = "Sequence: 1, 2, 4, 7, 11. What is the next number?"
    true = "16"  # differences 1,2,3,4
    if correct:
        return q, true
    wrong = "12"
    return q, wrong

# -----------------------
# Main generator
# -----------------------

ADVERSARIAL_FUNCS = [
    ambiguous_order,
    unit_confusion,
    implicit_assumption,
    distractor_numbers,
    ambiguous_fraction_text,
    ambiguous_wording,
    ambiguous_percent,
    trick_sequence
]

def generate_math_adversarial_v2(n_pairs: int = 300) -> List[Dict]:
    """
    Generate n_pairs * 2 samples:
      - For each pair: one correct (label 0), one adversarial/wrong (label 1)
      - Returns list of dicts with keys: index, category, question, answer, true_label
    """
    samples: List[Dict] = []
    for i in range(n_pairs):
        func = random.choice(ADVERSARIAL_FUNCS)
        # choose parameters based on function signature
        if func is ambiguous_order:
            a, b = _rand(1, 200), _rand(1, 200)
            q1, a1 = func(a, b, True)
            q2, a2 = func(a, b, False)
        elif func is unit_confusion:
            val = _rand(30, 300)
            unit = random.choice(["cm", "m", "mm"])
            q1, a1 = func(val, unit, True)
            q2, a2 = func(val, unit, False)
        elif func is implicit_assumption:
            a, b = sorted([_rand(1, 20), _rand(1, 20)])
            q1, a1 = func(a, b, True)
            q2, a2 = func(a, b, False)
        elif func is distractor_numbers:
            a, b, c = _rand(1, 10), _rand(1, 10), _rand(1, 10)
            q1, a1 = func(a, b, c, True)
            q2, a2 = func(a, b, c, False)
        elif func is ambiguous_fraction_text:
            n, d = _rand(1, 9), _rand(2, 12)
            q1, a1 = func(n, d, True)
            q2, a2 = func(n, d, False)
        elif func is ambiguous_wording:
            a, b = _rand(2, 20), _rand(2, 20)
            q1, a1 = func(a, b, True)
            q2, a2 = func(a, b, False)
        elif func is ambiguous_percent:
            a, p = _rand(10, 500), random.choice([5, 10, 12, 20, 25])
            q1, a1 = func(a, p, True)
            q2, a2 = func(a, p, False)
        else:  # trick_sequence
            q1, a1 = func(_rand(1, 10), True)
            q2, a2 = func(_rand(1, 10), False)

        idx_base = len(samples) + 1
        samples.append({
            "index": idx_base,
            "category": "Math-Adversarial-v2",
            "question": q1,
            "answer": a1,
            "true_label": 0
        })
        samples.append({
            "index": idx_base + 1,
            "category": "Math-Adversarial-v2",
            "question": q2,
            "answer": a2,
            "true_label": 1
        })
    return samples

def save_math_adversarial_v2_jsonl(path: str, n_pairs: int = 300):
    samples = generate_math_adversarial_v2(n_pairs=n_pairs)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    out = "math_adversarial_v2_demo.jsonl"
    save_math_adversarial_v2_jsonl(out, n_pairs=50)
    print(f"Wrote demo adversarial file to {out}")
