#!/usr/bin/env python3
# generators/math_synthetic_v2.py
"""
Math Synthetic v2 Generator
- Paired correct / wrong samples
- Focused on subtle arithmetic & CoT failures
- Designed for hallucination detection training & eval
"""

import random
from typing import List, Dict, Tuple

SEED = 42
random.seed(SEED)


# ────────────────────────────────────────────────────────────────
# CORE UTILITIES
# ────────────────────────────────────────────────────────────────

def _sample_int(low: int = 10, high: int = 999) -> int:
    return random.randint(low, high)


def _format_cot_addition(a: int, b: int, correct: bool, mode: str) -> Tuple[str, str]:
    """
    Generate a step-by-step addition CoT with either correct or subtly wrong reasoning.
    mode controls the type of error when correct=False.
    """
    true_sum = a + b

    # Decompose into tens/ones for CoT
    a_tens, a_ones = divmod(a, 10)
    b_tens, b_ones = divmod(b, 10)

    # Base correct reasoning
    steps = [
        f"{a} + {b} = ?",
        f"Step 1: Add ones: {a_ones} + {b_ones}",
        f"Step 2: Add tens and carry if needed",
    ]

    if correct:
        ones_sum = a_ones + b_ones
        carry = ones_sum // 10
        ones_digit = ones_sum % 10
        tens_sum = a_tens + b_tens + carry
        final = tens_sum * 10 + ones_digit

        steps.append(f"{a_ones} + {b_ones} = {ones_sum} → write {ones_digit}, carry {carry}")
        steps.append(f"{a_tens} + {b_tens} + {carry} = {tens_sum}")
        steps.append(f"Final: {final}")
        answer = str(final)
        return "\n".join(steps), answer

    # Wrong variants
    ones_sum = a_ones + b_ones
    carry = ones_sum // 10
    ones_digit = ones_sum % 10
    tens_sum = a_tens + b_tens + carry

    if mode == "off_by_one_final":
        final = true_sum + random.choice([-1, 1])
        steps.append(f"{a_ones} + {b_ones} = {ones_sum}")
        steps.append(f"{a_tens} + {b_tens} + {carry} = {tens_sum}")
        steps.append(f"Final: {final}")
        answer = str(final)

    elif mode == "wrong_carry":
        wrong_carry = max(0, carry - 1)
        wrong_tens_sum = a_tens + b_tens + wrong_carry
        final = wrong_tens_sum * 10 + ones_digit
        steps.append(f"{a_ones} + {b_ones} = {ones_sum} → write {ones_digit}, carry {wrong_carry}")
        steps.append(f"{a_tens} + {b_tens} + {wrong_carry} = {wrong_tens_sum}")
        steps.append(f"Final: {final}")
        answer = str(final)

    elif mode == "wrong_ones":
        wrong_ones = max(0, ones_sum - 1)
        wrong_carry = wrong_ones // 10
        wrong_ones_digit = wrong_ones % 10
        wrong_tens_sum = a_tens + b_tens + wrong_carry
        final = wrong_tens_sum * 10 + wrong_ones_digit
        steps.append(f"{a_ones} + {b_ones} = {wrong_ones}")
        steps.append(f"{a_tens} + {b_tens} + {wrong_carry} = {wrong_tens_sum}")
        steps.append(f"Final: {final}")
        answer = str(final)

    else:  # fallback: simple off-by-one
        final = true_sum + 1
        steps.append(f"{a_ones} + {b_ones} = {ones_sum}")
        steps.append(f"{a_tens} + {b_tens} + {carry} = {tens_sum}")
        steps.append(f"Final: {final}")
        answer = str(final)

    return "\n".join(steps), answer


def _format_plain_addition(a: int, b: int, correct: bool, subtle: bool) -> Tuple[str, str]:
    true_sum = a + b
    if correct:
        return f"What is {a} + {b}?", str(true_sum)

    # Wrong variants
    if subtle:
        # Off-by-one or digit swap
        candidates = [
            true_sum + 1,
            true_sum - 1,
            int(str(true_sum)[::-1]) if true_sum > 9 else true_sum + 2,
        ]
    else:
        # More obvious wrong
        candidates = [
            true_sum + random.randint(2, 10),
            true_sum - random.randint(2, 10),
        ]
    wrong = random.choice(candidates)
    return f"What is {a} + {b}?", str(wrong)


def _format_plain_subtraction(a: int, b: int, correct: bool, borrow_mode: bool) -> Tuple[str, str]:
    if a < b:
        a, b = b, a
    true_diff = a - b

    if correct:
        return f"What is {a} - {b}?", str(true_diff)

    if borrow_mode:
        # Wrong borrow: subtract ones correctly, but tens without borrow
        a_tens, a_ones = divmod(a, 10)
        b_tens, b_ones = divmod(b, 10)
        ones = a_ones - b_ones
        if ones < 0:
            ones += 10
        tens = a_tens - b_tens  # forgot to borrow
        wrong = tens * 10 + ones
    else:
        # Simple off-by-one or random nearby
        candidates = [
            true_diff + 1,
            true_diff - 1,
            true_diff + random.randint(2, 5),
        ]
        wrong = random.choice(candidates)

    return f"What is {a} - {b}?", str(wrong)


def _format_multiplication(a: int, b: int, correct: bool, subtle: bool) -> Tuple[str, str]:
    true_prod = a * b
    if correct:
        return f"What is {a} × {b}?", str(true_prod)

    if subtle:
        # Off-by-one factor or swapped digits
        candidates = [
            (a + 1) * b,
            (a - 1) * b if a > 1 else (a + 2) * b,
            int(str(true_prod)[::-1]) if true_prod > 9 else true_prod + 3,
        ]
    else:
        candidates = [
            true_prod + random.randint(3, 15),
            true_prod - random.randint(3, 15),
        ]
    wrong = random.choice(candidates)
    return f"What is {a} × {b}?", str(wrong)


# ────────────────────────────────────────────────────────────────
# PUBLIC API
# ────────────────────────────────────────────────────────────────

def generate_math_synthetic_pairs_v2(
    n_pairs: int = 200,
    include_cot: bool = True,
) -> List[Dict]:
    """
    Generate n_pairs * 2 samples:
      - For each pair: one correct, one wrong
      - label = 0 → non-hallucination (correct)
      - label = 1 → hallucination (wrong)
    """
    samples: List[Dict] = []

    for idx in range(n_pairs):
        mode = random.choice(["add_plain", "add_cot", "sub_plain", "mul_plain"])

        if mode == "add_cot" and include_cot:
            a = _sample_int(20, 199)
            b = _sample_int(20, 199)
            # Correct CoT
            q1, a1 = _format_cot_addition(a, b, correct=True, mode="")
            # Wrong CoT with subtle error
            wrong_mode = random.choice(["off_by_one_final", "wrong_carry", "wrong_ones"])
            q2, a2 = _format_cot_addition(a, b, correct=False, mode=wrong_mode)

            samples.append({
                "category": "Math-Synthetic-CoT",
                "question": q1,
                "answer": a1,
                "true_label": 0,
            })
            samples.append({
                "category": "Math-Synthetic-CoT",
                "question": q2,
                "answer": a2,
                "true_label": 1,
            })

        elif mode == "add_plain":
            a = _sample_int(10, 999)
            b = _sample_int(10, 999)
            q1, a1 = _format_plain_addition(a, b, correct=True, subtle=True)
            q2, a2 = _format_plain_addition(a, b, correct=False, subtle=True)

            samples.append({
                "category": "Math-Synthetic-Add",
                "question": q1,
                "answer": a1,
                "true_label": 0,
            })
            samples.append({
                "category": "Math-Synthetic-Add",
                "question": q2,
                "answer": a2,
                "true_label": 1,
            })

        elif mode == "sub_plain":
            a = _sample_int(20, 999)
            b = _sample_int(10, 899)
            q1, a1 = _format_plain_subtraction(a, b, correct=True, borrow_mode=False)
            q2, a2 = _format_plain_subtraction(a, b, correct=False, borrow_mode=True)

            samples.append({
                "category": "Math-Synthetic-Sub",
                "question": q1,
                "answer": a1,
                "true_label": 0,
            })
            samples.append({
                "category": "Math-Synthetic-Sub",
                "question": q2,
                "answer": a2,
                "true_label": 1,
            })

        elif mode == "mul_plain":
            a = _sample_int(3, 25)
            b = _sample_int(3, 25)
            q1, a1 = _format_multiplication(a, b, correct=True, subtle=True)
            q2, a2 = _format_multiplication(a, b, correct=False, subtle=True)

            samples.append({
                "category": "Math-Synthetic-Mul",
                "question": q1,
                "answer": a1,
                "true_label": 0,
            })
            samples.append({
                "category": "Math-Synthetic-Mul",
                "question": q2,
                "answer": a2,
                "true_label": 1,
            })

    # Add indices for reproducibility
    for i, s in enumerate(samples, start=1):
        s["index"] = i

    return samples


def save_math_synthetic_v2_jsonl(path: str, n_pairs: int = 200, include_cot: bool = True):
    """Convenience: write synthetic math v2 to a JSONL file."""
    import json
    samples = generate_math_synthetic_pairs_v2(n_pairs=n_pairs, include_cot=include_cot)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Quick smoke test
    out_path = "math_synthetic_v2_demo.jsonl"
    save_math_synthetic_v2_jsonl(out_path, n_pairs=5, include_cot=True)
    print(f"✅ Wrote demo synthetic math v2 to {out_path}")
