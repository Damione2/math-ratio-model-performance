#!/usr/bin/env python3
# generators/math_synthetic_v3.py
"""
Math Synthetic v3 Generator
- Maximum diversity
- Multi-step reasoning
- Algebra, inequalities, division, multi-digit arithmetic
- CoT drift, wrong intermediate steps, subtle numeric errors
"""

import random
from typing import List, Dict

SEED = 42
random.seed(SEED)

# ────────────────────────────────────────────────────────────────
# Utility
# ────────────────────────────────────────────────────────────────

def _rand(a, b):
    return random.randint(a, b)

def _off_by_one(x):
    return x + random.choice([-1, 1])

def _digit_reverse(x):
    s = str(x)
    return int(s[::-1]) if len(s) > 1 else x + 2

# ────────────────────────────────────────────────────────────────
# 1. Multi-digit addition (deep CoT)
# ────────────────────────────────────────────────────────────────

def addition_cot(a, b, correct=True):
    true_sum = a + b
    steps = [
        f"{a} + {b} = ?",
        f"Step 1: Add ones digits.",
        f"Step 2: Add tens digits with carry.",
        f"Step 3: Combine for final result."
    ]

    if correct:
        ones = (a % 10) + (b % 10)
        carry = ones // 10
        ones_digit = ones % 10
        tens = (a // 10) + (b // 10) + carry
        final = tens * 10 + ones_digit
        steps.append(f"Ones: {a%10} + {b%10} = {ones} → write {ones_digit}, carry {carry}")
        steps.append(f"Tens: {a//10} + {b//10} + {carry} = {tens}")
        steps.append(f"Final: {final}")
        return "\n".join(steps), str(final)

    # Wrong variants
    mode = random.choice(["wrong_carry", "wrong_ones", "off_by_one_final", "cot_drift"])
    if mode == "wrong_carry":
        ones = (a % 10) + (b % 10)
        wrong_carry = max(0, (ones // 10) - 1)
        ones_digit = ones % 10
        tens = (a // 10) + (b // 10) + wrong_carry
        final = tens * 10 + ones_digit
        steps.append(f"Ones: {a%10} + {b%10} = {ones} → carry {wrong_carry}")
        steps.append(f"Tens: {a//10} + {b//10} + {wrong_carry} = {tens}")
        steps.append(f"Final: {final}")
        return "\n".join(steps), str(final)

    if mode == "wrong_ones":
        wrong_ones = max(0, ((a % 10) + (b % 10)) - 1)
        wrong_carry = wrong_ones // 10
        ones_digit = wrong_ones % 10
        tens = (a // 10) + (b // 10) + wrong_carry
        final = tens * 10 + ones_digit
        steps.append(f"Ones: {a%10} + {b%10} = {wrong_ones}")
        steps.append(f"Tens: {a//10} + {b//10} + {wrong_carry} = {tens}")
        steps.append(f"Final: {final}")
        return "\n".join(steps), str(final)

    if mode == "off_by_one_final":
        wrong = _off_by_one(true_sum)
        steps.append(f"Final: {wrong}")
        return "\n".join(steps), str(wrong)

    if mode == "cot_drift":
        wrong = true_sum + random.choice([-2, 2])
        steps.append(f"Final: {true_sum} → Actually: {wrong}")
        return "\n".join(steps), str(wrong)

# ────────────────────────────────────────────────────────────────
# 2. Subtraction with borrow chains
# ────────────────────────────────────────────────────────────────

def subtraction_borrow(a, b, correct=True):
    if a < b:
        a, b = b, a
    true_diff = a - b

    if correct:
        return f"What is {a} - {b}?", str(true_diff)

    mode = random.choice(["wrong_borrow", "off_by_one", "digit_reverse"])
    if mode == "wrong_borrow":
        a_t, a_o = divmod(a, 10)
        b_t, b_o = divmod(b, 10)
        ones = a_o - b_o
        if ones < 0:
            ones += 10
        tens = a_t - b_t  # forgot to borrow
        wrong = tens * 10 + ones
        return f"What is {a} - {b}?", str(wrong)

    if mode == "off_by_one":
        return f"What is {a} - {b}?", str(_off_by_one(true_diff))

    if mode == "digit_reverse":
        return f"What is {a} - {b}?", str(_digit_reverse(true_diff))

# ────────────────────────────────────────────────────────────────
# 3. Multiplication multi-step
# ────────────────────────────────────────────────────────────────

def multiplication_steps(a, b, correct=True):
    true_prod = a * b
    if correct:
        return f"Compute {a} × {b} step-by-step.", str(true_prod)

    mode = random.choice(["wrong_factor", "off_by_one", "digit_reverse"])
    if mode == "wrong_factor":
        wrong = (a + 1) * b
        return f"Compute {a} × {b} step-by-step.", str(wrong)

    if mode == "off_by_one":
        return f"Compute {a} × {b} step-by-step.", str(_off_by_one(true_prod))

    if mode == "digit_reverse":
        return f"Compute {a} × {b} step-by-step.", str(_digit_reverse(true_prod))

# ────────────────────────────────────────────────────────────────
# 4. Division with remainder traps
# ────────────────────────────────────────────────────────────────

def division_remainder(a, b, correct=True):
    true_q = a // b
    true_r = a % b

    if correct:
        return f"Compute {a} ÷ {b}. Give quotient and remainder.", f"{true_q} R{true_r}"

    mode = random.choice(["wrong_remainder", "wrong_quotient"])
    if mode == "wrong_remainder":
        wrong_r = max(0, true_r - 1)
        return f"Compute {a} ÷ {b}. Give quotient and remainder.", f"{true_q} R{wrong_r}"

    if mode == "wrong_quotient":
        wrong_q = true_q + random.choice([-1, 1])
        return f"Compute {a} ÷ {b}. Give quotient and remainder.", f"{wrong_q} R{true_r}"

# ────────────────────────────────────────────────────────────────
# 5. Inequalities
# ────────────────────────────────────────────────────────────────

def inequality(a, b, correct=True):
    true = "<" if a < b else ">" if a > b else "="
    if correct:
        return f"Compare {a} and {b}.", true

    wrong = random.choice(["<", ">", "="])
    while wrong == true:
        wrong = random.choice(["<", ">", "="])
    return f"Compare {a} and {b}.", wrong

# ────────────────────────────────────────────────────────────────
# 6. Algebraic simplification
# ────────────────────────────────────────────────────────────────

def algebra_simplify(correct=True):
    x = _rand(2, 9)
    expr = f"2x + 3x - {x}"
    true = 4 * x
    if correct:
        return f"Simplify: {expr}", str(true)

    wrong = true + random.choice([-x, x, 1])
    return f"Simplify: {expr}", str(wrong)

# ────────────────────────────────────────────────────────────────
# 7. Solve linear equation
# ────────────────────────────────────────────────────────────────

def solve_linear(correct=True):
    x = _rand(2, 9)
    a = _rand(2, 5)
    b = _rand(1, 5)
    c = a * x + b
    eq = f"{a}x + {b} = {c}"

    if correct:
        return f"Solve: {eq}", str(x)

    wrong = x + random.choice([-1, 1])
    return f"Solve: {eq}", str(wrong)

# ────────────────────────────────────────────────────────────────
# 8. Long-CoT drift
# ────────────────────────────────────────────────────────────────

def long_cot_drift(a, b, correct=True):
    true = a + b
    steps = [
        f"Let's solve {a} + {b} carefully.",
        f"Break into tens and ones.",
        f"Add ones, then tens, then combine."
    ]

    if correct:
        steps.append(f"Final: {true}")
        return "\n".join(steps), str(true)

    wrong = true + random.choice([-2, -1, 1, 2])
    steps.append(f"Final: {wrong}")
    return "\n".join(steps), str(wrong)

# ────────────────────────────────────────────────────────────────
# MAIN GENERATOR
# ────────────────────────────────────────────────────────────────

def generate_math_synthetic_v3(n_pairs=300):
    samples = []
    modes = [
        "add_cot", "sub_borrow", "mul_steps", "div_rem",
        "ineq", "alg", "solve", "cot_drift"
    ]

    for _ in range(n_pairs):
        mode = random.choice(modes)

        if mode == "add_cot":
            a, b = _rand(20, 199), _rand(20, 199)
            q1, a1 = addition_cot(a, b, True)
            q2, a2 = addition_cot(a, b, False)

        elif mode == "sub_borrow":
            a, b = _rand(20, 999), _rand(10, 899)
            q1, a1 = subtraction_borrow(a, b, True)
            q2, a2 = subtraction_borrow(a, b, False)

        elif mode == "mul_steps":
            a, b = _rand(3, 25), _rand(3, 25)
            q1, a1 = multiplication_steps(a, b, True)
            q2, a2 = multiplication_steps(a, b, False)

        elif mode == "div_rem":
            a, b = _rand(20, 200), _rand(2, 15)
            q1, a1 = division_remainder(a, b, True)
            q2, a2 = division_remainder(a, b, False)

        elif mode == "ineq":
            a, b = _rand(1, 200), _rand(1, 200)
            q1, a1 = inequality(a, b, True)
            q2, a2 = inequality(a, b, False)

        elif mode == "alg":
            q1, a1 = algebra_simplify(True)
            q2, a2 = algebra_simplify(False)

        elif mode == "solve":
            q1, a1 = solve_linear(True)
            q2, a2 = solve_linear(False)

        elif mode == "cot_drift":
            a, b = _rand(20, 199), _rand(20, 199)
            q1, a1 = long_cot_drift(a, b, True)
            q2, a2 = long_cot_drift(a, b, False)

        samples.append({"category": "Math-Synthetic-v3", "question": q1, "answer": a1, "true_label": 0})
        samples.append({"category": "Math-Synthetic-v3", "question": q2, "answer": a2, "true_label": 1})

    return samples
