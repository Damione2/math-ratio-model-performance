#!/usr/bin/env python3
# generators/math_equation_systems_v1.py
"""
Math Equation Systems v1
- Generates paired correct / wrong samples for 2x2 linear systems
- Error families: arithmetic slip, sign flip, swapped coefficients, wrong elimination,
  treat as dependent (infinite solutions) or inconsistent (no solution) mistakes
- API:
    generate_math_equation_systems_v1(n_pairs=200) -> List[Dict]
    save_math_equation_systems_v1_jsonl(path, n_pairs=200)
"""

import random
import json
from typing import List, Dict, Tuple

SEED = 4242
random.seed(SEED)


# -----------------------
# Utilities
# -----------------------
def _rand(a: int, b: int) -> int:
    return random.randint(a, b)


def _safe_det(a: int, b: int, c: int, d: int) -> int:
    return a * d - b * c


def _solve_2x2(a: int, b: int, c: int, d: int, e: int, f: int) -> Tuple[bool, Tuple[float, float]]:
    """
    Solve:
      a*x + b*y = e
      c*x + d*y = f
    Returns (has_unique_solution, (x, y))
    """
    det = _safe_det(a, b, c, d)
    if det == 0:
        return False, (0.0, 0.0)
    x = (e * d - b * f) / det
    y = (a * f - e * c) / det
    return True, (x, y)


def _fmt_system(a: int, b: int, e: int, c: int, d: int, f: int) -> str:
    return f"Solve the system:\n{a}x + {b}y = {e}\n{c}x + {d}y = {f}"


def _fmt_solution(x: float, y: float, precision: int = 6) -> str:
    # Format with reasonable precision, strip trailing zeros
    xs = f"{x:.{precision}f}".rstrip("0").rstrip(".")
    ys = f"{y:.{precision}f}".rstrip("0").rstrip(".")
    return f"x = {xs}, y = {ys}"


# -----------------------
# Wrong-solution generators
# -----------------------
def _wrong_arithmetic(x: float, y: float) -> Tuple[float, float]:
    # small arithmetic slip on one variable
    dx = x + random.choice([-1, 1]) * random.choice([0.5, 1, 2])
    dy = y
    return dx, dy


def _wrong_sign(x: float, y: float) -> Tuple[float, float]:
    return -x, y


def _wrong_swap(x: float, y: float) -> Tuple[float, float]:
    return y, x


def _wrong_elimination(a, b, e, c, d, f, x_true, y_true) -> Tuple[float, float]:
    # simulate elimination mistake: subtract instead of add leading to off-by-one error
    dx = x_true + random.choice([-2, -1, 1, 2])
    dy = y_true + random.choice([-1, 1])
    return dx, dy


def _pretend_dependent(a, b, e, c, d, f) -> Tuple[str, str]:
    # produce a wrong answer claiming infinite or no solution
    # choose between "infinite solutions" and "no solution"
    if random.random() < 0.5:
        return "Infinite solutions", "The system has infinitely many solutions"
    else:
        return "No solution", "The system is inconsistent and has no solution"


# -----------------------
# Main pair generator
# -----------------------
def generate_math_equation_systems_v1(n_pairs: int = 200) -> List[Dict]:
    """
    Generate n_pairs * 2 samples:
      - For each pair: one correct (label 0), one wrong (label 1)
      - Returns list of dicts with keys: index, category, question, answer, true_label
    """
    samples: List[Dict] = []

    for i in range(n_pairs):
        # Generate coefficients ensuring a unique solution most of the time
        # Allow occasional degenerate systems to create adversarial cases
        attempt = 0
        while True:
            a = _rand(-9, 9)
            b = _rand(-9, 9)
            c = _rand(-9, 9)
            d = _rand(-9, 9)
            # avoid trivial zero rows
            if a == b == 0 or c == d == 0:
                attempt += 1
                if attempt > 10:
                    a = 1
                    b = 1
                    c = 1
                    d = 2
                    break
                continue
            # choose RHS to produce reasonable numeric solutions
            x_true = random.uniform(-10, 10)
            y_true = random.uniform(-10, 10)
            # compute RHS exactly to ensure consistent solution
            e = int(round(a * x_true + b * y_true))
            f = int(round(c * x_true + d * y_true))
            # recompute true solution from integer system
            has_unique, (x_sol, y_sol) = _solve_2x2(a, b, c, d, e, f)
            # prefer unique solutions but allow some degenerate cases
            if has_unique or random.random() < 0.05:
                break
            attempt += 1
            if attempt > 20:
                break

        question = _fmt_system(a, b, e, c, d, f)

        # Correct answer
        has_unique, (x_sol, y_sol) = _solve_2x2(a, b, c, d, e, f)
        if has_unique:
            correct_answer = _fmt_solution(x_sol, y_sol, precision=6)
        else:
            # if determinant zero, decide whether infinite or none based on consistency
            # check proportionality
            if a == c == 0 and b == d == 0:
                correct_answer = "Infinite solutions"
            else:
                # check if rows proportional and RHS proportional
                if a * d == b * c and (a * f == e * c and b * f == e * d):
                    correct_answer = "Infinite solutions"
                else:
                    correct_answer = "No solution"

        # Wrong answer: pick an error mode
        error_mode = random.choice([
            "arithmetic_slip",
            "sign_error",
            "swap_variables",
            "elimination_mistake",
            "pretend_degenerate"
        ])

        if error_mode == "arithmetic_slip" and has_unique:
            wx, wy = _wrong_arithmetic(x_sol, y_sol)
            wrong_answer = _fmt_solution(wx, wy, precision=6)
        elif error_mode == "sign_error" and has_unique:
            wx, wy = _wrong_sign(x_sol, y_sol)
            wrong_answer = _fmt_solution(wx, wy, precision=6)
        elif error_mode == "swap_variables" and has_unique:
            wx, wy = _wrong_swap(x_sol, y_sol)
            wrong_answer = _fmt_solution(wx, wy, precision=6)
        elif error_mode == "elimination_mistake" and has_unique:
            wx, wy = _wrong_elimination(a, b, e, c, d, f, x_sol, y_sol)
            wrong_answer = _fmt_solution(wx, wy, precision=6)
        else:
            # pretend degenerate: claim infinite/no solution even if unique
            label_text, label_explanation = _pretend_dependent(a, b, e, c, d, f)
            wrong_answer = label_explanation

        idx_base = len(samples) + 1
        samples.append({
            "index": idx_base,
            "category": "Math-Equation-Systems-v1",
            "question": question,
            "answer": correct_answer,
            "true_label": 0
        })
        samples.append({
            "index": idx_base + 1,
            "category": "Math-Equation-Systems-v1",
            "question": question,
            "answer": wrong_answer,
            "true_label": 1
        })

    return samples


def save_math_equation_systems_v1_jsonl(path: str, n_pairs: int = 200):
    samples = generate_math_equation_systems_v1(n_pairs=n_pairs)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    out = "math_equation_systems_v1_demo.jsonl"
    save_math_equation_systems_v1_jsonl(out, n_pairs=50)
    print(f"Wrote demo math_equation_systems_v1 to {out}")
