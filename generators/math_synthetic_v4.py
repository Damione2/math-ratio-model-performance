#!/usr/bin/env python3
# generators/math_synthetic_v4.py
"""
Math Synthetic v4 Generator
- Focus: fractions, decimals, exponents
- Paired correct / wrong samples
- Error families: simplification mistakes, common-denominator errors,
  rounding/precision traps, decimal-to-fraction conversion errors,
  exponent sign mistakes, root/power confusion, scientific notation traps,
  CoT explanations for fractions and exponent steps.
"""

import random
import math
import json
from typing import List, Dict, Tuple

SEED = 12345
random.seed(SEED)


# -----------------------
# Utilities
# -----------------------
def _rand(a: int, b: int) -> int:
    return random.randint(a, b)

def _gcd(a: int, b: int) -> int:
    return math.gcd(a, b)

def _simplify_fraction(n: int, d: int) -> Tuple[int, int]:
    g = _gcd(abs(n), abs(d))
    return n // g, d // g

def _to_mixed(n: int, d: int) -> Tuple[int, int, int]:
    whole = n // d
    rem = n % d
    return whole, rem, d

def _float_str(x: float, prec: int = 3) -> str:
    return f"{x:.{prec}f}"

def _maybe_neg(x: int, allow_neg: bool = True) -> int:
    return x if random.random() > 0.2 or not allow_neg else -x

# -----------------------
# Fraction families
# -----------------------
def fraction_plain(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """Simple fraction addition a/b + c/d with common-denominator and simplification errors."""
    # a and b are numerator/denominator for first fraction; generate second
    c = _rand(1, max(3, b))
    d = _rand(2, max(4, b + 2))
    # ensure denominators not zero
    n1, d1 = a, b
    n2, d2 = c, d
    # compute true result
    num = n1 * d2 + n2 * d1
    den = d1 * d2
    snum, sden = _simplify_fraction(num, den)
    q = f"What is {n1}/{d1} + {n2}/{d2}?"
    if correct:
        # sometimes present as mixed number
        if snum >= sden:
            whole, rem, denm = _to_mixed(snum, sden)
            if rem == 0:
                return q, str(whole)
            return q, f"{whole} {rem}/{denm}"
        return q, f"{snum}/{sden}"
    # wrong variants
    mode = random.choice(["no_simplify", "wrong_common", "swap_num_den", "mixed_mistake"])
    if mode == "no_simplify":
        return q, f"{num}/{den}"
    if mode == "wrong_common":
        # incorrectly add numerators and denominators directly
        wrong = f"{n1 + n2}/{d1 + d2}"
        return q, wrong
    if mode == "swap_num_den":
        wrong = f"{den}/{num}"
        return q, wrong
    # mixed_mistake: compute mixed incorrectly
    whole, rem, denm = _to_mixed(snum, sden)
    wrong_rem = max(1, rem + random.choice([-1, 1]))
    return q, f"{whole} {wrong_rem}/{denm}"

def fraction_cot(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """CoT for fraction addition with explicit steps and subtle mistakes."""
    c = _rand(1, max(3, b))
    d = _rand(2, max(4, b + 2))
    n1, d1 = a, b
    n2, d2 = c, d
    steps = []
    steps.append(f"Add {n1}/{d1} and {n2}/{d2}.")
    steps.append("Step 1: Find common denominator.")
    common = d1 * d2
    steps.append(f"Common denominator = {d1} * {d2} = {common}.")
    steps.append("Step 2: Convert fractions.")
    nn1 = n1 * d2
    nn2 = n2 * d1
    steps.append(f"{n1}/{d1} -> {nn1}/{common}; {n2}/{d2} -> {nn2}/{common}.")
    total = nn1 + nn2
    steps.append(f"Step 3: Add numerators: {nn1} + {nn2} = {total}.")
    snum, sden = _simplify_fraction(total, common)
    steps.append(f"Simplify {total}/{common} -> {snum}/{sden}.")
    if correct:
        if snum >= sden:
            whole, rem, denm = _to_mixed(snum, sden)
            if rem == 0:
                return "\n".join(steps + [f"Final: {whole}"]), str(whole)
            return "\n".join(steps + [f"Final: {whole} {rem}/{denm}"]), f"{whole} {rem}/{denm}"
        return "\n".join(steps + [f"Final: {snum}/{sden}"]), f"{snum}/{sden}"
    # wrong variants: wrong common, wrong convert, wrong simplify
    mode = random.choice(["wrong_common", "wrong_convert", "wrong_simplify", "off_by_one"])
    if mode == "wrong_common":
        wrong_common = d1 + d2
        steps[2] = f"Common denominator = {d1} + {d2} = {wrong_common}."
        wrong_total = n1 * (wrong_common // d1) + n2 * (wrong_common // d2)
        return "\n".join(steps + [f"Final: {wrong_total}/{wrong_common}"]), f"{wrong_total}/{wrong_common}"
    if mode == "wrong_convert":
        # forget to multiply one numerator
        steps[3] = f"{n1}/{d1} -> {nn1}/{common}; {n2}/{d2} -> {n2}/{common} (forgot multiply)."
        wrong_total = nn1 + n2
        wsnum, wsden = _simplify_fraction(wrong_total, common)
        return "\n".join(steps + [f"Final: {wsnum}/{wsden}"]), f"{wsnum}/{wsden}"
    if mode == "wrong_simplify":
        # simplify incorrectly by dividing by wrong gcd
        wrong_div = max(2, _gcd(total, common) - 1)
        wsnum = total // wrong_div
        wsden = common // wrong_div
        return "\n".join(steps + [f"Final: {wsnum}/{wsden} (incorrect simplification)"]), f"{wsnum}/{wsden}"
    # off_by_one
    return "\n".join(steps + [f"Final: {total + random.choice([-1,1])}/{common}"]), str(total + random.choice([-1,1])) + f"/{common}"

# -----------------------
# Decimal families
# -----------------------
def decimal_rounding(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """Decimal addition and rounding traps."""
    # create decimals with varying precision
    da = a / (10 ** _rand(1, 3))
    db = b / (10 ** _rand(1, 3))
    true = da + db
    q = f"What is {da} + {db}? Round to 3 decimal places."
    if correct:
        return q, _float_str(true, 3)
    # wrong variants: rounding before adding, truncation, precision off-by-one
    mode = random.choice(["round_before", "truncate", "off_by_one"])
    if mode == "round_before":
        ra = round(da, 3)
        rb = round(db, 3)
        wrong = ra + rb
        return q, _float_str(wrong, 3)
    if mode == "truncate":
        ta = math.floor(da * 1000) / 1000.0
        tb = math.floor(db * 1000) / 1000.0
        wrong = ta + tb
        return q, _float_str(wrong, 3)
    return q, _float_str(true + random.choice([-0.001, 0.001, 0.002]), 3)

def decimal_to_fraction(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """Convert decimal to fraction with common conversion mistakes."""
    # create decimal with finite representation
    denom = 10 ** _rand(1, 3)
    num = a
    dec = num / denom
    q = f"Convert {dec} to a fraction in simplest form."
    if correct:
        sn, sd = _simplify_fraction(num, denom)
        return q, f"{sn}/{sd}"
    mode = random.choice(["no_simplify", "wrong_denominator", "approximate"])
    if mode == "no_simplify":
        return q, f"{num}/{denom}"
    if mode == "wrong_denominator":
        return q, f"{num}/{denom//10 if denom>=10 else denom+1}"
    return q, f"{round(dec, 3)}"

# -----------------------
# Exponent families
# -----------------------
def exponent_basic(base: int, exp: int, correct: bool = True) -> Tuple[str, str]:
    """Basic exponent evaluation with sign and overflow traps."""
    true = base ** exp
    q = f"What is {base}^{exp}?"
    if correct:
        return q, str(true)
    mode = random.choice(["sign_error", "exp_off_by_one", "root_confusion", "scientific_mistake"])
    if mode == "sign_error":
        # confuse negative exponent sign
        if exp < 0:
            wrong = base ** abs(exp)
            return q, str(wrong)
        else:
            wrong = base ** (-exp)
            return q, str(wrong)
    if mode == "exp_off_by_one":
        return q, str(base ** (exp + random.choice([-1, 1])))
    if mode == "root_confusion":
        # confuse power with root: return base ** (exp // 2)
        return q, str(base ** max(1, exp // 2))
    # scientific_mistake: represent in scientific notation incorrectly
    sci = f"{true:.2e}"
    wrong = sci.replace("e", "E+")
    return q, wrong

def exponent_cot(base: int, exp: int, correct: bool = True) -> Tuple[str, str]:
    """CoT for exponentiation showing repeated multiplication and common mistakes."""
    steps = [f"Compute {base}^{exp} by repeated multiplication."]
    prod = 1
    for i in range(exp):
        steps.append(f"Step {i+1}: multiply by {base} -> {prod} * {base} = {prod * base}")
        prod *= base
    steps.append(f"Final: {prod}")
    if correct:
        return "\n".join(steps), str(prod)
    # wrong variant: stop one step early or multiply wrong factor
    mode = random.choice(["stop_early", "wrong_factor", "off_by_one"])
    if mode == "stop_early":
        wrong = prod // base
        steps[-1] = f"Final: {wrong} (stopped early)"
        return "\n".join(steps), str(wrong)
    if mode == "wrong_factor":
        wrong = prod + base
        steps[-1] = f"Final: {wrong} (added base instead of multiply)"
        return "\n".join(steps), str(wrong)
    return "\n".join(steps[:-1] + [f"Final: {prod + 1}"]), str(prod + 1)

# -----------------------
# Mixed families and edge cases
# -----------------------
def fraction_decimal_mix(a: int, b: int, correct: bool = True) -> Tuple[str, str]:
    """Add a fraction and a decimal; common mistakes: convert incorrectly or round early."""
    # fraction n/d and decimal dec
    n = a
    d = b if b != 0 else 2
    dec_denom = 10 ** _rand(1, 3)
    dec_num = _rand(1, dec_denom - 1)
    dec = dec_num / dec_denom
    q = f"What is {n}/{d} + {dec}? Give exact fraction."
    # true: convert decimal to fraction and add
    dec_num_scaled = dec_num * d
    total_num = n * dec_denom + dec_num * d
    # compute exact fraction via common denom
    common = d * dec_denom
    total_num = n * dec_denom + dec_num * d
    sn, sd = _simplify_fraction(int(total_num), common)
    if correct:
        return q, f"{sn}/{sd}"
    mode = random.choice(["round_decimal", "wrong_convert", "no_simplify"])
    if mode == "round_decimal":
        # round decimal to 2 decimals then convert
        rdec = round(dec, 2)
        return q, f"{rdec}"
    if mode == "wrong_convert":
        # convert decimal to fraction with wrong denom
        return q, f"{dec_num}/{dec_denom//10 if dec_denom>=10 else dec_denom+1}"
    return q, f"{int(total_num)}/{common}"

# -----------------------
# Public API
# -----------------------
def generate_math_synthetic_v4(n_pairs: int = 300) -> List[Dict]:
    """
    Generate n_pairs * 2 samples:
      - For each pair: one correct, one wrong
      - label = 0 → non-hallucination (correct)
      - label = 1 → hallucination (wrong)
    """
    samples: List[Dict] = []
    modes = [
        "frac_plain", "frac_cot", "dec_round", "dec_to_frac",
        "exp_basic", "exp_cot", "frac_dec_mix", "edge_cases"
    ]
    for i in range(n_pairs):
        mode = random.choice(modes)
        if mode == "frac_plain":
            a = _rand(1, 9)
            b = _rand(2, 12)
            q1, a1 = fraction_plain(a, b, True)
            q2, a2 = fraction_plain(a, b, False)
        elif mode == "frac_cot":
            a = _rand(2, 9)
            b = _rand(2, 12)
            q1, a1 = fraction_cot(a, b, True)
            q2, a2 = fraction_cot(a, b, False)
        elif mode == "dec_round":
            a = _rand(10, 999)
            b = _rand(10, 999)
            q1, a1 = decimal_rounding(a, b, True)
            q2, a2 = decimal_rounding(a, b, False)
        elif mode == "dec_to_frac":
            a = _rand(1, 9)
            b = _rand(2, 100)
            q1, a1 = decimal_to_fraction(a, b, True)
            q2, a2 = decimal_to_fraction(a, b, False)
        elif mode == "exp_basic":
            base = _rand(2, 6)
            exp = random.choice([2, 3, 4, -1, -2])
            q1, a1 = exponent_basic(base, exp, True)
            q2, a2 = exponent_basic(base, exp, False)
        elif mode == "exp_cot":
            base = _rand(2, 5)
            exp = random.choice([2, 3, 4])
            q1, a1 = exponent_cot(base, exp, True)
            q2, a2 = exponent_cot(base, exp, False)
        elif mode == "frac_dec_mix":
            a = _rand(1, 9)
            b = _rand(2, 12)
            q1, a1 = fraction_decimal_mix(a, b, True)
            q2, a2 = fraction_decimal_mix(a, b, False)
        else:  # edge_cases
            # combine exponent and fraction: (a/b)^c or decimal exponent confusion
            a = _rand(1, 9)
            b = _rand(2, 9)
            c = _rand(2, 4)
            q_true = f"What is ({a}/{b})^{c} simplified?"
            true_num = (a ** c)
            true_den = (b ** c)
            sn, sd = _simplify_fraction(true_num, true_den)
            q1, a1 = q_true, f"{sn}/{sd}"
            # wrong: raise numerator only or wrong simplification
            wrong = f"{true_num}/{true_den // 2 if true_den % 2 == 0 else true_den + 1}"
            q2, a2 = q_true, wrong

        idx_base = len(samples) + 1
        samples.append({
            "index": idx_base,
            "category": "Math-Synthetic-v4",
            "question": q1,
            "answer": a1,
            "true_label": 0
        })
        samples.append({
            "index": idx_base + 1,
            "category": "Math-Synthetic-v4",
            "question": q2,
            "answer": a2,
            "true_label": 1
        })

    # ensure reproducible ordering
    return samples

def save_math_synthetic_v4_jsonl(path: str, n_pairs: int = 300):
    samples = generate_math_synthetic_v4(n_pairs=n_pairs)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    out = "math_synthetic_v4_demo.jsonl"
    save_math_synthetic_v4_jsonl(out, n_pairs=20)
    print(f"✅ Wrote demo math_synthetic_v4 to {out}")
