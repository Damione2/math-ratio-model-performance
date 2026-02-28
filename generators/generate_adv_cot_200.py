import json
import random
from pathlib import Path

# Add centralized config support
try:
    from config import DATA_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR
    
# ============================================
# Utility
# ============================================

def choice(*xs):
    return random.choice(xs)

def q_for(q):
    return f"Solve the following problem step-by-step: {q}"

# ============================================
# CoT templates (valid + drift)
# ============================================

MATH_PROBLEMS = [
    ("What is 12 * 7?", 84),
    ("What is 15 + 28?", 43),
    ("What is 9 * 9?", 81),
    ("What is 144 / 12?", 12),
    ("What is 17 + 19?", 36),
    ("What is 25 * 4?", 100),
    ("What is 63 - 18?", 45),
    ("What is 8 * 11?", 88),
    ("What is 72 / 8?", 9),
    ("What is 14 + 27?", 41),
]

REAL_WORLD_PROBLEMS = [
    ("How many days are in 3 weeks?", 21),
    ("If a car travels 60 km/h for 2 hours, how far does it go?", 120),
    ("How many minutes are in 2 hours?", 120),
    ("If you buy 3 apples and each costs 2 dollars, total cost?", 6),
    ("How many seconds are in 5 minutes?", 300),
]

# ============================================
# CoT generators
# ============================================

def gen_valid_cot(question, answer):
    steps = [
        f"Let's break it down.",
        f"We identify the operation needed.",
        f"We compute the intermediate values.",
        f"The final result is {answer}.",
    ]
    return "\n".join(steps)

def gen_invalid_cot(question, answer):
    # Drift happens in the last step
    wrong = answer + random.choice([-3, -2, -1, 1, 2, 3])
    steps = [
        f"Let's break it down.",
        f"We identify the operation needed.",
        f"We compute the intermediate values.",
        f"Therefore the final answer is {wrong}.",  # drift
    ]
    return "\n".join(steps)

def gen_invalid_mid_drift(question, answer):
    # Drift happens in the middle step
    wrong_mid = answer + random.choice([-5, -4, 4, 5])
    wrong_final = wrong_mid  # consistent wrong chain
    steps = [
        f"Let's break it down.",
        f"We compute an intermediate value incorrectly as {wrong_mid}.",
        f"Using that, we conclude the answer is {wrong_final}.",
    ]
    return "\n".join(steps)

# ============================================
# Main generator
# ============================================

def main():
    random.seed(42)

    # Use centralized data directory
    output = DATA_DIR / "llm_adv_cot_200.jsonl"
    output.unlink(missing_ok=True)
    final = []
    needed_pairs = 100  # 100 valid + 100 invalid = 200 total

    while len(final) < needed_pairs * 2:
        if random.random() < 0.6:
            q, ans = random.choice(MATH_PROBLEMS)
            domain = "math"
        else:
            q, ans = random.choice(REAL_WORLD_PROBLEMS)
            domain = "real_world"

        valid = gen_valid_cot(q, ans)
        invalid = choice(gen_invalid_cot(q, ans), gen_invalid_mid_drift(q, ans))

        valid_item = {
            "question": q_for(q),
            "answer": valid,
            "label": 0,
            "domain": domain,
        }

        invalid_item = {
            "question": q_for(q),
            "answer": invalid,
            "label": 1,
            "domain": domain,
        }

        final.append(valid_item)
        final.append(invalid_item)

    with output.open("w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(final)} chain-of-thought drift examples into {output}")

if __name__ == "__main__":
    main()
