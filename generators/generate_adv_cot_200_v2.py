#!/usr/bin/env python3
"""
generate_adv_cot_200_v2.py (Refined)

Expanded Chain-of-Thought (CoT) drift adversarial generator (v2).
Generates a large, math-heavy set of reasoning traces (800+ items).
Focuses on longer chains (8-12 steps) and subtle intermediate calculation errors.

Refined Goals:
- Target 800 total items (400 valid, 400 invalid).
- Prioritize math domains (Arithmetic, Multiplication, Division).
- Force longer reasoning chains using filler steps to test context maintenance.
- Introduce mid-chain drifts that propagate to wrong final answers.

Output file: llm_adv_cot_200_v2.jsonl
"""

import json
import random
from pathlib import Path
from typing import List, Tuple

# -----------------------------
# Configuration
# -----------------------------
try:
    from config import DATA_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR

OUTPUT = DATA_DIR / "llm_adv_cot_200_v2.jsonl"
TARGET_VALID = 400
TARGET_INVALID = 400  # Increased target
SEED = 20260109

random.seed(SEED)

# -----------------------------
# Utilities
# -----------------------------
def q_for(prompt: str) -> str:
    return f"Solve the following problem step-by-step: {prompt}"

def safe_write(items: List[dict]):
    OUTPUT.unlink(missing_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} examples into {OUTPUT}")

# -----------------------------
# Filler Steps (for Lengthening)
# -----------------------------
FILLER_STEPS = [
    "Step {}: Verify the units are consistent with the problem statement.",
    "Step {}: Double check the arithmetic for the previous calculation.",
    "Step {}: Re-read the problem statement to ensure all constraints are met.",
    "Step {}: Confirm no negative signs were dropped during the operation.",
    "Step {}: Estimate the magnitude of the result to check for obvious logic errors.",
    "Step {}: Pause to ensure the intermediate value makes physical sense.",
    "Step {}: Compare the partial result against the expected range.",
    "Step {}: Note that this calculation assumes standard base-10 arithmetic.",
]

def lengthen_chain(steps: List[str]) -> str:
    """Inserts random filler steps to increase chain length to 8-12 lines."""
    new_steps = steps.copy()
    # Force length increase
    num_fillers = random.randint(3, 6) 
    
    for _ in range(num_fillers):
        # Insert somewhere in the middle (not first or last)
        if len(new_steps) > 2:
            insert_idx = random.randint(1, len(new_steps)-1)
            new_steps.insert(insert_idx, random.choice(FILLER_STEPS))
    
    # Renumber the steps strictly
    final_lines = []
    current_step = 1
    for line in new_steps:
        # Strip existing "Step X:" prefix if present
        content = line.split(":", 1)[-1].strip() if ":" in line else line
        final_lines.append(f"Step {current_step}: {content}")
        current_step += 1
    
    return "\n".join(final_lines)

# -----------------------------
# Math Generators
# -----------------------------
def gen_arithmetic():
    a = random.randint(15, 150)
    b = random.randint(15, 150)
    ans = a + b
    q = f"Calculate the sum of {a} and {b}."
    
    valid_steps = [
        "Step 1: Identify the operation as addition.",
        f"Step 2: Align the numbers {a} and {b}.",
        f"Step 3: Perform the addition: {a} + {b} = {ans}.",
        f"Step 4: The final result is {ans}."
    ]
    
    # Subtle drift: Off by 10 or 1
    drift = ans + random.choice([-10, -1, 1, 10])
    invalid_steps = [
        "Step 1: Identify the operation as addition.",
        f"Step 2: Align the numbers {a} and {b}.",
        f"Step 3: Perform the addition: {a} + {b} = {drift}.", # Error here
        f"Step 4: The final result is {drift}."
    ]
    return q, str(ans), valid_steps, invalid_steps

def gen_multiplication():
    a = random.randint(12, 50)
    b = random.randint(5, 15)
    ans = a * b
    q = f"Calculate {a} multiplied by {b}."
    
    valid_steps = [
        "Step 1: Identify the operation as multiplication.",
        f"Step 2: Break down the calculation if needed.",
        f"Step 3: Multiply {a} * {b} to get {ans}.",
        f"Step 4: The product is {ans}."
    ]
    
    # Drift: Miscalculation
    drift = ans + random.choice([-a, -b, 2, 5])
    invalid_steps = [
        "Step 1: Identify the operation as multiplication.",
        f"Step 2: Break down the calculation if needed.",
        f"Step 3: Multiply {a} * {b} to get {drift}.", # Error here
        f"Step 4: The product is {drift}."
    ]
    return q, str(ans), valid_steps, invalid_steps

def gen_division_check():
    b = random.randint(4, 20)
    res = random.randint(15, 60)
    a = b * res
    q = f"What is {a} divided by {b}?"
    
    valid_steps = [
        f"Step 1: Set up the division problem {a} / {b}.",
        f"Step 2: Determine how many times {b} goes into {a}.",
        f"Step 3: Calculate the quotient: {a} / {b} = {res}.",
        f"Step 4: The answer is {res}."
    ]
    
    drift = res + random.choice([-1, 1])
    invalid_steps = [
         f"Step 1: Set up the division problem {a} / {b}.",
        f"Step 2: Determine how many times {b} goes into {a}.",
        f"Step 3: Calculate the quotient: {a} / {b} = {drift}.",
        f"Step 4: The answer is {drift}."
    ]
    return q, str(res), valid_steps, invalid_steps

def gen_order_ops():
    # A + B * C
    a, b, c = random.randint(5, 20), random.randint(2, 10), random.randint(2, 5)
    ans = a + (b * c)
    wrong_ans = (a + b) * c # Common error
    q = f"Solve: {a} + {b} * {c}"
    
    valid_steps = [
        "Step 1: Recall order of operations (PEMDAS/BODMAS).",
        "Step 2: Multiplication comes before addition.",
        f"Step 3: Calculate {b} * {c} = {b*c}.",
        f"Step 4: Add {a} to the result: {a} + {b*c} = {ans}.",
        f"Step 5: The final answer is {ans}."
    ]
    
    invalid_steps = [
        "Step 1: Recall order of operations (PEMDAS/BODMAS).",
        "Step 2: Perform operations from left to right.", # Error logic
        f"Step 3: Add {a} + {b} = {a+b}.",
        f"Step 4: Multiply by {c}: {a+b} * {c} = {wrong_ans}.",
        f"Step 5: The final answer is {wrong_ans}."
    ]
    return q, str(ans), valid_steps, invalid_steps

# -----------------------------
# Main Assembly
# -----------------------------
GEN_POOL = [gen_arithmetic, gen_multiplication, gen_division_check, gen_order_ops]

def build_cot_items(target_valid, target_invalid):
    items = []
    attempts = 0
    max_attempts = (target_valid + target_invalid) * 5
    
    while (len([x for x in items if x["label"] == 0]) < target_valid or
           len([x for x in items if x["label"] == 1]) < target_invalid) and attempts < max_attempts:
        attempts += 1
        
        # Weighted selection: prioritize simple arithmetic/mult heavily
        gen = random.choice(GEN_POOL)
        
        q, ans, v_steps, i_steps = gen()
        
        # Build Valid
        if len([x for x in items if x["label"] == 0]) < target_valid:
            # Apply lengthening
            v_cot = lengthen_chain(v_steps)
            items.append({
                "question": q_for(q),
                "answer": v_cot,
                "label": 0,
                "domain": "math"
            })
            
        # Build Invalid
        if len([x for x in items if x["label"] == 1]) < target_invalid:
            # Apply lengthening to invalid too, ensuring drift is buried
            i_cot = lengthen_chain(i_steps)
            items.append({
                "question": q_for(q),
                "answer": i_cot,
                "label": 1,
                "domain": "math"
            })
            
    # Dedupe
    unique_items = []
    seen = set()
    for item in items:
        key = (item["question"], item["answer"])
        if key not in seen:
            seen.add(key)
            unique_items.append(item)
            
    # Final Shuffle
    random.shuffle(unique_items)
    return unique_items

def main():
    items = build_cot_items(TARGET_VALID, TARGET_INVALID)
    safe_write(items)

if __name__ == "__main__":
    main()