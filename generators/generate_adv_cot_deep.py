# generate_adv_cot_deep.py (New - Deeper chains with multiple drifts)
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

OUTPUT = DATA_DIR / "llm_adv_cot_deep.jsonl"
TARGET_VALID = 1000
TARGET_INVALID = 1000
MIN_STEPS = 15
MAX_STEPS = 25
DRIFT_POINTS = (2, 3)  # 2-3 drifts per invalid chain
SEED = 20260111

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
    print(f"Generated {len(items)} deep CoT examples into {OUTPUT}")

# -----------------------------
# Expanded Filler Steps
# -----------------------------
FILLER_STEPS = [
    "Step {}: Verify the units and assumptions.",
    "Step {}: Break down the components further.",
    "Step {}: Consider alternative approaches briefly.",
    "Step {}: Double-check intermediate calculations.",
    "Step {}: Relate to real-world context.",
    "Step {}: Simplify the expression if possible.",
    "Step {}: Estimate the order of magnitude.",
    "Step {}: Proceed with exact computation.",
]

# -----------------------------
# Generation Pool (Math/Real balanced)
# -----------------------------
def gen_arithmetic():
    a, b = random.randint(100, 999), random.randint(10, 99)
    correct = a + b
    q = f"What is {a} + {b}?"
    v_steps = [f"Step 1: Add {a} and {b}.", f"Step 2: Result is {correct}."]
    i_steps = [f"Step 1: Add {a} and {b}.", f"Step 2: Result is {correct + random.choice([-2, -1, 1, 2])}."]
    return q, correct, v_steps, i_steps

# Add more gens for balance (e.g., real-world)
def gen_real_world():
    problems = [("How many weeks in 84 days?", 12), ("Distance at 50 km/h for 3 hours?", 150)]
    q, correct = random.choice(problems)
    v_steps = [f"Step 1: Divide 84 by 7.", f"Step 2: Result is {correct}."]
    i_steps = [f"Step 1: Divide 84 by 7.", f"Step 2: Result is {correct + 1}."]
    return q, correct, v_steps, i_steps

GEN_POOL = [gen_arithmetic] * 3 + [gen_real_world] * 3  # 50/50 math/real

# -----------------------------
# Chain Lengthening with Multi-Drifts
# -----------------------------
def lengthen_chain(base_steps: List[str], is_invalid: bool = False) -> str:
    num_steps = random.randint(MIN_STEPS, MAX_STEPS)
    chain = []
    step_num = 1
    
    if is_invalid:
        num_drifts = random.randint(*DRIFT_POINTS)
        drift_positions = random.sample(range(1, num_steps + 1), num_drifts)
    
    for i in range(num_steps):
        if i < len(base_steps):
            step = base_steps[i].format(step_num)
        else:
            step = random.choice(FILLER_STEPS).format(step_num)
        
        if is_invalid and step_num in drift_positions:
            step += f" (with subtle error: off by {random.choice([-1,1])})"
        
        chain.append(step)
        step_num += 1
    
    return "\n".join(chain)

# -----------------------------
# Build Items (50/50 balance enforced)
# -----------------------------
def build_cot_items(target_valid, target_invalid):
    items = []
    valid_count = 0
    invalid_count = 0
    seen = set()
    
    while valid_count < target_valid or invalid_count < target_invalid:
        gen = random.choice(GEN_POOL)
        q, _, v_steps, i_steps = gen()
        
        # Valid
        if valid_count < target_valid:
            v_cot = lengthen_chain(v_steps)
            v_item = {"question": q_for(q), "answer": v_cot, "label": 0, "domain": "math" if "What is" in q else "real_world"}
            key = (v_item["question"], v_item["answer"])
            if key not in seen:
                seen.add(key)
                items.append(v_item)
                valid_count += 1
        
        # Invalid with multi-drifts
        if invalid_count < target_invalid:
            i_cot = lengthen_chain(i_steps, is_invalid=True)
            i_item = {"question": q_for(q), "answer": i_cot, "label": 1, "domain": "math" if "What is" in q else "real_world"}
            key = (i_item["question"], i_item["answer"])
            if key not in seen:
                seen.add(key)
                items.append(i_item)
                invalid_count += 1
    
    random.shuffle(items)
    return items

def main():
    items = build_cot_items(TARGET_VALID, TARGET_INVALID)
    safe_write(items)

if __name__ == "__main__":
    main()