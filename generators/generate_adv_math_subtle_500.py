import json
import random
from pathlib import Path

# Configuration
try:
    from config import DATA_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR

OUTPUT = DATA_DIR / "llm_adv_math_subtle_500.jsonl"
TARGET_PAIRS = 300  # Generates 300 valid + 300 invalid = 600 items
SEED = 2026

random.seed(SEED)

def q_for(q):
    return f"Calculate the result: {q}"

def generate_multiplication():
    # Large numbers to prevent memorization
    a = random.randint(112, 999)
    b = random.randint(12, 99)
    correct = a * b
    
    # Subtle drift: +/- 1, 2, or 10 (common human errors)
    drift = correct + random.choice([-10, -2, -1, 1, 2, 10])
    
    q = f"{a} * {b}"
    return q, str(correct), str(drift)

def generate_division():
    # Ensure clean divisibility for valid, then drift
    b = random.randint(3, 25)
    ans = random.randint(15, 150)
    a = b * ans
    
    correct = ans
    # Subtle drift
    drift = ans + random.choice([-1, 1])
    
    q = f"{a} / {b}"
    return q, str(correct), str(drift)

def generate_addition_large():
    a = random.randint(1000, 50000)
    b = random.randint(1000, 50000)
    correct = a + b
    # Digit flip drift (e.g. 100 -> 1000 or off by 1)
    drift = correct + random.choice([-100, -10, -1, 1, 10, 100])
    q = f"{a} + {b}"
    return q, str(correct), str(drift)

GENERATORS = [generate_multiplication, generate_multiplication, generate_division, generate_addition_large]

def main():
    items = []
    
    # Generate balanced pairs
    for _ in range(TARGET_PAIRS):
        gen = random.choice(GENERATORS)
        q_text, valid_ans, invalid_ans = gen()
        
        # Valid Item
        # Varied templates to prevent overfitting to "The answer is"
        v_templates = [
            f"The result is {valid_ans}.",
            f"{valid_ans}",
            f"Calculated precisely, it is {valid_ans}.",
            f"Answer: {valid_ans}"
        ]
        items.append({
            "question": q_for(q_text), 
            "answer": random.choice(v_templates), 
            "label": 0, 
            "domain": "math"
        })
        
        # Invalid Item (Subtle)
        # Add hedging to some invalid math to make it tricky
        i_templates = [
            f"The result is {invalid_ans}.",
            f"{invalid_ans}",
            f"It is approximately {invalid_ans}.",
            f"I believe the answer is {invalid_ans}."
        ]
        items.append({
            "question": q_for(q_text), 
            "answer": random.choice(i_templates), 
            "label": 1, 
            "domain": "math"
        })

    random.shuffle(items)
    
    OUTPUT.unlink(missing_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Generated {len(items)} subtle math examples into {OUTPUT}")

if __name__ == "__main__":
    main()