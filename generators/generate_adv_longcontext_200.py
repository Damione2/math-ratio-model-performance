import json
import random
from pathlib import Path

# ============================================
# Utility
# ============================================

def choice(*xs):
    return random.choice(xs)

def q_for(q):
    return f"Read the passage and answer the question: {q}"

# ============================================
# Base facts (valid + wrong alternatives)
# ============================================

FACTS = [
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Charles Dickens", "Emily Brontë"]),
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Marie Curie", "Isaac Newton"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael"]),
    ("What is the largest planet?", "Jupiter", ["Saturn", "Neptune"]),
    ("What is the tallest mountain?", "Mount Everest", ["K2", "Kangchenjunga"]),
    ("What is the longest river?", "Nile", ["Amazon", "Yangtze"]),
    ("Who developed the theory of relativity?", "Albert Einstein", ["Niels Bohr", "Stephen Hawking"]),
    ("What gas do plants absorb?", "Carbon dioxide", ["Oxygen", "Nitrogen"]),
]

# ============================================
# Long‑context templates
# ============================================

VALID_CONTEXT = [
    lambda correct: (
        f"The topic has been studied extensively by historians and researchers. "
        f"Many sources confirm the details surrounding this subject. "
        f"In particular, the individual associated with this achievement has been "
        f"recognized globally for their contribution. "
        f"After reviewing the historical records, it is clear that the correct answer is {correct}."
    ),
    lambda correct: (
        f"Throughout history, scholars have debated various aspects of this topic, "
        f"but modern consensus is strong. "
        f"Multiple academic references point toward the same conclusion. "
        f"Based on the most reliable evidence, the answer is {correct}."
    ),
]

INVALID_CONTEXT = [
    lambda wrong: (
        f"The subject has a rich historical background, and many people often confuse "
        f"the details due to overlapping contributions from different figures. "
        f"Despite this, some sources incorrectly attribute the achievement to others. "
        f"After reviewing the available information, the answer is {wrong}."
    ),
    lambda wrong: (
        f"Although the topic is widely discussed, misconceptions persist in popular culture. "
        f"Some educational materials even present conflicting information. "
        f"Given the context and common interpretations, the answer appears to be {wrong}."
    ),
]

# ============================================
# Main generator
# ============================================

def main():
    random.seed(42)
    
    try:
        from config import DATA_DIR
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from config import DATA_DIR

    output = DATA_DIR / "llm_adv_longcontext_200.jsonl"
    output.unlink(missing_ok=True)

    final = []
    needed_pairs = 100  # 100 valid + 100 invalid = 200 total

    while len(final) < needed_pairs * 2:
        q, correct, wrongs = random.choice(FACTS)
        wrong = random.choice(wrongs)

        valid_context = choice(*VALID_CONTEXT)(correct)
        invalid_context = choice(*INVALID_CONTEXT)(wrong)

        valid_item = {
            "question": q_for(q),
            "answer": valid_context,
            "label": 0,
            "domain": "real_world",
        }

        invalid_item = {
            "question": q_for(q),
            "answer": invalid_context,
            "label": 1,
            "domain": "real_world",
        }

        final.append(valid_item)
        final.append(invalid_item)

    with output.open("w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(final)} long‑context contradiction examples into {output}")

if __name__ == "__main__":
    main()
