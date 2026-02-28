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
# Fact base (broad, diverse)
# ============================================

FACTS = [
    ("Who developed the theory of evolution?", "Charles Darwin", ["Gregor Mendel", "Louis Pasteur"]),
    ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza"]),
    ("Who painted Starry Night?", "Vincent van Gogh", ["Claude Monet", "Pablo Picasso"]),
    ("What is the largest desert?", "Sahara", ["Gobi", "Arabian"]),
    ("Who discovered electricity?", "Benjamin Franklin", ["Thomas Edison", "Nikola Tesla"]),
    ("What is the smallest prime number?", "2", ["1", "3"]),
    ("What is the boiling point of water in Celsius?", "100", ["90", "120"]),
    ("Who wrote Hamlet?", "William Shakespeare", ["Christopher Marlowe", "John Milton"]),
    ("What is the chemical symbol for sodium?", "Na", ["So", "Sn"]),
    ("What is the longest bone in the human body?", "Femur", ["Tibia", "Humerus"]),
]

# ============================================
# Valid context templates
# ============================================

VALID_TEMPLATES = [
    lambda correct: (
        f"The topic has been widely studied and documented in academic literature. "
        f"Historical records, scientific analyses, and expert consensus all point toward a single conclusion. "
        f"After reviewing the most reliable sources, the correct answer is {correct}."
    ),
    lambda correct: (
        f"Many scholars have examined this subject from different angles, but the consensus remains consistent. "
        f"Primary sources and verified data confirm that the correct answer is {correct}."
    ),
    lambda correct: (
        f"Although the topic spans multiple disciplines, the factual evidence is clear. "
        f"Researchers agree that the correct answer is {correct}."
    ),
]

# ============================================
# Invalid context templates (subtle contradictions)
# ============================================

INVALID_TEMPLATES = [
    lambda wrong: (
        f"The subject is often misunderstood due to overlapping historical narratives. "
        f"Some sources present conflicting interpretations, but many popular accounts suggest the answer is {wrong}. "
        f"Based on this perspective, the answer appears to be {wrong}."
    ),
    lambda wrong: (
        f"While experts debate certain aspects of this topic, a number of widely circulated explanations "
        f"point toward {wrong} as the correct answer. "
        f"Given this context, the answer seems to be {wrong}."
    ),
    lambda wrong: (
        f"Despite the availability of factual information, misconceptions persist in educational materials. "
        f"One common interpretation identifies the answer as {wrong}. "
        f"Following that line of reasoning, the answer is {wrong}."
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

    output = DATA_DIR / "llm_adv_various_300.jsonl"
    output.unlink(missing_ok=True)
    
    final = []
    needed_pairs = 150  # 150 valid + 150 invalid = 300 total

    while len(final) < needed_pairs * 2:
        q, correct, wrongs = random.choice(FACTS)
        wrong = random.choice(wrongs)

        valid_context = choice(*VALID_TEMPLATES)(correct)
        invalid_context = choice(*INVALID_TEMPLATES)(wrong)

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

    print(f"Generated {len(final)} various adversarial examples into {output}")

if __name__ == "__main__":
    main()