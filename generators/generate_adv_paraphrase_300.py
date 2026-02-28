import json
import random
from pathlib import Path

# ============================================
# Utility
# ============================================

def choice(*xs):
    return random.choice(xs)

def q_for(topic):
    return f"Answer the following question: {topic}"

# ============================================
# Fact templates (valid + adversarial paraphrases)
# ============================================

FACT_PAIRS = [
    # Capitals
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver"]),
    ("What is the capital of Brazil?", "Brasília", ["Rio de Janeiro", "São Paulo"]),
    ("What is the capital of Switzerland?", "Bern", ["Zurich", "Geneva"]),

    # Authors
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Charles Dickens", "Emily Brontë"]),
    ("Who wrote '1984'?", "George Orwell", ["Aldous Huxley", "Ray Bradbury"]),
    ("Who wrote 'The Hobbit'?", "J.R.R. Tolkien", ["C.S. Lewis", "George R.R. Martin"]),

    # Dates
    ("When did World War II end?", "1945", ["1944", "1950"]),
    ("When was the Declaration of Independence signed?", "1776", ["1775", "1781"]),
    ("When did the Apollo 11 moon landing occur?", "1969", ["1971", "1959"]),

    # Science
    ("What is the chemical symbol for gold?", "Au", ["Ag", "Go"]),
    ("What planet is known as the Red Planet?", "Mars", ["Jupiter", "Venus"]),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide", ["Oxygen", "Nitrogen"]),

    # Geography
    ("Which river is the longest in the world?", "Nile", ["Amazon", "Yangtze"]),
    ("Which mountain is the tallest in the world?", "Mount Everest", ["K2", "Kangchenjunga"]),
    ("Which ocean is the largest?", "Pacific Ocean", ["Atlantic Ocean", "Indian Ocean"]),

    # History
    ("Who was the first President of the United States?", "George Washington", ["Thomas Jefferson", "Abraham Lincoln"]),
    ("Where did the Renaissance begin?", "Italy", ["France", "Germany"]),
    ("Which empire built the Colosseum?", "Roman Empire", ["Greek Empire", "Ottoman Empire"]),

    # Biology
    ("What is the largest mammal?", "Blue whale", ["Elephant", "Giraffe"]),
    ("What is the fastest land animal?", "Cheetah", ["Lion", "Greyhound"]),
    ("What do bees produce?", "Honey", ["Wax", "Pollen"]),

    # Technology
    ("Who founded Microsoft?", "Bill Gates", ["Steve Jobs", "Elon Musk"]),
    ("What does CPU stand for?", "Central Processing Unit", ["Computer Power Unit", "Central Program Utility"]),
    ("What company created the Android OS?", "Google", ["Samsung", "Nokia"]),
]

# ============================================
# Paraphrase wrappers
# ============================================

VALID_TEMPLATES = [
    "The correct answer is {}.",
    "{} is the correct answer.",
    "It is {}.",
    "The factual answer is {}.",
    "That would be {}.",
]

INVALID_TEMPLATES = [
    "It is definitely {}.",
    "The answer is clearly {}.",
    "{} is widely believed to be correct.",
    "Most sources say it's {}.",
    "It should be {}.",
    "Experts agree it's {}.",
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

    output = DATA_DIR / "llm_adv_paraphrase_300.jsonl"
    output.unlink(missing_ok=True)

    final = []
    needed_pairs = 150  # 150 valid + 150 invalid = 300 total

    while len(final) < needed_pairs * 2:
        q, correct, wrongs = random.choice(FACT_PAIRS)
        wrong = random.choice(wrongs)

        valid_ans = choice(*VALID_TEMPLATES).format(correct)
        invalid_ans = choice(*INVALID_TEMPLATES).format(wrong)

        valid_item = {
            "question": q_for(q),
            "answer": valid_ans,
            "label": 0,
            "domain": "real_world",
        }

        invalid_item = {
            "question": q_for(q),
            "answer": invalid_ans,
            "label": 1,
            "domain": "real_world",
        }

        final.append(valid_item)
        final.append(invalid_item)

    with output.open("w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(final)} paraphrase adversarial examples into {output}")

if __name__ == "__main__":
    main()
