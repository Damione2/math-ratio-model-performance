#!/usr/bin/env python3
"""
generate_adv_paraphrase_300_v2.py (Refined)

Expanded paraphrase / RAG misinformation adversarial generator (v2).
Generates a larger, more diverse set of paraphrase factual QA examples
(valid + adversarial paraphrases) and writes them to llm_adv_paraphrase_300_v2.jsonl.

Refined Goals:
- Increase TARGET_TOTAL to 600.
- Enhance hedging templates to target false negatives (hedged blatant facts).
- Expand fact base across many domains.
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

OUTPUT = DATA_DIR / "llm_adv_paraphrase_300_v2.jsonl"
TARGET_TOTAL = 1000   # Increased target for dedupe overshoot
SEED = 20260109

random.seed(SEED)

# -----------------------------
# Utilities
# -----------------------------
def q_for(question: str) -> str:
    return f"Answer the following question: {question}"

def safe_write(items: List[dict]):
    OUTPUT.unlink(missing_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} examples into {OUTPUT}")

# -----------------------------
# Expanded fact base
# -----------------------------
FACTS: List[Tuple[str, str, List[str]]] = [
    # Geography / Capitals
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver", "Montreal"]),
    ("What is the capital of Brazil?", "Brasília", ["Rio de Janeiro", "São Paulo", "Salvador"]),
    ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza", "Luxor"]),
    ("What is the capital of Switzerland?", "Bern", ["Zurich", "Geneva", "Basel"]),
    ("What is the capital of Turkey?", "Ankara", ["Istanbul", "Izmir", "Bursa"]),
    ("What is the capital of Japan?", "Tokyo", ["Kyoto", "Osaka", "Nagoya"]),
    ("Where is London located?", "United Kingdom", ["Japan", "France", "Germany"]),

    # History / Dates / People
    ("When did World War II end?", "1945", ["1944", "1950", "1939"]),
    ("Who was the first President of the United States?", "George Washington", ["Thomas Jefferson", "John Adams", "Abraham Lincoln"]),
    ("When was the Declaration of Independence signed?", "1776", ["1775", "1781", "1783"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Robert Koch"]),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Donatello"]),

    # Science
    ("What planet is known as the Red Planet?", "Mars", ["Jupiter", "Venus", "Mercury"]),
    ("What gas do plants absorb during photosynthesis?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Hydrogen"]),
    ("What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go"]),
    ("What is the largest planet in our solar system?", "Jupiter", ["Saturn", "Neptune", "Earth"]),
    ("What is the boiling point of water at sea level in Celsius?", "100", ["90", "212", "80"]),

    # Literature / Authors
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Charlotte Brontë", "Emily Brontë", "Charles Dickens"]),
    ("Who wrote '1984'?", "George Orwell", ["Aldous Huxley", "Ray Bradbury", "J.R.R. Tolkien"]),
    ("Who wrote 'The Odyssey'?", "Homer", ["Virgil", "Sophocles", "Euripides"]),

    # Biology / Medicine
    ("What is the largest mammal?", "Blue whale", ["African elephant", "Giraffe", "Hippopotamus"]),
    ("What is the fastest land animal?", "Cheetah", ["Lion", "Pronghorn", "Greyhound"]),
    ("Which organ pumps blood through the body?", "Heart", ["Lungs", "Liver", "Kidney"]),

    # Misc / Culture
    ("What is the primary language spoken in Brazil?", "Portuguese", ["Spanish", "English", "French"]),
    ("What is the currency of Japan?", "Yen", ["Dollar", "Euro", "Won"]),
    ("What is the largest continent by area?", "Asia", ["Africa", "North America", "Antarctica"]),
]

# -----------------------------
# Template libraries
# -----------------------------
VALID_TEMPLATES = [
    "The correct answer is {}.",
    "{} is the correct answer.",
    "It is {}.",
    "The factual answer is {}.",
    "According to reliable sources, the answer is {}.",
    "Verified sources indicate {}.",
    "The widely accepted answer is {}.",
]

# TARGETING WEAK MODES: Hedged blatant facts
INVALID_TEMPLATES = [
    "It is definitely {}.",
    "The answer is clearly {}.",
    "I believe the answer is {}.",
    "From what I recall, it is {}.",
    "I'm pretty sure it's {}.",
    "It seems like {} is the answer.",
    "I think the answer might be {}.",
    "Most sources suggest it's {}.",
    "It is often reported as {}.",
    "Popular belief holds that it is {}.",
    "Some references list {} as the answer.",
]

FILLER_SENTENCES = [
    "This is a common question in general knowledge.",
    "Historical records usually confirm this detail.",
    "Context is important when answering this.",
    "This fact is frequently cited in educational materials.",
]

# -----------------------------
# Generation Logic
# -----------------------------
def paraphrase_template_valid(correct: str) -> str:
    template = random.choice(VALID_TEMPLATES)
    ans = template.format(correct)
    if random.random() < 0.3:
        ans += " " + random.choice(FILLER_SENTENCES)
    return ans

def paraphrase_template_invalid(wrong: str) -> str:
    template = random.choice(INVALID_TEMPLATES)
    # Target hedging specifically
    ans = template.format(wrong)
    if random.random() < 0.3:
        ans += " " + random.choice(FILLER_SENTENCES)
    return ans

def build_paraphrase_items(target_total: int) -> List[dict]:
    items: List[dict] = []
    seen = set()
    fact_pool = FACTS.copy()

    while len(items) < target_total:
        f = random.choice(fact_pool)
        q_text, correct, wrongs = f
        wrong = random.choice(wrongs)

        # Valid Item
        v_text = paraphrase_template_valid(correct)
        v_item = {"question": q_for(q_text), "answer": v_text, "label": 0, "domain": "real_world"}
        
        # Invalid Item
        i_text = paraphrase_template_invalid(wrong)
        i_item = {"question": q_for(q_text), "answer": i_text, "label": 1, "domain": "real_world"}

        for item in (v_item, i_item):
            key = (item["question"], item["answer"])
            if key not in seen and len(items) < target_total:
                seen.add(key)
                items.append(item)

    random.shuffle(items)
    return items

def main():
    items = build_paraphrase_items(TARGET_TOTAL)
    safe_write(items)

if __name__ == "__main__":
    main()