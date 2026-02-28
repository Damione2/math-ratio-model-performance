#!/usr/bin/env python3
"""
generate_adv_longcontext_200_v2.py

Expanded long-context contradiction adversarial generator (v2).
Generates a larger, more diverse set of long-paragraph (multi-paragraph)
passages where a single subtle incorrect fact is embedded (adversarial)
or where the passage is coherent and correct (valid). Writes output to
llm_adv_longcontext_200_v2.jsonl.

Design goals:
- Produce many more unique long-context items than the original generator.
- Use a broad fact base across many domains (history, science, geography,
  literature, technology, biology, astronomy, culture, economics).
- Create multi-paragraph passages (2-4 paragraphs) with varied tones:
  narrative, academic, conversational, analytical.
- Embed a single subtle incorrect fact for adversarial items, often in
  paragraph 2 or the final sentence.
- Add distractors, filler sentences, quotes, statistics, and blended facts
  to increase entropy and realism.
- Output is balanced valid/invalid and JSONL-ready.

Output file: llm_adv_longcontext_200_v2.jsonl
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

OUTPUT = DATA_DIR / "llm_adv_longcontext_200_v2.jsonl"
TARGET_VALID = 300
TARGET_INVALID = 300  # number of adversarial long-context examples
SEED = 20260109

random.seed(SEED)

# -----------------------------
# Utilities
# -----------------------------
def q_for(question: str) -> str:
    return f"Read the passage and answer the question: {question}"

def safe_write(items: List[dict]):
    OUTPUT.unlink(missing_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} examples into {OUTPUT}")

def choice(*xs):
    # Accept either choice(a, b, c) or choice(list_or_tuple)
    if len(xs) == 1 and isinstance(xs[0], (list, tuple)):
        seq = xs[0]
    else:
        seq = xs
    return random.choice(seq)
    

def maybe(text: str, p: float = 0.5) -> str:
    return text if random.random() < p else ""

# -----------------------------
# Expanded fact base
# Each entry: (question, correct_answer, [wrong_alternatives...])
# -----------------------------
FACTS: List[Tuple[str, str, List[str]]] = [
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Donatello"]),
    ("What is the capital of Australia?", "Canberra", ["Sydney", "Melbourne", "Perth"]),
    ("Which river is the longest in the world?", "Nile", ["Amazon", "Yangtze", "Mississippi"]),
    ("Who developed the theory of evolution by natural selection?", "Charles Darwin", ["Gregor Mendel", "Alfred Russel Wallace", "Jean-Baptiste Lamarck"]),
    ("What is the chemical symbol for gold?", "Au", ["Ag", "Gd", "Go"]),
    ("Which planet is known as the Red Planet?", "Mars", ["Jupiter", "Venus", "Mercury"]),
    ("Who wrote 'Pride and Prejudice'?", "Jane Austen", ["Charlotte Brontë", "Emily Brontë", "Charles Dickens"]),
    ("When did World War II end?", "1945", ["1944", "1950", "1939"]),
    ("Who discovered penicillin?", "Alexander Fleming", ["Louis Pasteur", "Marie Curie", "Robert Koch"]),
    ("What is the largest mammal?", "Blue whale", ["African elephant", "Giraffe", "Hippopotamus"]),
    ("What is the tallest mountain in the world?", "Mount Everest", ["K2", "Kangchenjunga", "Lhotse"]),
    ("What is the largest ocean?", "Pacific Ocean", ["Atlantic Ocean", "Indian Ocean", "Arctic Ocean"]),
    ("Who proposed the theory of relativity?", "Albert Einstein", ["Isaac Newton", "Niels Bohr", "Galileo Galilei"]),
    ("What is the boiling point of water at sea level in Celsius?", "100", ["90", "212", "80"]),
    ("What is the currency of Japan?", "Yen", ["Dollar", "Euro", "Won"]),
    ("Which element has atomic number 1?", "Hydrogen", ["Helium", "Oxygen", "Carbon"]),
    ("Who wrote '1984'?", "George Orwell", ["Aldous Huxley", "Ray Bradbury", "J.R.R. Tolkien"]),
    ("What is the capital of Canada?", "Ottawa", ["Toronto", "Vancouver", "Montreal"]),
    ("Who painted Starry Night?", "Vincent van Gogh", ["Claude Monet", "Pablo Picasso", "Edvard Munch"]),
    ("Which gas do plants absorb during photosynthesis?", "Carbon dioxide", ["Oxygen", "Nitrogen", "Hydrogen"]),
    ("What is the largest continent by area?", "Asia", ["Africa", "North America", "Antarctica"]),
    ("Which country is known as the Land of the Rising Sun?", "Japan", ["China", "South Korea", "Thailand"]),
    ("Who was the first President of the United States?", "George Washington", ["Thomas Jefferson", "John Adams", "Abraham Lincoln"]),
    ("What is the smallest prime number?", "2", ["1", "0", "3"]),
    ("Which scientist is credited with the laws of motion and universal gravitation?", "Isaac Newton", ["Albert Einstein", "Galileo Galilei", "Johannes Kepler"]),
    ("What is the capital of Turkey?", "Ankara", ["Istanbul", "Izmir", "Bursa"]),
    ("Who painted The Last Supper?", "Leonardo da Vinci", ["Michelangelo", "Raphael", "Titian"]),
    ("Which organ pumps blood through the body?", "Heart", ["Lungs", "Liver", "Kidney"]),
    ("What is the primary language spoken in Brazil?", "Portuguese", ["Spanish", "English", "French"]),
    ("Which mountain range contains Mount Everest?", "Himalayas", ["Andes", "Alps", "Rockies"]),
    ("What is the largest desert on Earth?", "Sahara", ["Gobi", "Arabian", "Kalahari"]),
    ("Who is known as the father of modern computing?", "Alan Turing", ["Charles Babbage", "John von Neumann", "Ada Lovelace"]),
    ("Which planet has the most moons in our solar system?", "Jupiter", ["Saturn", "Uranus", "Neptune"]),
    ("What is the capital of Egypt?", "Cairo", ["Alexandria", "Giza", "Luxor"]),
    ("Who wrote 'Hamlet'?", "William Shakespeare", ["Christopher Marlowe", "Ben Jonson", "John Donne"]),
    ("Which element is represented by the symbol 'O'?", "Oxygen", ["Gold", "Osmium", "Oganesson"]),
    ("What is the longest mountain range in the world?", "Andes", ["Himalayas", "Rockies", "Ural"]),
    ("Which city hosted the 2012 Summer Olympics?", "London", ["Beijing", "Rio de Janeiro", "Tokyo"]),
    ("What is the largest island in the world?", "Greenland", ["New Guinea", "Borneo", "Madagascar"]),
    ("Who discovered America (commonly credited)?", "Christopher Columbus", ["Leif Erikson", "Amerigo Vespucci", "Vasco da Gama"]),
    ("What is the primary ingredient in traditional hummus?", "Chickpeas", ["Lentils", "Beans", "Potatoes"]),
    ("Which scientist discovered the structure of DNA?", "James Watson and Francis Crick", ["Rosalind Franklin", "Linus Pauling", "Gregor Mendel"]),
    ("What is the capital of Russia?", "Moscow", ["Saint Petersburg", "Kazan", "Novosibirsk"]),
    ("Which metal is liquid at room temperature?", "Mercury", ["Gallium", "Bromine", "Cesium"]),
    ("What is the largest city by population in the United States?", "New York City", ["Los Angeles", "Chicago", "Houston"]),
    ("Which instrument measures atmospheric pressure?", "Barometer", ["Thermometer", "Hygrometer", "Anemometer"]),
    ("What is the chemical formula for water?", "H2O", ["HO2", "O2H", "H3O"]),
    ("Which famous scientist developed the laws of inheritance?", "Gregor Mendel", ["Charles Darwin", "Louis Pasteur", "Thomas Hunt Morgan"]),
    ("What is the capital of India?", "New Delhi", ["Mumbai", "Bangalore", "Kolkata"]),
    ("Which ocean lies between Africa and Australia?", "Indian Ocean", ["Pacific Ocean", "Atlantic Ocean", "Southern Ocean"]),
    ("Who is the author of 'The Hobbit'?", "J.R.R. Tolkien", ["C.S. Lewis", "J.K. Rowling", "George R.R. Martin"]),
    ("What is the largest organ in the human body?", "Skin", ["Liver", "Lungs", "Heart"]),
    ("Which gas makes up most of Earth's atmosphere?", "Nitrogen", ["Oxygen", "Carbon dioxide", "Argon"]),
    ("What is the capital of France?", "Paris", ["Lyon", "Marseille", "Nice"]),
]

# -----------------------------
# Passage templates and styles
# -----------------------------
PARA_OPENERS = [
    "Recent studies and historical records provide a useful overview.",
    "Scholars have long debated this topic, and the literature is rich with detail.",
    "In everyday discussions, people often reference a few well-known facts.",
    "A careful review of primary and secondary sources reveals several consistent points.",
    "Across textbooks and popular accounts, the subject is described in similar terms.",
]

PARA_MIDDLES = [
    "The evidence includes archival documents, experimental results, and eyewitness accounts.",
    "Statistical summaries and expert analyses tend to support the mainstream interpretation.",
    "Contemporary commentary often highlights the broader implications for society.",
    "Technical descriptions and lay summaries both emphasize the core facts.",
    "Multiple independent sources corroborate the central claim.",
]

PARA_FILLERS = [
    "It is worth noting that context and nuance matter when interpreting these details.",
    "Some sources provide additional background that clarifies the timeline.",
    "Experts caution against overgeneralizing from a single example.",
    "Secondary literature often synthesizes the primary evidence into a coherent narrative.",
    "Readers should consider both the direct evidence and the interpretive frameworks.",
]

PARA_CLOSERS = [
    "Taken together, the most reliable conclusion is {}.",
    "Therefore, the correct answer is {}.",
    "In summary, the evidence points to {}.",
    "Consequently, the accepted response is {}.",
    "Thus, the factual answer is {}.",
]

PARA_QUOTE_OPENERS = [
    "As one historian put it,",
    "A contemporary account reads,",
    "In a well-known quote,",
    "A primary source states,",
]

TONE_STYLES = [
    "academic",
    "narrative",
    "conversational",
    "analytical",
    "expository",
]

# -----------------------------
# Helpers to build passages
# -----------------------------
def build_paragraph(correct_fact: str, include_filler: bool = True, tone: str = "academic") -> str:
    opener = choice(PARA_OPENERS)
    middle = choice(PARA_MIDDLES)
    filler = choice(PARA_FILLERS) if include_filler and random.random() < 0.6 else ""
    closer_template = choice(PARA_CLOSERS)
    closer = closer_template.format(correct_fact)
    parts = [opener, middle]
    if filler:
        parts.append(filler)
    parts.append(closer)
    paragraph = " ".join(p for p in parts if p)
    # tone adjustments
    if tone == "conversational":
        paragraph = paragraph.replace("Consequently,", "So,").replace("Therefore,", "So,")
    elif tone == "narrative":
        paragraph = paragraph + " The narrative around this has evolved over time."
    elif tone == "analytical":
        paragraph = paragraph + " This conclusion follows from the data above."
    return paragraph

def build_paragraph_with_wrong(wrong_fact: str, distractor: str = "", tone: str = "academic") -> str:
    opener = choice(PARA_OPENERS)
    middle = choice(PARA_MIDDLES)
    filler = choice(PARA_FILLERS) if random.random() < 0.6 else ""
    # place the wrong fact subtly in the middle or final sentence
    if random.random() < 0.5:
        # embed wrong fact in middle
        middle = middle + " " + maybe("Notably, " + wrong_fact + ".", 0.9)
        closer = choice(["In light of this, the passage suggests {}.", "Thus, some conclude {}."]).format(wrong_fact)
    else:
        closer = choice(["In summary, the evidence points to {}.", "Therefore, the answer is {}."]).format(wrong_fact)
    parts = [opener, middle]
    if filler:
        parts.append(filler)
    parts.append(closer)
    paragraph = " ".join(p for p in parts if p)
    if tone == "conversational":
        paragraph = paragraph.replace("Therefore,", "So,")
    return paragraph

def maybe_quote() -> str:
    if random.random() < 0.25:
        return '"' + choice(
            "This account is widely cited.",
            "Contemporary observers recorded similar details.",
            "The primary source is explicit on this point."
        ) + '"'
    return ""

def assemble_passage(correct: str, wrong: str = None, paragraphs: int = 2, tone: str = "academic", include_quote: bool = False) -> str:
    paras = []
    # paragraph 1: context and setup (usually correct)
    paras.append(build_paragraph(correct, include_filler=True, tone=tone))
    # paragraph 2..n-1: details; if wrong is provided, embed it in one of these paragraphs
    for i in range(2, paragraphs + 1):
        if wrong and i == random.randint(2, paragraphs):
            # embed wrong fact here
            paras.append(build_paragraph_with_wrong(wrong, tone=tone))
        else:
            paras.append(build_paragraph(correct, include_filler=random.random() < 0.7, tone=tone))
    # optionally add a quote or statistic
    if include_quote and random.random() < 0.6:
        paras.insert(random.randint(1, len(paras)), maybe_quote())
    # join paragraphs with double newlines
    return "\n\n".join(paras)

# -----------------------------
# Main generation loop
# -----------------------------
def build_longcontext_items(target_valid: int, target_invalid: int) -> List[dict]:
    items: List[dict] = []
    seen = set()
    attempts = 0
    max_attempts = (target_valid + target_invalid) * 20

    # expand fact pool by duplicating with small variations to increase entropy
    fact_pool = FACTS.copy()

    while (len([x for x in items if x["label"] == 0]) < target_valid or
           len([x for x in items if x["label"] == 1]) < target_invalid) and attempts < max_attempts:
        attempts += 1
        # pick a base fact
        q, correct, wrongs = random.choice(fact_pool)
        wrong = random.choice(wrongs)

        # choose passage parameters
        paragraphs = random.choice([2, 2, 3, 3, 4])  # bias toward 2-3 paragraphs
        tone = random.choice(TONE_STYLES)
        include_quote = random.random() < 0.4

        # build valid passage
        valid_passage = assemble_passage(correct, wrong=None, paragraphs=paragraphs, tone=tone, include_quote=include_quote)
        # build invalid passage with wrong fact embedded
        invalid_passage = assemble_passage(correct, wrong=wrong, paragraphs=paragraphs, tone=tone, include_quote=include_quote)

        # sometimes create blended/multi-fact passages to increase difficulty
        if random.random() < 0.12:
            # pick a second fact and blend
            q2, correct2, wrongs2 = random.choice(fact_pool)
            if q2 != q:
                # append a paragraph referencing the second fact
                valid_passage = valid_passage + "\n\n" + build_paragraph(correct2, include_filler=True, tone=tone)
                invalid_passage = invalid_passage + "\n\n" + build_paragraph_with_wrong(random.choice(wrongs2), tone=tone)

                # update question to reflect multi-fact
                q_combined = f"{q} Also: {q2}"
                question_text = q_for(q_combined)
            else:
                question_text = q_for(q)
        else:
            question_text = q_for(q)

        # small textual variations to increase uniqueness
        if random.random() < 0.3:
            valid_passage = valid_passage.replace("The evidence", "The available evidence")
            invalid_passage = invalid_passage.replace("The evidence", "The available evidence")
        if random.random() < 0.2:
            valid_passage = valid_passage + "\n\n" + "Note: this summary is intended for general informational purposes."
            invalid_passage = invalid_passage + "\n\n" + "Note: this summary is intended for general informational purposes."

        # build items
        valid_item = {"question": question_text, "answer": valid_passage, "label": 0, "domain": "real_world"}
        invalid_item = {"question": question_text, "answer": invalid_passage, "label": 1, "domain": "real_world"}

        # dedupe and add
        for item in (valid_item, invalid_item):
            key = (item["question"], item["answer"])
            if key in seen:
                continue
            seen.add(key)
            items.append(item)

        # safety: break if we've exceeded attempts
        if attempts % 100 == 0 and attempts > 0:
            # small shuffle to vary selection
            random.shuffle(fact_pool)

    # ensure we have at least the requested counts (may be slightly over)
    valid_count = len([x for x in items if x["label"] == 0])
    invalid_count = len([x for x in items if x["label"] == 1])

    # If we are short on either side, try to generate more focused items
    extra_attempts = 0
    while (valid_count < target_valid or invalid_count < target_invalid) and extra_attempts < 1000:
        extra_attempts += 1
        q, correct, wrongs = random.choice(fact_pool)
        wrong = random.choice(wrongs)
        paragraphs = random.choice([2, 3, 4])
        tone = random.choice(TONE_STYLES)
        include_quote = random.random() < 0.4

        if valid_count < target_valid:
            vp = assemble_passage(correct, wrong=None, paragraphs=paragraphs, tone=tone, include_quote=include_quote)
            v_item = {"question": q_for(q), "answer": vp, "label": 0, "domain": "real_world"}
            key = (v_item["question"], v_item["answer"])
            if key not in seen:
                seen.add(key)
                items.append(v_item)
                valid_count += 1

        if invalid_count < target_invalid:
            ip = assemble_passage(correct, wrong=wrong, paragraphs=paragraphs, tone=tone, include_quote=include_quote)
            i_item = {"question": q_for(q), "answer": ip, "label": 1, "domain": "real_world"}
            key = (i_item["question"], i_item["answer"])
            if key not in seen:
                seen.add(key)
                items.append(i_item)
                invalid_count += 1

    # final shuffle
    random.shuffle(items)
    return items

# -----------------------------
# Main
# -----------------------------
def main():
    items = build_longcontext_items(TARGET_VALID, TARGET_INVALID)
    safe_write(items)

if __name__ == "__main__":
    main()