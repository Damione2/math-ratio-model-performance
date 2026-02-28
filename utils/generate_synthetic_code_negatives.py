#!/usr/bin/env python3
# utils/generate_synthetic_code_negatives.py
"""
Generate 1,000 synthetic code-negative JSONL examples for mining/training.
Each line is a JSON object with keys: question, answer, label (1 = hallucination), domain, meta.
"""

import json
import random
from pathlib import Path

OUT = Path("data/synthetic_code_negatives.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Templates for prompts (question) and correct answers (for context)
PROMPTS = [
    "Write a Python list comprehension for squares of 0..9.",
    "Write a Python function that returns x squared.",
    "Open a file and read all lines in Python.",
    "Sort a list of integers in ascending order in Python.",
    "Write a Python loop that prints numbers 0..9.",
    "Return the sum of two numbers in Python.",
    "Create a Python dictionary mapping names to ages.",
    "Slice a list to get the first 5 elements in Python.",
    "Check if a number is even in Python.",
    "Concatenate two strings in Python."
]

# Perturbation generators
def operator_swap(ans: str) -> str:
    # swap ** <-> ^, + <-> -, * <-> /, == <-> =
    swaps = [("**", "^"), ("^", "**"), (" + ", " - "), (" - ", " + "),
             (" * ", " / "), (" / ", " * "), (" == ", " = ")]
    s = ans
    a, b = random.choice(swaps)
    return s.replace(a, b)

def missing_call(ans: str) -> str:
    # remove a common method call (e.g., .read(), .strip(), .split())
    calls = [".read()", ".strip()", ".split()", ".lower()", ".upper()"]
    for c in calls:
        if c in ans:
            return ans.replace(c, "")
    # fallback: remove parentheses from a function call
    return ans.replace("()", "")

def syntax_error(ans: str) -> str:
    # remove colon, remove comma, or break indentation
    choices = [
        lambda s: s.replace(":", "", 1),
        lambda s: s.replace(",", "", 1),
        lambda s: s.replace("    ", "", 1),
        lambda s: s.replace(")", "", 1),
    ]
    return random.choice(choices)(ans)

def wrong_return(ans: str) -> str:
    # change return value to an incorrect one
    if "return" in ans:
        return ans.replace("return", "return 0 # WRONG", 1)
    return ans + "\n# WRONG RETURN"

def var_misuse(ans: str) -> str:
    # rename variable names incorrectly
    return ans.replace("x", "y", 1)

def malformed_comprehension(ans: str) -> str:
    # produce a comprehension with syntax issues
    return ans.replace("for", "for in", 1) if "for" in ans else ans + " [x for in range(10)]"

PERTURBATIONS = [
    operator_swap,
    missing_call,
    syntax_error,
    wrong_return,
    var_misuse,
    malformed_comprehension
]

# Small set of canonical correct answers to perturb
CANONICAL_ANSWERS = {
    "Write a Python list comprehension for squares of 0..9.": "[x**2 for x in range(10)]",
    "Write a Python function that returns x squared.": "def square(x):\n    return x * x",
    "Open a file and read all lines in Python.": "with open('file.txt','r') as f:\n    lines = f.read().splitlines()",
    "Sort a list of integers in ascending order in Python.": "sorted_list = sorted(my_list)",
    "Write a Python loop that prints numbers 0..9.": "for i in range(10):\n    print(i)",
    "Return the sum of two numbers in Python.": "def add(a,b):\n    return a + b",
    "Create a Python dictionary mapping names to ages.": "d = {'alice': 30, 'bob': 25}",
    "Slice a list to get the first 5 elements in Python.": "first_five = a[:5]",
    "Check if a number is even in Python.": "def is_even(n):\n    return n % 2 == 0",
    "Concatenate two strings in Python.": "s = a + b"
}

def make_negative(prompt: str) -> str:
    base = CANONICAL_ANSWERS.get(prompt, "pass")
    # apply 1-2 perturbations
    p = random.choice(PERTURBATIONS)
    ans = p(base)
    if random.random() < 0.35:
        q = random.choice(PERTURBATIONS)
        ans = q(ans)
    # add a short comment indicating it's intentionally wrong (keeps context)
    if random.random() < 0.2:
        ans += "\n# intentionally incorrect example"
    return ans

def generate(n=1000):
    out = []
    for i in range(n):
        prompt = random.choice(PROMPTS)
        answer = make_negative(prompt)
        item = {
            "question": prompt,
            "answer": answer,
            "label": 1,
            "domain": "code",
            "meta": {"source": "synthetic_miner", "id": f"syn_{i:04d}"}
        }
        out.append(item)
    return out

def main():
    items = generate(1000)
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} synthetic negatives to {OUT}")

if __name__ == "__main__":
    main()
