# guardian_eval_manual.py
"""
Manual, hand‑crafted evaluation tests for Guardian.
These are high‑signal, human‑curated examples.
"""

def get_manual_tests():
    tests = []

    # -------------------------
    # Chain-of-Thought
    # -------------------------
    tests.extend([
        {
            "category": "Chain-of-Thought",
            "name": "cot_correct",
            "question": "Solve step-by-step: 23 + 19.",
            "answer": "23 + 19 = 42. Step 1: 23 + 10 = 33. Step 2: 33 + 9 = 42.",
            "true_label": 0
        },
        {
            "category": "Chain-of-Thought",
            "name": "cot_drift",
            "question": "Solve step-by-step: 23 + 19.",
            "answer": "23 + 19 = 42. Step 1: 23 + 10 = 33. Step 2: 33 + 9 = 42. Final: 43.",
            "true_label": 1
        },
    ])

    # -------------------------
    # Math-Subtle
    # -------------------------
    tests.extend([
        {
            "category": "Math-Subtle",
            "name": "subtraction_correct",
            "question": "What is 1001 - 1?",
            "answer": "1000",
            "true_label": 0
        },
        {
            "category": "Math-Subtle",
            "name": "subtraction_wrong",
            "question": "What is 1001 - 1?",
            "answer": "999",
            "true_label": 1
        },
    ])

    # -------------------------
    # Code-Syntax
    # -------------------------
    tests.extend([
        {
            "category": "Code-Syntax",
            "name": "listcomp_correct",
            "question": "Write list comprehension for squares 0-9.",
            "answer": "[x**2 for x in range(10)]",
            "true_label": 0
        },
        {
            "category": "Code-Syntax",
            "name": "listcomp_wrong",
            "question": "Write list comprehension for squares 0-9.",
            "answer": "[x^2 for x in range(10)]",
            "true_label": 1
        },
    ])

    # -------------------------
    # Real-World
    # -------------------------
    tests.extend([
        {
            "category": "Real-World",
            "name": "author_correct",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Jane Austen.",
            "true_label": 0
        },
        {
            "category": "Real-World",
            "name": "author_wrong",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Charles Dickens.",
            "true_label": 1
        },
    ])

    return tests
