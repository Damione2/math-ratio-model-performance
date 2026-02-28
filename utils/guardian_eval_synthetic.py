# guardian_eval_synthetic.py
"""
Synthetic evaluation generators for Guardian.
These produce large-scale, automatically generated test sets.
"""

import random

def generate_math_synthetic(n=200):
    tests = []
    for _ in range(n):
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        correct = a + b
        wrong = correct + random.choice([-2, -1, 1, 2])

        tests.append({
            "category": "Math-Synthetic",
            "name": f"math_add_{a}_{b}",
            "question": f"What is {a} + {b}?",
            "answer": str(correct),
            "true_label": 0
        })
        tests.append({
            "category": "Math-Synthetic",
            "name": f"math_add_wrong_{a}_{b}",
            "question": f"What is {a} + {b}?",
            "answer": str(wrong),
            "true_label": 1
        })
    return tests


def generate_code_synthetic(n=200):
    tests = []
    for i in range(n):
        x = random.randint(2, 12)
        tests.append({
            "category": "Code-Synthetic",
            "name": f"code_square_correct_{i}",
            "question": "Write a Python function that returns x squared.",
            "answer": f"def square(x): return x * x",
            "true_label": 0
        })
        tests.append({
            "category": "Code-Synthetic",
            "name": f"code_square_wrong_{i}",
            "question": "Write a Python function that returns x squared.",
            "answer": f"def square(x): return x * x * x",
            "true_label": 1
        })
    return tests


def generate_realworld_synthetic(n=200):
    tests = []
    for i in range(n):
        tests.append({
            "category": "RealWorld-Synthetic",
            "name": f"rw_correct_{i}",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Jane Austen.",
            "true_label": 0
        })
        tests.append({
            "category": "RealWorld-Synthetic",
            "name": f"rw_wrong_{i}",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Charles Dickens.",
            "true_label": 1
        })
    return tests


def get_synthetic_tests():
    return (
        generate_math_synthetic(100) +
        generate_code_synthetic(100) +
        generate_realworld_synthetic(100)
    )
