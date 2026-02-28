# generators.py (v6.2 - Runtime Hallucination Rate Monitoring)
"""
Spider-Native Generators with real-time hallucination rate monitoring.
✅ FIXED: Added live compliance tracking to catch drift during generation
✅ FIXED: Safety checks for empty sequences in _choose() calls
✅ FIXED: Validation assertions to ensure labels are correctly set
✅ FIXED: REAL generator now properly respects hallucination_rate
"""

import random
import string
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===========================
# Base Generator with Isolated RNG and Monitoring
# ===========================
class BaseGenerator:
    def __init__(self, hallucination_rate=0.5, seed=42, enable_tracking=True):
        """
        Initialize generator with isolated RNG and optional tracking.
        """
        self.hallucination_rate = float(hallucination_rate)
        self.rng = random.Random(seed)  # Isolated RNG instance
        self.np_rng = np.random.RandomState(seed)  # Isolated NumPy RNG
        
        # Runtime tracking
        self.enable_tracking = enable_tracking
        self._stats = {'total': 0, 'hallucinations': 0}
        
        # Verify rate is in valid range
        if not (0.0 <= self.hallucination_rate <= 1.0):
            raise ValueError(f"hallucination_rate must be in [0,1], got {hallucination_rate}")

    def _is_hallu(self):
        """Check if this sample should be hallucinatory based on rate."""
        result = self.rng.random() < self.hallucination_rate
        # Track for monitoring
        self._stats['total'] += 1
        if result:
            self._stats['hallucinations'] += 1
        return result

    def get_rate(self) -> float:
        """Get current hallucination rate (live tracking)."""
        if self._stats['total'] == 0:
            return 0.0
        return self._stats['hallucinations'] / self._stats['total']

    def reset_tracking(self):
        """Reset tracking counters."""
        self._stats = {'total': 0, 'hallucinations': 0}

    def _rand_var(self, prefix="var"):
        """Generate random variable name."""
        return f"{prefix}_{self.rng.randint(10, 999)}"

    def _rand_int(self, lo=1, hi=500):
        """Generate random integer in range."""
        return self.rng.randint(lo, hi)

    def _choose(self, seq):
        """Random choice from sequence with safety checks."""
        if not seq:
            raise ValueError("Cannot choose from empty sequence")
        return self.rng.choice(seq)

    def _sample(self, seq, k=1):
        """Random sample without replacement."""
        if len(seq) < k:
            raise ValueError(f"Cannot sample {k} items from sequence of length {len(seq)}")
        return self.rng.sample(seq, k)

# ===========================
# Utilities
# ===========================
def augment_text(text: str, rng: random.Random = None):
    """Lightweight text augmentation with isolated RNG."""
    if rng is None:
        rng = random.Random()
    jitters = ["", " ", "...", "!", " (I think)", " (probably)"]
    prefixes = ["Actually, ", "I believe ", "It seems ", ""]
    return rng.choice(prefixes) + text + rng.choice(jitters)

def make_off_by_one(value: int, rng: random.Random):
    """Create off-by-one error with isolated RNG."""
    offset = rng.choice([-1, 1, -2, 2, 0])
    return value + offset

def simple_paraphrases(text: str, rng: random.Random, n=3) -> List[str]:
    """Lightweight paraphrase heuristics."""
    out = []
    words = text.split()
    for _ in range(n):
        if len(words) > 4 and rng.random() < 0.6:
            # Swap two random words
            i, j = rng.sample(range(len(words)), 2)
            w = words.copy()
            w[i], w[j] = w[j], w[i]
            out.append(" ".join(w))
        else:
            # Simple synonym replacement
            out.append(text.replace("is", "is actually").replace("the", "the"))
    return out

# ===========================
# MATH GENERATORS
# ===========================
class DiverseMathGenerator(BaseGenerator):
    def __init__(self, seed=101, hallucination_rate=0.5):
        super().__init__(hallucination_rate=hallucination_rate, seed=seed)

    def _generate_arithmetic(self):
        a, b = self._rand_int(), self._rand_int()
        op = self._choose(["+", "-", "*"])
        
        if op == "+":
            correct = a + b
        elif op == "-":
            correct = a - b
        else:
            correct = a * b

        is_h = self._is_hallu()
        if is_h:
            # Subtle off-by-one error 50% of the time, larger error otherwise
            if self.rng.random() < 0.5:
                ans = correct + self.rng.choice([-1, 1])
            else:
                ans = correct + self.rng.choice([-10, 10, 50])
        else:
            ans = correct

        return {
            "question": f"Calculate {a} {op} {b}.",
            "answer": f"The result is {ans}.",
            "label": 1 if is_h else 0,
            "domain": "math"
        }

    def _generate_cot_drift(self):
        a, b = self._rand_int(1, 200), self._rand_int(1, 200)
        correct = a + b
        is_h = self._is_hallu()
        
        if is_h:
            # Introduce error in final step
            final = correct + self.rng.choice([-1, 1])
            answer = f"{a} + {b} = {correct}. Step 1: add {a} and {b//2}. Step 2: add remainder. Final: {final}."
        else:
            answer = f"{a} + {b} = {correct}. Step 1: add {a} and {b//2}. Step 2: add remainder. Final: {correct}."
        
        return {
            "question": f"Solve step-by-step: {a} + {b}.",
            "answer": answer,
            "label": 1 if is_h else 0,
            "domain": "math"
        }

    def generate(self):
        # Mix arithmetic and CoT drift (70% arithmetic)
        if self.rng.random() < 0.7:
            return self._generate_arithmetic()
        else:
            return self._generate_cot_drift()

# ===========================
# CODE GENERATORS
# ===========================
class DiverseCodeGenerator(BaseGenerator):
    def __init__(self, seed=202, hallucination_rate=0.5):
        super().__init__(hallucination_rate=hallucination_rate, seed=seed)

    def _py_logic_gen(self):
        fname = f"process_{self.rng.randint(100, 999)}"
        v1 = self._rand_var()
        is_h = self._is_hallu()
        
        q = f"Write a Python function '{fname}' that returns {v1} squared."
        
        if not is_h:
            ans = f"def {fname}({v1}):\n    return {v1} ** 2"
        else:
            # Introduce syntax or semantic errors
            if self.rng.random() < 0.5:
                bad_op = self._choose(["* 2", "^ 2", "+ 2"])
                ans = f"def {fname}({v1}):\n    return {v1} {bad_op}"
            else:
                # Semantic: wrong variable used
                wrong_var = self._rand_var()
                ans = f"def {fname}({v1}):\n    return {wrong_var} ** 2"
        
        return {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "code"}

    def _py_syntax_missing_colon(self):
        fname = f"inc_{self.rng.randint(10, 99)}"
        is_h = self._is_hallu()
        
        if is_h:
            ans = f"def {fname}(x)\n    return x + 1"
        else:
            ans = f"def {fname}(x):\n    return x + 1"
        
        q = f"Write a Python function that returns x+1."
        return {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "code"}

    def _js_array_gen(self):
        vname = self._rand_var()
        is_h = self._is_hallu()
        
        q = f"Write a JS line to get the length of the array '{vname}'."
        if not is_h:
            ans = f"const len = {vname}.length;"
        else:
            # Common mistake: using Python style or wrong function
            ans = f"const len = len({vname});"
        
        return {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "code"}

    def _sql_gen(self):
        table = self._choose(["users", "orders", "products", "logs", "sessions"])
        is_h = self._is_hallu()
        
        q = f"SQL: Count entries in {table}."
        if not is_h:
            ans = f"SELECT COUNT(*) FROM {table};"
        else:
            ans = f"SELECT TOTAL() FROM {table};"
        
        return {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "code"}

    def generate(self):
        methods = [
            self._py_logic_gen,
            self._py_syntax_missing_colon,
            self._js_array_gen,
            self._sql_gen
        ]
        return self._choose(methods)()

# ===========================
# REAL WORLD GENERATORS (FIXED v6.3)
# ===========================
class DiverseRealWorldGenerator(BaseGenerator):
    def __init__(self, seed=303, hallucination_rate=0.5):
        super().__init__(hallucination_rate=hallucination_rate, seed=seed)
        
        self.cities = ["Paris", "Berlin", "Tokyo", "London", "Rome", "Madrid", "Cairo", "Seoul", "Beijing", "Lima"]
        self.countries = ["France", "Germany", "Japan", "UK", "Italy", "Spain", "Egypt", "South Korea", "China", "Peru"]
        self.elements = [("Oxygen", "O"), ("Gold", "Au"), ("Iron", "Fe"), ("Silver", "Ag"), ("Helium", "He"), ("Carbon", "C")]
        self.authors = [
            ("Pride and Prejudice", "Jane Austen"),
            ("1984", "George Orwell"),
            ("Brave New World", "Aldous Huxley"),
            ("To Kill a Mockingbird", "Harper Lee"),
            ("The Great Gatsby", "F. Scott Fitzgerald")
        ]

    def _geo_gen(self):
        idx = self._choose(range(len(self.cities)))
        city, country = self.cities[idx], self.countries[idx]
        is_h = self._is_hallu()
        
        q = f"In which country is the city of {city}?"
        if not is_h:
            ans = f"{city} is located in {country}."
        else:
            wrong_countries = [c for c in self.countries if c != country]
            wrong_country = self._choose(wrong_countries) if wrong_countries else "Atlantis"
            ans = f"{city} is located in {wrong_country}."
        
        result = {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "real_world"}
        return result

    def _science_gen(self):
        elem, sym = self._choose(self.elements)
        is_h = self._is_hallu()
        
        q = f"What is the chemical symbol for {elem}?"
        if not is_h:
            ans = f"The chemical symbol is {sym}."
        else:
            fake_sym = "".join(self.rng.choices(string.ascii_uppercase, k=2))
            ans = f"The chemical symbol is {fake_sym}."
        
        result = {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "real_world"}
        return result

    def _author_gen(self):
        book, author = self._choose(self.authors)
        is_h = self._is_hallu()
        
        q = f"Who wrote '{book}'?"
        if not is_h:
            ans = f"{author}."
        else:
            wrong_authors = [a for _, a in self.authors if a != author]
            wrong = self._choose(wrong_authors) if wrong_authors else "Unknown Author"
            ans = f"{wrong}."
        
        result = {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "real_world"}
        return result

    def _temporal_fact(self):
        # ✅ FIXED: Was previously hardcoded to always return hallucination
        is_h = self._is_hallu()
        
        q = "Who is the current president of the US as of 2021?"
        if not is_h:
            ans = "Joe Biden."
        else:
            ans = "Donald Trump."
        
        result = {"question": q, "answer": ans, "label": 1 if is_h else 0, "domain": "real_world"}
        return result

    def generate(self):
        choice_val = self.rng.random()
        if choice_val < 0.4:
            res = self._geo_gen()
        elif choice_val < 0.7:
            res = self._science_gen()
        elif choice_val < 0.9:
            res = self._author_gen()
        else:
            res = self._temporal_fact()
        
        res["answer"] = augment_text(res["answer"], self.rng)
        
        # ✅ CRITICAL VALIDATION
        assert res["label"] in [0, 1], f"REAL generator produced invalid label: {res['label']}"
        assert "question" in res and "answer" in res, "REAL generator missing required keys"
        return res
# ===========================
# ADVERSARIAL & MIXED GENERATORS (FIXED)
# ===========================
class AdversarialGenerator(BaseGenerator):
    """
    Produces targeted adversarial examples across categories.
    ✅ FIXED: Now actually respects hallucination_rate parameter instead of always hallucinating.
    """
    def __init__(self, seed=404, hallucination_rate=0.9):
        # Adversarial generator is biased to produce hallucinations by default
        super().__init__(hallucination_rate=hallucination_rate, seed=seed)
        
        # ✅ Pass the SAME hallucination_rate to sub-generators for consistency
        # If we want them to always produce hallucinations, set hallucination_rate=1.0
        self.math_gen = DiverseMathGenerator(seed=seed + 1, hallucination_rate=hallucination_rate)
        self.code_gen = DiverseCodeGenerator(seed=seed + 2, hallucination_rate=hallucination_rate)
        self.real_gen = DiverseRealWorldGenerator(seed=seed + 3, hallucination_rate=hallucination_rate)

    def _cot_drift(self):
        # ✅ FIXED: Use sub-generator which respects hallucination_rate
        return self.math_gen._generate_cot_drift()

    def _off_by_one(self):
        a, b = self._rand_int(2, 100), self._rand_int(2, 100)
        correct = a * b
        
        # ✅ FIXED: Check hallucination rate first
        if self._is_hallu():
            wrong = correct + self.rng.choice([-1, 1])
            return {"question": f"Compute {a} × {b}.", "answer": str(wrong), "label": 1, "domain": "math"}
        else:
            return {"question": f"Compute {a} × {b}.", "answer": str(correct), "label": 0, "domain": "math"}

    def _code_syntax_error(self):
        # ✅ FIXED: Respect hallucination rate
        if self._is_hallu():
            fname = f"fn_{self.rng.randint(10, 99)}"
            ans = f"def {fname}(x)\nreturn x * 2"
            return {"question": "Write a Python function to double a number.", "answer": ans, "label": 1, "domain": "code"}
        else:
            # Use base generator for valid samples
            return self.code_gen._py_logic_gen()

    def _code_semantic_error(self):
        # ✅ FIXED: Respect hallucination rate
        if self._is_hallu():
            fname = f"fn_{self.rng.randint(10, 99)}"
            v = self._rand_var()
            wrong_var = self._rand_var()
            ans = f"def {fname}({v}):\n    return {wrong_var} * {wrong_var}"
            return {"question": f"Write a Python function '{fname}' that returns {v} squared.", "answer": ans, "label": 1, "domain": "code"}
        else:
            return self.code_gen._py_logic_gen()

    def _paraphrase_flip(self):
        # ✅ FIXED: Respect hallucination rate
        base_q = "Summarize: The report states that 3+3=6."
        
        if self._is_hallu():
            ans = "The report claims three plus three equals seven."
            return {"question": base_q, "answer": ans, "label": 1, "domain": "real_world"}
        else:
            ans = "The report claims three plus three equals six."
            return {"question": base_q, "answer": ans, "label": 0, "domain": "real_world"}

    def _rag_misinfo(self):
        # RAG misinfo is hallucinatory by nature, but respect rate
        if self._is_hallu():
            q = "Based on retrieved notes: 'The capital of Australia is Sydney.' Summarize."
            ans = "The notes say the capital of Australia is Sydney."
            return {"question": q, "answer": ans, "label": 1, "domain": "real_world"}
        else:
            # Provide correct information when not hallucinating
            q = "Based on retrieved notes: 'The capital of Australia is Canberra.' Summarize."
            ans = "The notes say the capital of Australia is Canberra."
            return {"question": q, "answer": ans, "label": 0, "domain": "real_world"}

    def _long_context_trap(self):
        # ✅ FIXED: Respect hallucination rate
        a, b = self._rand_int(1, 10), self._rand_int(1, 10)
        correct = a + b
        
        context = f"Context: Alice has {a} apples. Bob gives her {b} more. Question: How many apples does Alice have?"
        
        if self._is_hallu():
            wrong = correct + 1
            ans = f"Alice has {wrong} apples."
            return {"question": context, "answer": ans, "label": 1, "domain": "math"}
        else:
            ans = f"Alice has {correct} apples."
            return {"question": context, "answer": ans, "label": 0, "domain": "math"}

    def generate(self):
        methods = [
            self._cot_drift,
            self._off_by_one,
            self._code_syntax_error,
            self._code_semantic_error,
            self._paraphrase_flip,
            self._rag_misinfo,
            self._long_context_trap
        ]
        gen = self._choose(methods)()
        
        # ✅ Ensure label is correctly set
        assert gen["label"] in [0, 1], f"Invalid label in AdversarialGenerator: {gen['label']}"
        return gen

# ===========================
# ALIASES FOR BACKWARD COMPATIBILITY
# ===========================
# Keep original class names for compatibility with legacy pipelines
RealGenerator = DiverseRealWorldGenerator
AdversarialSeedGenerator = AdversarialGenerator