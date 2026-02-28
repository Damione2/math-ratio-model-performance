# generators/math_long_cot_generator.py (v3.0 - 384-token CoT)
"""
Generates long-chain-of-thought math examples with subtle hallucinations.
Focuses on: algebra, multi-step arithmetic, word problems, exponents.
"""

import random
import sympy as sp
from typing import Dict, Any, List

class LongCoTMathGenerator:
    def __init__(self, seed=42, hallucination_rate=0.5):
        self.rng = random.Random(seed + 5001)
        self.hallucination_rate = hallucination_rate
        
        # Templates that force 15-30 reasoning steps
        self.cot_templates = [
            self._generate_polynomial_cot,
            self._generate_system_of_equations_cot,
            self._generate_compound_interest_cot,
            self._generate_geometric_series_cot,
            self._generate_rate_problem_cot
        ]
    
    def generate(self) -> Dict[str, Any]:
        template = self.rng.choice(self.cot_templates)
        return template()
    
    def _polynomial_format(self, expr, steps=12):
        """Format polynomial with detailed factorization steps"""
        x = sp.symbols('x')
        # Force sympy to show steps
        steps_list = []
        expanded = sp.expand(expr)
        steps_list.append(f"Expanding: {expr} → {expanded}")
        
        factored = sp.factor(expanded)
        steps_list.append(f"Factorization: {expanded} → {factored}")
        
        roots = sp.solve(factored, x)
        steps_list.append(f"Solving for roots: {factored} = 0 → x = {roots}")
        
        # Pad to required steps
        while len(steps_list) < steps:
            steps_list.append(f"Simplify: {expanded} = {expanded}")
        
        return " → ".join(steps_list[:steps])
    
    def _generate_polynomial_cot(self):
        """Long polynomial factorization with subtle sign error"""
        x = sp.symbols('x')
        # Real answer: (x-3)(x+5) = x² + 2x - 15
        real_poly = (x - 3)*(x + 5)
        
        # Hallucinated version (subtle sign error): (x+3)(x+5) = x² + 8x + 15
        hallu_poly = (x + 3)*(x + 5)
        
        is_hallu = self.rng.random() < self.hallucination_rate
        
        if not is_hallu:
            answer = self._polynomial_format(real_poly, steps=18)
            label = 0
        else:
            answer = self._polynomial_format(hallu_poly, steps=18)
            # Make it look correct at first glance
            answer += f" → Final answer: x = 3, -5"  # Wrong roots for hallu_poly
            label = 1
        
        return {
            "question": "Factor the polynomial (x-3)(x+5) and find its roots. Show detailed steps.",
            "answer": answer,
            "label": label,
            "domain": "math"
        }
    
    def _generate_system_of_equations_cot(self):
        """2x2 system with subtle coefficient error in step 7"""
        is_hallu = self.rng.random() < self.hallucination_rate
        
        # Real: 2x + 3y = 12, 4x - y = 5 → solution (x=1.5, y=3)
        # Hallu: 2x + 3y = 12, 4x - y = 6 → solution (x=1.8, y=2.8) but reports (1.5, 3)
        
        steps = [
            "Given: 2x + 3y = 12  (eq1)",
            "Given: 4x - y = 5    (eq2)" if not is_hallu else "Given: 4x - y = 6    (eq2)",  # Subtle error
            "From eq2: y = 4x - 5" if not is_hallu else "From eq2: y = 4x - 6",
            # ... 12 more steps of substitution ...
            "Substitute y into eq1: 2x + 3(4x-5) = 12" if not is_hallu else "Substitute: 2x + 3(4x-6) = 12",
            "Expand: 2x + 12x - 15 = 12" if not is_hallu else "Expand: 2x + 12x - 18 = 12",
            "Combine: 14x = 27" if not is_hallu else "Combine: 14x = 30",
            "x = 27/14 ≈ 1.93" if not is_hallu else "x = 30/14 ≈ 2.14",
            "Wait, let's recheck: x = 1.5" if is_hallu else "x = 1.5",  # Hallucination: ignores actual math
            "Then y = 4(1.5) - 5 = 3" if not is_hallu else "Then y = 4(1.5) - 6 = 0",  # More errors
            "Final answer: (x=1.5, y=3)"  # Always same final answer
        ]
        
        # Pad to 18-20 steps
        while len(steps) < 20:
            steps.append(f"Verify: substitute back into equation")
        
        answer = " → ".join(steps[:20])
        
        return {
            "question": "Solve the system: 2x + 3y = 12 and 4x - y = 5. Show your work.",
            "answer": answer,
            "label": 0 if not is_hallu else 1,
            "domain": "math"
        }
    
    def _generate_compound_interest_cot(self):
        """Financial math with exponent error"""
        principal = self.rng.randint(1000, 10000)
        rate = self.rng.uniform(0.03, 0.08)
        time = self.rng.randint(5, 30)
        
        is_hallu = self.rng.random() < self.hallucination_rate
        
        # A = P(1+r)^t
        real_amount = principal * (1 + rate)**time
        
        # Hallucination: use simple interest A = P(1 + rt) instead of compound
        hallu_amount = principal * (1 + rate * time)
        
        steps = [
            f"Compound interest formula: A = P(1+r)^t",
            f"P = ${principal}, r = {rate:.3f}, t = {time} years",
            f"A = {principal}(1 + {rate:.3f})^{time}",
            f"A = {principal}({1+rate:.5f})^{time}" if not is_hallu else f"A = {principal}(1 + {rate*time:.3f})",
            f"Calculate exponent: (1+{rate})^{time} = {(1+rate)**time:.2f}" if not is_hallu else f"Simple interest: {principal} × {1+rate*time:.3f} = {hallu_amount:.2f}",
            f"Final amount: ${real_amount:,.2f}" if not is_hallu else f"Final amount: ${hallu_amount:,.2f}",
            f"Interest earned: ${real_amount-principal:,.2f}" if not is_hallu else f"Interest earned: ${hallu_amount-principal:,.2f}"
        ]
        
        # Pad to 15-18 steps
        while len(steps) < 16:
            steps.append(f"Check: is this reasonable?")
        
        return {
            "question": f"Calculate the compound interest on ${principal} at {rate*100:.1f}% for {time} years.",
            "answer": " → ".join(steps),
            "label": 0 if not is_hallu else 1,
            "domain": "math"
        }
    
    def _generate_geometric_series_cot(self):
        """Sum of series with off-by-one error in number of terms"""
        a = self.rng.randint(1, 5)  # First term
        r = self.rng.choice([2, 3, 0.5])  # Ratio
        n = self.rng.randint(8, 15)  # Terms
        
        is_hallu = self.rng.random() < self.hallucination_rate
        
        # Real: S_n = a(1-r^n)/(1-r)
        real_sum = a * (1 - r**n) / (1 - r)
        
        # Hallucination: off-by-one error (uses n-1 or n+1)
        hallu_sum = a * (1 - r**(n-1)) / (1 - r) if r != 1 else a * (n-1)
        
        steps = [
            f"Geometric series: a={a}, r={r}, n={n}",
            f"Formula: S_n = a(1-r^n)/(1-r)",
            f"Plug in: S_{n} = {a}(1-{r}^{n})/(1-{r})",
            f"Calculate r^n: {r}^{n} = {r**n:.0f}" if r != 1 else f"Special case: r=1",
            f"Numerator: 1 - {r**n:.0f} = {1-r**n:.0f}" if not is_hallu else f"Using n-1={n-1}: r^{n-1} = {r**(n-1):.0f}",
            f"Denominator: 1 - {r} = {1-r:.1f}",
            f"Total: ${real_sum:.0f}$" if not is_hallu else f"Total: ${hallu_sum:.0f}$",
            f"Verify by adding terms: {a} + {a*r} + ..." if not is_hallu else "Pattern looks correct"
        ]
        
        # Pad to 14-16 steps
        while len(steps) < 15:
            steps.append(f"Double-check calculation")
        
        return {
            "question": f"Find the sum of the geometric series with first term {a}, ratio {r}, and {n} terms.",
            "answer": " → ".join(steps),
            "label": 0 if not is_hallu else 1,
            "domain": "math"
        }
    
    def _generate_rate_problem_cot(self):
        """Work-rate problem with unit conversion error"""
        # e.g., "If Alice paints a room in 3 hours and Bob in 5 hours, how long together?"
        rate1 = self.rng.randint(2, 8)
        rate2 = self.rng.randint(3, 12)
        
        is_hallu = self.rng.random() < self.hallucination_rate
        
        # Real: 1/t = 1/rate1 + 1/rate2
        real_time = 1 / (1/rate1 + 1/rate2)
        
        # Hallucination: add rates instead of reciprocals (common student error)
        hallu_time = (rate1 + rate2) / 2  # Wrong: averages instead of harmonic mean
        
        steps = [
            f"Alice's rate: 1 room per {rate1} hours",
            f"Bob's rate: 1 room per {rate2} hours",
            "Combined rate: 1/rate1 + 1/rate2" if not is_hallu else "Combined: (rate1 + rate2)/2",
            f"1/{rate1} + 1/{rate2} = {1/rate1 + 1/rate2:.3f} rooms/hour" if not is_hallu else f"({rate1}+{rate2})/2 = {hallu_time:.1f} hours",
            f"Time = 1 / rate = {real_time:.2f} hours" if not is_hallu else f"Time = {hallu_time:.1f} hours",
            f"In minutes: {real_time*60:.0f} minutes" if not is_hallu else f"In minutes: {hallu_time*60:.0f} minutes"
        ]
        
        # Pad to 12-15 steps
        while len(steps) < 14:
            steps.append(f"Check: does this make sense?")
        
        return {
            "question": f"Alice can paint a room in {rate1} hours. Bob can paint it in {rate2} hours. How long if they work together?",
            "answer": " → ".join(steps),
            "label": 0 if not is_hallu else 1,
            "domain": "math"
        }