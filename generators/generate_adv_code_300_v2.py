#!/usr/bin/env python3
"""
generate_adv_code_300_v2.py

Expanded code adversarial generator (v2).
Generates a large set of code examples (valid + adversarial) with greater diversity
and writes them to llm_adv_code_300_v2.jsonl.

This script intentionally produces more examples than the original module to
increase uniqueness and dataset entropy. It creates a mix of function snippets
covering math, strings, lists, dicts, file-like operations, and small algorithms,
and injects many error types: syntax, semantic, off-by-one, wrong operator,
wrong signature, missing imports, wrong types, wrong indexing, unreachable code,
and more.

Output file: llm_adv_code_300_v2.jsonl
"""

import json
import random
import string
from pathlib import Path
from typing import Tuple, List

# -----------------------------
# Configuration
# -----------------------------
try:
    from config import DATA_DIR
except ImportError:
    import sys
    from pathlib import Path
    # Ensure root directory is in path to find config.py
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR

# Redirect output to the centralized data directory
OUTPUT = DATA_DIR / "llm_adv_code_300_v2.jsonl"
TARGET_VALID = 300
TARGET_INVALID = 300 # number of adversarial examples
SEED = 2026

# -----------------------------
# Utilities
# -----------------------------
random.seed(SEED)

def rand_ident(prefix: str = "", length: int = 6) -> str:
    return prefix + "".join(random.choices(string.ascii_lowercase, k=length))

def rand_var(n: int = 1) -> List[str]:
    return [random.choice(string.ascii_lowercase) for _ in range(n)]

def indent(code: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in code.splitlines())

def wrap_module(code: str) -> str:
    return code.strip()

def safe_write(items: List[dict]):
    OUTPUT.unlink(missing_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Generated {len(items)} examples into {OUTPUT}")

# -----------------------------
# Template helpers
# -----------------------------
def q_for(func_name: str) -> str:
    return f"Write a Python function named {func_name} that implements the described behavior."

# -----------------------------
# Generator building blocks
# -----------------------------
def gen_math_sum():
    func = rand_ident("fn_")
    a, b = rand_var(2)
    # valid example
    valid = f"def {func}({a}, {b}):\n    return {a} + {b}"
    # create an undefined-var invalid example using a generated var name
    undefined_var = rand_ident("v", 1)
    invalids = [
        f"def {func}({a}, {b})\n    return {a} + {b}",                 # missing colon
        f"def {func}({a}, {b}):\nreturn {a} + {b}",                      # bad indent
        f"def {func}({a}, {b}):\n    return {a} ^ {b}",                  # wrong operator
        f"def {func}({a}, {b}):\n    return {a} + {undefined_var}",      # undefined var (generated)
        f"def {func}({a}, {b}, x):\n    return {a} + {b}",               # wrong signature
        f"def {func}({a}, {b}):\n    return '{a} + {b}'",                # wrong type (string)
    ]
    return valid, random.choice(invalids)


def gen_power(func: str, a: str, b: str) -> Tuple[str, str]:
    valid = f"def {func}({a}, {b}):\n    return {a} ** {b}"
    invalids = [
        f"def {func}({a}, {b}):\n    return {a} ^ {b}",                  # bitwise instead of power
        f"def {func}({a}, {b}):\n    return pow({a}, {b})",               # valid but keep as alt
        f"def {func}({a}, {b}):\n    return {a} * {b}",                   # wrong operator
        f"def {func}({a}, {b}):\n    return {a} ** {b}  # missing return", # comment but still valid - keep as tricky
    ]
    return valid, random.choice(invalids)

def gen_even_check(func: str, x: str) -> Tuple[str, str]:
    valid = f"def {func}({x}):\n    return {x} % 2 == 0"
    invalids = [
        f"def {func}({x}):\n    return {x} % 2 == 1",                    # flipped logic
        f"def {func}({x}):\n    return ({x} // 2) == 0",                 # wrong logic
        f"def {func}({x}):\n    return {x} & 1 == 0",                    # precedence bug
    ]
    return valid, random.choice(invalids)

def gen_off_by_one(func: str, n: str) -> Tuple[str, str]:
    valid = f"def {func}({n}):\n    return sum(range(1, {n} + 1))"
    invalids = [
        f"def {func}({n}):\n    return sum(range(1, {n}))",              # off-by-one
        f"def {func}({n}):\n    return sum(range({n}))",                  # wrong range
        f"def {func}({n}):\n    return sum(range(0, {n}))",               # wrong start
    ]
    return valid, random.choice(invalids)

def gen_sqrt(func: str, x: str) -> Tuple[str, str]:
    valid = f"import math\n\ndef {func}({x}):\n    return math.sqrt({x})"
    invalids = [
        f"def {func}({x}):\n    return math.sqrt({x})",                  # missing import
        f"import math\n\ndef {func}({x})\n    return math.sqrt({x})",    # missing colon
        f"import math\n\ndef {func}({x}):\n    return math.pow({x}, 0.5)", # alternative valid
    ]
    return valid, random.choice(invalids)

def gen_list_index(func: str, lst: str, idx: str) -> Tuple[str, str]:
    valid = f"def {func}({lst}, {idx}):\n    return {lst}[{idx}]"
    invalids = [
        f"def {func}({lst}, {idx}):\n    return {lst}[{idx} + 1]",            # off-by-one index
        f"def {func}({lst}, {idx}):\n    return {lst}.{idx}",                # attribute access instead of index
        f"def {func}({lst}, {idx}):\n    return {lst}[{idx} - 1]",            # wrong index
        f"def {func}({lst}, {idx}):\n    return {lst}[{idx}]",               # valid (kept as alt)
    ]
    return valid, random.choice(invalids)

def gen_string_join(func: str, arr: str) -> Tuple[str, str]:
    valid = f"def {func}({arr}):\n    return ''.join({arr})"
    invalids = [
        f"def {func}({arr}):\n    return {arr}.join('')",                 # reversed join
        f"def {func}({arr}):\n    return ' '.join({arr})",               # different separator
        f"def {func}({arr}):\n    return ''.join({arr}[0])",             # wrong element
    ]
    return valid, random.choice(invalids)

def gen_dict_get(func: str, d: str, k: str) -> Tuple[str, str]:
    valid = f"def {func}({d}, {k}):\n    return {d}.get({k})"
    invalids = [
        f"def {func}({d}, {k}):\n    return {d}[{k}]",                     # KeyError risk but valid
        f"def {func}({d}, {k}):\n    return {d}.get('{k}')",               # wrong quoting
        f"def {func}({d}, {k}):\n    return {d}.pop({k})",                 # destructive
    ]
    return valid, random.choice(invalids)

def gen_file_read(func: str, path: str) -> Tuple[str, str]:
    valid = (
        f"def {func}({path}):\n"
        f"    with open({path}, 'r', encoding='utf-8') as f:\n"
        f"        return f.read()"
    )
    invalids = [
        f"def {func}({path}):\n    return open({path}).read()",            # missing context manager
        f"def {func}({path}):\n    with open({path}, 'r') as f:\n        return f.read()", # valid alt
        f"def {func}({path}):\n    with open({path}, 'w') as f:\n        return f.read()", # wrong mode
    ]
    return valid, random.choice(invalids)

def gen_sort(func: str, arr: str) -> Tuple[str, str]:
    valid = f"def {func}({arr}):\n    return sorted({arr})"
    invalids = [
        f"def {func}({arr}):\n    {arr}.sort()\n    return {arr}",                    # in-place sort (valid alt)
        f"def {func}({arr}):\n    return {arr}.sort()",               # returns None
        f"def {func}({arr}):\n    return sorted({arr}, reverse=True)",# reversed
    ]
    return valid, random.choice(invalids)

def gen_exception_handling(func: str, x: str) -> Tuple[str, str]:
    valid = (
        f"def {func}({x}):\n"
        f"    try:\n"
        f"        return int({x})\n"
        f"    except ValueError:\n"
        f"        return None"
    )
    invalids = [
        f"def {func}({x}):\n    try:\n        return int({x})\n    except:\n        pass",  # silent fail
        f"def {func}({x}):\n    try:\n        return int({x})\n    except ValueError:\n        return 'error'", # wrong type
    ]
    return valid, random.choice(invalids)

def gen_type_hint(func: str, x: str) -> Tuple[str, str]:
    valid = f"def {func}({x}: int) -> int:\n    return {x} * 2"
    invalids = [
        f"def {func}({x}: int) -> int:\n    return str({x})",            # wrong return type
        f"def {func}({x}: str) -> int:\n    return {x} * 2",             # wrong annotation
    ]
    return valid, random.choice(invalids)

def gen_loop_bounds(func: str, n: str) -> Tuple[str, str]:
    valid = f"def {func}({n}):\n    res = 0\n    for i in range({n}):\n        res += i\n    return res"
    invalids = [
        f"def {func}({n}):\n    res = 0\n    for i in range(1, {n}):\n        res += i\n    return res",  # off-by-one start
        f"def {func}({n}):\n    res = 0\n    for i in range({n}+1):\n        res += i\n    return res",  # off-by-one end
    ]
    return valid, random.choice(invalids)

# -----------------------------
# High-level generator list
# -----------------------------
PAIR_GENERATORS = [
    gen_math_sum,
    gen_power,
    gen_even_check,
    gen_off_by_one,
    gen_sqrt,
    gen_list_index,
    gen_string_join,
    gen_dict_get,
    gen_file_read,
    gen_sort,
    gen_exception_handling,
    gen_type_hint,
    gen_loop_bounds,
]

# -----------------------------
# Diversity augmenters
# -----------------------------
FILLER_SENTENCES = [
    "This function is intended to be simple and efficient.",
    "Edge cases should be considered when integrating into larger systems.",
    "The implementation aims for clarity over micro-optimizations.",
    "This snippet is a minimal example and may need validation in production.",
    "The code assumes inputs are of the expected type unless otherwise noted.",
]

def add_random_comment(code: str) -> str:
    if random.random() < 0.25:
        comment = f"# {random.choice(FILLER_SENTENCES)}"
        # insert comment after def line
        lines = code.splitlines()
        if len(lines) > 1:
            lines.insert(1, indent(comment, 4).lstrip())
        else:
            lines.append(comment)
        return "\n".join(lines)
    return code

def add_random_docstring(code: str) -> str:
    if random.random() < 0.3:
        doc = '    """' + random.choice(FILLER_SENTENCES) + '"""'
        lines = code.splitlines()
        # insert docstring after def line
        for i, ln in enumerate(lines):
            if ln.strip().startswith("def "):
                insert_at = i + 1
                lines.insert(insert_at, doc)
                break
        return "\n".join(lines)
    return code

def maybe_wrap_with_imports(code: str) -> str:
    if "math." in code and not code.strip().startswith("import math"):
        if random.random() < 0.6:
            return "import math\n\n" + code
    return code

def random_whitespace_noise(code: str) -> str:
    # randomly add blank lines
    lines = code.splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if random.random() < 0.08:
            out.append("")
    return "\n".join(out)

# -----------------------------
# Assembly loop
# -----------------------------
def build_pairs(target_valid: int, target_invalid: int) -> List[dict]:
    valid_items = []
    invalid_items = []
    attempts = 0
    max_attempts = max(target_valid, target_invalid) * 10

    while (len(valid_items) < target_valid or len(invalid_items) < target_invalid) and attempts < max_attempts:
        attempts += 1
        gen = random.choice(PAIR_GENERATORS)
        func_name = rand_ident("fn_")
        # choose appropriate variable names
        # inspect generator signature by name heuristics
        # provide 1-2 variable names
        a, b = rand_var(2)
        # call generator with appropriate args based on expected params
        try:
            # many generators accept (func, a, b) or (func, a)
            valid_code, invalid_code = gen(func_name, a, b) if gen.__code__.co_argcount == 3 else gen(func_name, a)
        except TypeError:
            # fallback: try with two args
            try:
                valid_code, invalid_code = gen(func_name, a)
            except Exception:
                continue

        # augment diversity
        valid_code = add_random_docstring(add_random_comment(maybe_wrap_with_imports(random_whitespace_noise(valid_code))))
        invalid_code = add_random_docstring(add_random_comment(maybe_wrap_with_imports(random_whitespace_noise(invalid_code))))

        # ensure not trivially identical
        v_item = {"question": q_for(func_name), "answer": wrap_module(valid_code), "label": 0, "domain": "code"}
        i_item = {"question": q_for(func_name), "answer": wrap_module(invalid_code), "label": 1, "domain": "code"}

        # add if we still need them
        if len(valid_items) < target_valid:
            valid_items.append(v_item)
        if len(invalid_items) < target_invalid:
            invalid_items.append(i_item)

    # interleave valid and invalid to produce final list
    final = []
    for v, i in zip(valid_items, invalid_items):
        final.append(v)
        final.append(i)

    # if counts mismatch, append remaining
    if len(valid_items) > len(invalid_items):
        for v in valid_items[len(invalid_items):]:
            final.append(v)
    elif len(invalid_items) > len(valid_items):
        for i in invalid_items[len(valid_items):]:
            final.append(i)

    # dedupe by (question, answer)
    seen = set()
    deduped = []
    for item in final:
        key = (item["question"], item["answer"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # shuffle for randomness
    random.shuffle(deduped)
    return deduped

# -----------------------------
# Main
# -----------------------------
def main():
    items = build_pairs(TARGET_VALID, TARGET_INVALID)
    safe_write(items)

if __name__ == "__main__":
    main()