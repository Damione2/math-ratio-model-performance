# utils/merge_code_jsonl.py
import json
from pathlib import Path

base = Path("data/llm_adv_code_existing.jsonl")
syn = Path("data/synthetic_code_negatives.jsonl")
out = Path("data/code_augmented.jsonl")

seen = set()
out.parent.mkdir(parents=True, exist_ok=True)

def write_line(f, obj):
    f.write(json.dumps(obj, ensure_ascii=False) + "\n")

with out.open("w", encoding="utf-8") as fo:
    if base.exists():
        for line in base.open("r", encoding="utf-8"):
            try:
                obj = json.loads(line)
                key = (obj.get("question",""), obj.get("answer",""))
                if key not in seen:
                    write_line(fo, obj)
                    seen.add(key)
            except Exception:
                continue
    if syn.exists():
        for line in syn.open("r", encoding="utf-8"):
            try:
                obj = json.loads(line)
                key = (obj.get("question",""), obj.get("answer",""))
                if key not in seen:
                    write_line(fo, obj)
                    seen.add(key)
            except Exception:
                continue

print("Wrote merged file to", out)
