#!/usr/bin/env python3
# generators/merge_adv_sources.py
"""
Merge multiple JSONL generator outputs into a single adversarial math JSONL.
Integrated into Guardian 8-step pipeline - outputs format compatible with Step 1.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def merge_jsonl(sources: List[str], out_path: str, target_count: int = None) -> int:
    """
    Merge multiple JSONL sources into one, with optional down/upsampling.
    
    Args:
        sources: List of paths to JSONL files
        out_path: Output path for merged file
        target_count: If set, randomly sample this many entries (for balancing)
    
    Returns:
        Number of entries written
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all entries
    all_entries = []
    for source_path in sources:
        p = Path(source_path)
        if not p.exists():
            print(f"[merge] ⚠️  Missing source {p}, skipping")
            continue
        
        print(f"[merge] Loading {p}...")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    # Normalize format for Guardian pipeline
                    entry = {
                        "question": obj.get("question", ""),
                        "answer": obj.get("answer", ""),
                        "label": int(obj.get("true_label", obj.get("label", 0))),
                        "domain": "math",
                        "meta": {
                            "category": obj.get("category", "Math-Synthetic"),
                            "source": p.stem,
                            "index": len(all_entries) + 1
                        }
                    }
                    all_entries.append(entry)
                except json.JSONDecodeError:
                    continue
    
    # Optional: resample to target count
    if target_count and len(all_entries) > target_count:
        import random
        random.seed(42)
        all_entries = random.sample(all_entries, target_count)
        print(f"[merge] Downsampled to {target_count} entries")
    
    # Write merged output
    with out.open("w", encoding="utf-8") as fw:
        for i, entry in enumerate(all_entries, 1):
            entry["meta"]["index"] = i
            fw.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"[merge] ✅ Wrote {len(all_entries)} entries to {out_path}")
    return len(all_entries)


def load_merged_for_pipeline(merged_path: str) -> List[Dict[str, Any]]:
    """
    Load merged JSONL and convert to Guardian pipeline format.
    Called by Step 1 to inject additional math samples.
    """
    entries = []
    p = Path(merged_path)
    if not p.exists():
        return []
    
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Ensure Guardian-compatible format
            entries.append({
                "question": obj["question"],
                "answer": obj["answer"],
                "label": obj["label"],  # 0 or 1
                "domain": "math",
                "meta": obj.get("meta", {"category": "Math-Synthetic"})
            })
    
    return entries


if __name__ == "__main__":
    # Default: merge all math adversarial sources
    sources = [
        "data/math_synth_v3.jsonl",
        "data/math_synth_v4.jsonl", 
        "data/math_long_cot_short.jsonl",
        "data/math_adversarial_v2.jsonl",
        "data/math_eqsys_v1.jsonl"
    ]
    
    count = merge_jsonl(sources, "data/adv_math_merged.jsonl")
    print(f"\nMerged {count} total math adversarial samples")