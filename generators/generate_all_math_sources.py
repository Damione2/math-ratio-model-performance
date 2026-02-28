#!/usr/bin/env python3
"""
generators/generate_all_math_sources.py - FIXED VERSION
Returns samples directly for pipeline integration.
"""

import sys
from pathlib import Path
import json
import pickle

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Import available generators
try:
    from generators.math_synthetic_v3 import generate_math_synthetic_v3
    HAS_V3 = True
except ImportError:
    HAS_V3 = False

try:
    from generators.math_synthetic_v4 import save_math_synthetic_v4_jsonl
    HAS_V4 = True
except ImportError:
    HAS_V4 = False

try:
    from generators.math_long_cot_v2 import save_math_long_cot_v2_short_jsonl
    HAS_LONG = True
except ImportError:
    HAS_LONG = False

try:
    from generators.math_adversarial_v2 import save_math_adversarial_v2_jsonl
    HAS_ADV = True
except ImportError:
    HAS_ADV = False

try:
    from generators.math_equation_systems_v1 import save_math_equation_systems_v1_jsonl
    HAS_EQ = True
except ImportError:
    HAS_EQ = False


def load_jsonl(path: Path) -> list:
    """Load samples from JSONL file."""
    samples = []
    if not path.exists():
        return samples
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return samples


def convert_to_guardian_format(raw_sample: dict, source: str) -> dict:
    """Convert generator output to Guardian pipeline format."""
    label = raw_sample.get('true_label', raw_sample.get('label', 0))
    if isinstance(label, str):
        label = 1 if label.lower() in ['hallucination', 'wrong', 'false', '1'] else 0
    
    return {
        'question': raw_sample.get('question', raw_sample.get('q', '')),
        'answer': raw_sample.get('answer', raw_sample.get('a', '')),
        'label': int(label),
        'domain': 'math',
        'meta': {
            'category': raw_sample.get('category', f'Math-{source}'),
            'source': source
        }
    }


def generate_v3_samples(n_pairs: int) -> list:
    """Generate v3 samples."""
    if not HAS_V3:
        return []
    
    print(f"  Generating {n_pairs} math_synthetic_v3 samples...")
    
    try:
        result = generate_math_synthetic_v3(n=n_pairs)
    except TypeError:
        result = generate_math_synthetic_v3()
    
    if isinstance(result, list):
        samples = [convert_to_guardian_format(s, 'math_synthetic_v3') for s in result]
        save_jsonl(samples, DATA_DIR / "math_synth_v3.jsonl")
        return samples
    elif hasattr(result, '__iter__'):
        samples = [convert_to_guardian_format(s, 'math_synthetic_v3') for s in result]
        save_jsonl(samples, DATA_DIR / "math_synth_v3.jsonl")
        return samples
    else:
        print(f"  ⚠️  Unexpected return type from v3: {type(result)}")
        return []


def generate_v4_samples(n_pairs: int) -> list:
    """Generate v4 samples."""
    if not HAS_V4:
        return []
    
    print(f"  Generating {n_pairs} math_synthetic_v4 samples...")
    output_path = DATA_DIR / "math_synth_v4.jsonl"
    
    try:
        save_math_synthetic_v4_jsonl(str(output_path), n_pairs=n_pairs)
        samples = load_jsonl(output_path)
        return [convert_to_guardian_format(s, 'math_synthetic_v4') for s in samples]
    except Exception as e:
        print(f"  ⚠️  v4 generation failed: {e}")
        return []


def generate_long_samples(n_pairs: int) -> list:
    """Generate long CoT v2 samples."""
    if not HAS_LONG:
        return []
    
    print(f"  Generating {n_pairs} math_long_cot_v2 samples...")
    output_path = DATA_DIR / "math_long_cot_short.jsonl"
    
    try:
        save_math_long_cot_v2_short_jsonl(str(output_path), n_pairs=n_pairs)
        samples = load_jsonl(output_path)
        return [convert_to_guardian_format(s, 'math_long_cot_v2') for s in samples]
    except Exception as e:
        print(f"  ⚠️  long_cot generation failed: {e}")
        return []


def generate_adv_samples(n_pairs: int) -> list:
    """Generate adversarial v2 samples."""
    if not HAS_ADV:
        return []
    
    print(f"  Generating {n_pairs} math_adversarial_v2 samples...")
    output_path = DATA_DIR / "math_adversarial_v2.jsonl"
    
    try:
        save_math_adversarial_v2_jsonl(str(output_path), n_pairs=n_pairs)
        samples = load_jsonl(output_path)
        return [convert_to_guardian_format(s, 'math_adversarial_v2') for s in samples]
    except Exception as e:
        print(f"  ⚠️  adversarial generation failed: {e}")
        return []


def generate_eq_samples(n_pairs: int) -> list:
    """Generate equation systems v1 samples."""
    if not HAS_EQ:
        return []
    
    print(f"  Generating {n_pairs} math_equation_systems_v1 samples...")
    output_path = DATA_DIR / "math_eqsys_v1.jsonl"
    
    try:
        save_math_equation_systems_v1_jsonl(str(output_path), n_pairs=n_pairs)
        samples = load_jsonl(output_path)
        return [convert_to_guardian_format(s, 'math_equation_systems_v1') for s in samples]
    except Exception as e:
        print(f"  ⚠️  eqsys generation failed: {e}")
        return []


def save_jsonl(samples: list, path: Path):
    """Save samples to JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')


def generate_and_merge(
    n_v3: int = 500,
    n_v4: int = 500,
    n_long: int = 400,
    n_adv: int = 600,
    n_eq: int = 300,
    target_merged: int = None
) -> list:  # <-- CHANGED: Returns list instead of int
    """
    Generate all math sources and return as list for Guardian pipeline.
    
    Returns:
        List of samples in Guardian format (not just count!)
    """
    
    print("\n" + "=" * 60)
    print("GENERATING ADVERSARIAL MATH SOURCES")
    print("=" * 60)
    
    all_samples = []
    
    if n_v3 > 0:
        samples = generate_v3_samples(n_v3)
        if samples:
            all_samples.extend(samples)
            print(f"  ✅ v3: {len(samples)} samples")
    
    if n_v4 > 0:
        samples = generate_v4_samples(n_v4)
        if samples:
            all_samples.extend(samples)
            print(f"  ✅ v4: {len(samples)} samples")
    
    if n_long > 0:
        samples = generate_long_samples(n_long)
        if samples:
            all_samples.extend(samples)
            print(f"  ✅ long_cot: {len(samples)} samples")
    
    if n_adv > 0:
        samples = generate_adv_samples(n_adv)
        if samples:
            all_samples.extend(samples)
            print(f"  ✅ adversarial: {len(samples)} samples")
    
    if n_eq > 0:
        samples = generate_eq_samples(n_eq)
        if samples:
            all_samples.extend(samples)
            print(f"  ✅ eqsys: {len(samples)} samples")
    
    if not all_samples:
        print("\n⚠️  No samples generated from any source!")
        return []
    
    # Apply target limit if specified
    if target_merged and len(all_samples) > target_merged:
        import random
        random.seed(42)
        all_samples = random.sample(all_samples, target_merged)
        print(f"\n  Downsampled to {target_merged} samples")
    
    # Save merged outputs (for caching)
    print(f"\n{'='*60}")
    print("SAVING MERGED OUTPUT")
    print(f"{'='*60}")
    
    # Save JSONL
    merged_jsonl = DATA_DIR / "adv_math_merged.jsonl"
    save_jsonl(all_samples, merged_jsonl)
    print(f"  JSONL: {merged_jsonl} ({len(all_samples)} samples)")
    
    # Save pickle for fast loading
    merged_pkl = DATA_DIR / "adv_math_merged.pkl"
    with open(merged_pkl, 'wb') as f:
        pickle.dump(all_samples, f)
    print(f"  Pickle: {merged_pkl}")
    
    # Stats
    from collections import Counter
    sources = Counter([s.get('meta', {}).get('source', 'unknown') for s in all_samples])
    labels = Counter([s['label'] for s in all_samples])
    
    print(f"\n  Source breakdown:")
    for src, cnt in sorted(sources.items()):
        print(f"    {src}: {cnt}")
    print(f"\n  Label distribution: {dict(labels)}")
    
    # RETURN THE SAMPLES, not just count!
    return all_samples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-v3", type=int, default=500)
    parser.add_argument("--n-v4", type=int, default=500)
    parser.add_argument("--n-long", type=int, default=400)
    parser.add_argument("--n-adv", type=int, default=600)
    parser.add_argument("--n-eq", type=int, default=300)
    parser.add_argument("--target-merged", type=int, default=None)
    args = parser.parse_args()
    
    samples = generate_and_merge(
        n_v3=args.n_v3,
        n_v4=args.n_v4,
        n_long=args.n_long,
        n_adv=args.n_adv,
        n_eq=args.n_eq,
        target_merged=args.target_merged
    )
    
    if samples:
        print(f"\n{'='*60}")
        print(f"✅ READY: {len(samples)} adversarial math samples")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("❌ NO SAMPLES GENERATED")
        print(f"{'='*60}")