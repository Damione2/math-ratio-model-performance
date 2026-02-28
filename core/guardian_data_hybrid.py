# core/guardian_data_hybrid.py (v12.1 - Import Fix & Math Merging)
"""
Hybrid dataset generator with hallucination rate verification, post-hoc balancing,
and math data merging for long-context training.
"""

# ==================== CRITICAL IMPORT FIX - MUST BE FIRST ====================
import sys
from pathlib import Path

# Add project root to sys.path BEFORE any local imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# ============================================================================

import pickle
import numpy as np
from tqdm import tqdm
from collections import Counter
import os
import argparse
import random

# NOW safe to import from project root using absolute imports
try:
    from generators import (
        DiverseMathGenerator,
        DiverseCodeGenerator,
        DiverseRealWorldGenerator as RealGenerator,
        AdversarialSeedGenerator
    )
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Make sure 'generators' folder exists at project root")
    sys.exit(1)

try:
    from core.guardian_utils import FeatureExtractor  # ✅ FIXED: Absolute import
except ImportError as e:
    print(f"⚠️  Import warning: {e}")
    print("   Make sure guardian_utils.py exists in core/")
    sys.exit(1)

try:
    from config import ARTIFACTS_DIR
except ImportError:
    ARTIFACTS_DIR = Path("artifacts")

# CRITICAL: Disable Triton compilation completely
os.environ["TRITON_DISABLE"] = "1"
# ✅ FIXED: Use proper Windows path instead of invalid "NUL"
if os.name == 'nt':  # Windows
    triton_cache = os.path.join(os.path.expanduser("~"), ".triton_cache")
    os.makedirs(triton_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache
os.environ.setdefault("TRITON_DISABLE", "1")

import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# -------------------------
# Dataset generation with compliance verification
# -------------------------
def generate_hybrid_dataset(total_count=6000, adv_seed_count=0, adv_weight=1.5, seed=42):
    """
    Generate a balanced hybrid dataset with hallucination rate verification.
    
    Args:
        total_count (int): total number of base samples (excluding seeds)
        adv_seed_count (int): number of adversarial seed examples to generate
        adv_weight (float): relative sampling weight for adversarial examples
        seed (int): RNG seed for reproducibility
    
    Returns:
        dataset (list of dict): perfectly balanced list with 50% hallucination rate
    """
    # Use separate RNG for balancing operations
    balance_rng = random.Random(seed + 999)
    
    print(f"\n{'='*80}")
    print(f"Generating {total_count} base samples + {adv_seed_count} seeds (seed={seed})...")
    print(f"Target overall hallucination rate: 50%")
    
    # ✅ Step 1: Calculate dynamic hallucination rate for base generators
    if adv_seed_count > 0:
        target_rate = 0.5
        # Formula: base_rate = (target * (total + seeds) - seeds) / total
        base_hallucination_rate = (target_rate * (total_count + adv_seed_count) - adv_seed_count) / total_count
        base_hallucination_rate = max(0.1, min(0.9, base_hallucination_rate))  # Clamp for safety
        print(f"  → Required base generator hallucination rate: {base_hallucination_rate:.4f}")
    else:
        base_hallucination_rate = 0.5
        print(f"  → Using default hallucination rate: 0.5")

    # ✅ Step 2: Initialize generators with isolated RNG seeds
    # CRITICAL: Each generator gets a UNIQUE seed to prevent RNG interference
    gens = {
        'math': DiverseMathGenerator(seed=seed + 1001, hallucination_rate=base_hallucination_rate),
        'code': DiverseCodeGenerator(seed=seed + 2001, hallucination_rate=base_hallucination_rate),
        'real': RealGenerator(seed=seed + 3001, hallucination_rate=base_hallucination_rate)
    }
    
    per_domain = total_count // 3
    dataset = []
    seen = set()
    
    # ✅ Step 3: Track actual hallucination rates per generator for verification
    generator_stats = {domain: {'total': 0, 'hallucinations': 0} for domain in gens.keys()}
    
    print(f"\n{'='*80}")
    print(f"Phase 1: Generating base samples with rate verification...")
    print(f"{'='*80}")

    for domain_key, gen in gens.items():
        print(f"\n[DOMAIN: {domain_key.upper()} | Target: {per_domain} samples | Rate: {base_hallucination_rate:.2%}]")
        count = 0
        consecutive_fails = 0
        
        # Add compliance verification for this generator
        compliance_tracker = {'expected_hallu': 0, 'actual_hallu': 0}

        with tqdm(total=per_domain,
                  desc=f"{domain_key.upper():<10}",
                  unit="sample",
                  ascii=True,
                  leave=True,
                  miniters=1,
                  dynamic_ncols=True) as pbar:
            
            while count < per_domain:
                try:
                    sample = gen.generate()
                except Exception as e:
                    print(f"\n[ERROR] Generator failed for {domain_key}: {e}")
                    consecutive_fails += 1
                    if consecutive_fails > 1000:
                        print(f"\n[STOP] Exceeded max retries for {domain_key}. Breaking.")
                        break
                    continue

                # Normalize keys and dedupe by question+answer
                qa = f"{sample['question'].strip().lower()}||{sample['answer'].strip().lower()}"
                h = hash(qa)

                if h in seen:
                    consecutive_fails += 1
                    if consecutive_fails > 1000:
                        print(f"\n[WARNING] Max duplicate retries reached for {domain_key}. Breaking early.")
                        break
                    continue

                consecutive_fails = 0
                seen.add(h)
                dataset.append(sample)
                
                # ✅ Track actual hallucination rate
                generator_stats[domain_key]['total'] += 1
                if sample['label'] == 1:
                    generator_stats[domain_key]['hallucinations'] += 1
                
                count += 1
                pbar.update(1)

    # ✅ Step 4: Report actual vs expected rates
    print(f"\n{'='*80}")
    print(f"Phase 1 Complete: Generator Compliance Report")
    print(f"{'='*80}")
    
    total_base_hallu = 0
    total_base_samples = 0
    
    for domain, stats in generator_stats.items():
        actual_rate = stats['hallucinations'] / max(1, stats['total'])
        expected_rate = base_hallucination_rate
        
        print(f"  {domain.upper():12}: {stats['hallucinations']:5d}/{stats['total']:5d} = {actual_rate:.2%} (expected {expected_rate:.2%})")
        
        total_base_hallu += stats['hallucinations']
        total_base_samples += stats['total']
        
        # ⚠️ Alert if deviation is >5%
        deviation = abs(actual_rate - expected_rate)
        if deviation > 0.05:
            print(f"     ⚠️  WARNING: Deviation of {deviation:.2%} detected!")

    base_overall_rate = total_base_hallu / max(1, total_base_samples)
    print(f"  {'OVERALL':12}: {total_base_hallu:5d}/{total_base_samples:5d} = {base_overall_rate:.2%}")

    # ✅ Step 5: Generate adversarial seed examples (always 100% hallucinations)
    adv_samples = []
    if adv_seed_count and adv_seed_count > 0:
        print(f"\n{'='*80}")
        print(f"Phase 2: Generating {adv_seed_count} adversarial seed examples...")
        print(f"{'='*80}")
        
        # Adversarial generator should produce 100% hallucinations regardless of rate
        adv_gen = AdversarialSeedGenerator(seed=seed + 9999, hallucination_rate=1.0)  # Force 100%
        adv_seen = set()
        
        with tqdm(total=adv_seed_count, desc="ADV_SEED", unit="sample", ascii=True, leave=True) as pbar:
            adv_count = 0
            consecutive_fails = 0
            
            while adv_count < adv_seed_count:
                try:
                    s = adv_gen.generate()
                except Exception as e:
                    print(f"\n[ERROR] Adversarial generator failed: {e}")
                    consecutive_fails += 1
                    if consecutive_fails > 1000:
                        print(f"\n[STOP] Max retries for adversarial seeds. Breaking.")
                        break
                    continue

                qa = f"{s['question'].strip().lower()}||{s['answer'].strip().lower()}"
                h = hash(qa)
                if h in seen or h in adv_seen:
                    consecutive_fails += 1
                    if consecutive_fails > 1000:
                        print(f"\n[WARNING] Max duplicate retries for adversarial seeds. Breaking early.")
                        break
                    continue
                
                adv_seen.add(h)
                adv_samples.append(s)
                adv_count += 1
                pbar.update(1)

        print(f"  → Generated {len(adv_samples)} adversarial seeds (all hallucinations)")
        dataset.extend(adv_samples)

    # ✅ Step 6: FINAL POST-HOC BALANCING (safety net)
    print(f"\n{'='*80}")
    print(f"Phase 3: Post-hoc balancing to enforce 50% hallucination rate...")
    print(f"{'='*80}")
    
    # Separate by label
    valid_samples = [d for d in dataset if d['label'] == 0]
    hallu_samples = [d for d in dataset if d['label'] == 1]
    
    print(f"  Before balancing: {len(valid_samples)} valid, {len(hallu_samples)} hallucination")
    
    # Enforce exact 50% split
    target_count = min(len(valid_samples), len(hallu_samples))
    
    if target_count == 0:
        raise ValueError("Cannot balance: no samples of one class")
    
    # Randomly downsample the larger class
    if len(valid_samples) > target_count:
        valid_samples = balance_rng.sample(valid_samples, target_count)
        print(f"  → Downsampled valid to {target_count}")
    
    if len(hallu_samples) > target_count:
        hallu_samples = balance_rng.sample(hallu_samples, target_count)
        print(f"  → Downsampled hallucination to {target_count}")
    
    # Combine and shuffle
    dataset = valid_samples + hallu_samples
    balance_rng.shuffle(dataset)
    
    print(f"  After balancing: {len(dataset)} total samples (50% hallucination rate)")
    
    # ✅ Step 7: Final validation
    domain_counts = Counter(d['domain'] for d in dataset)
    label_counts = Counter(d['label'] for d in dataset)
    total_len = len(dataset)
    hallu_rate = (label_counts[1] / total_len) if total_len > 0 else 0.0
    
    if not abs(hallu_rate - 0.5) < 0.001:  # Should be exactly 0.5
        print(f"🚨 CRITICAL: Final hallucination rate is {hallu_rate:.2%}, not 50%!")
        raise RuntimeError("Post-hoc balancing failed")
    
    print(f"\n✅ Dataset generation complete:")
    print(f"   Total samples: {total_len}")
    print(f"   Domain distribution: {dict(domain_counts)}")
    print(f"   Label distribution: {dict(label_counts)} (Perfect 50% balance)")
    
    return dataset

# -------------------------
# Math data merging for long-context experiments
# -------------------------
def merge_math_data():
    """
    Merge new long-CoT math data with existing balanced dataset.
    Keeps 1000 old math examples + all new math + all code/real_world.
    """
    import pickle
    from collections import Counter
    
    print(f"\n{'='*80}")
    print(f"🧬 MERGING LONG-CoT MATH DATA")
    print(f"{'='*80}")
    
    # Load existing data
    existing_path = ARTIFACTS_DIR / "01_raw_data.pkl"
    if not existing_path.exists():
        raise FileNotFoundError(f"Existing dataset not found: {existing_path}")
    
    with open(existing_path, "rb") as f:
        existing_data = pickle.load(f)
    
    # Load new math data
    math_path = ARTIFACTS_DIR / "01_raw_data_math_long.pkl"
    if not math_path.exists():
        raise FileNotFoundError(f"New math dataset not found: {math_path}. Generate it first!")
    
    with open(math_path, "rb") as f:
        math_data = pickle.load(f)
    
    print(f"   Existing dataset: {len(existing_data)} samples")
    print(f"   New math data: {len(math_data)} samples")
    
    # Separate by domain
    old_math = [d for d in existing_data if d['domain'] == 'math']
    other_domains = [d for d in existing_data if d['domain'] != 'math']
    
    print(f"   Old math examples: {len(old_math)}")
    print(f"   Code/real examples: {len(other_domains)}")
    
    # Sample 1000 old math examples (preserve some diversity)
    import random
    random.seed(42)
    old_math_sampled = random.sample(old_math, min(1000, len(old_math)))
    
    # Combine: 1000 old math + all new math + all code/real_world
    merged = old_math_sampled + math_data + other_domains
    
    # Shuffle to avoid domain clumping
    random.shuffle(merged)
    
    # Save merged dataset
    merged_path = ARTIFACTS_DIR / "01_raw_data_merged.pkl"
    with open(merged_path, "wb") as f:
        pickle.dump(merged, f)
    
    # Statistics
    domain_counts = Counter(d['domain'] for d in merged)
    label_counts = Counter(d['label'] for d in merged)
    
    print(f"\n✅ Merged dataset saved: {merged_path}")
    print(f"   Total samples: {len(merged)}")
    print(f"   Domain distribution: {dict(domain_counts)}")
    print(f"   Label distribution: {dict(label_counts)}")
    print(f"   Hallucination rate: {label_counts[1]/len(merged):.2%}")
    
    return merged_path

# -------------------------
# Feature extraction pipeline
# -------------------------
def extract_features_pipeline(raw_data, batch_size=4):
    """
    Extracts features for a list of raw samples using FeatureExtractor.
    Each sample in raw_data is expected to be a dict with keys:
      - question
      - answer
      - label
      - domain
    
    The function appends a 'features' key to each sample with shape (28, 1536).
    
    Returns:
        processed (list): list of samples with 'features' added
    """
    extractor = FeatureExtractor()
    processed = []

    # Sort by length for stable batching
    raw_data = sorted(raw_data, key=lambda x: len(x['question'] + str(x['answer'])))

    print(f"\nExtracting features for {len(raw_data)} samples (batch_size={batch_size})...")
    
    for i in tqdm(range(0, len(raw_data), batch_size), desc="Extracting", ascii=True, dynamic_ncols=True):
        batch = raw_data[i:i + batch_size]
        q_list = [b['question'] for b in batch]
        a_list = [b['answer'] for b in batch]

        try:
            # Extract features from this batch
            feats = extractor.extract(q_list, a_list)
            
            # Validate shape
            if feats.shape != (len(batch), 28, 1536):
                print(f"Warning: Unexpected feature shape {feats.shape} for batch starting at index {i}")
            
            # Attach features to samples
            for j, item in enumerate(batch):
                item['features'] = feats[j]
                processed.append(item)
                
        except Exception as e:
            print(f"\nFATAL: Feature extraction failed for batch starting at index {i}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("Feature extraction failed. Ensure model can load on CPU and Triton is disabled.")

    # Final validation
    if len(processed) != len(raw_data):
        print(f"Warning: Only extracted features for {len(processed)}/{len(raw_data)} samples")
    
    # Validate first sample
    if processed and 'features' in processed[0]:
        feat_shape = np.array(processed[0]['features']).shape
        if feat_shape != (28, 1536):
            raise ValueError(f"Feature shape validation failed: expected (28, 1536), got {feat_shape}")

    print(f"Successfully extracted features for {len(processed)} samples")
    return processed

# -------------------------
# Convenience helpers
# -------------------------
def build_and_save_dataset(total_count=6000, adv_seed_count=0, out_prefix="01_raw_data.pkl"):
    """
    Generates dataset and saves raw dataset to artifacts.
    """
    raw = generate_hybrid_dataset(total_count=total_count, adv_seed_count=adv_seed_count)
    raw_path = ARTIFACTS_DIR / out_prefix
    
    # Ensure directory exists
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(raw_path, "wb") as f:
        pickle.dump(raw, f)
    print(f"Saved raw dataset to {raw_path}")
    return raw_path

def save_processed_features(processed, split_name="train"):
    """
    Save processed (features added) list to artifacts as 03_{split}_features.pkl
    """
    path = ARTIFACTS_DIR / f"03_{split_name}_features.pkl"
    
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(processed, f)
    print(f"Saved processed features to {path}")
    return path

# -------------------------
# Command-line interface
# -------------------------
def split_merged_data():
    """
    Split merged dataset into train/val/test with stratification.
    Call this AFTER merging math data.
    """
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    
    print(f"\n{'='*80}")
    print(f"📊 SPLITTING MERGED DATASET")
    print(f"{'='*80}")
    
    # Load merged data
    merged_path = ARTIFACTS_DIR / "01_raw_data_merged.pkl"
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged dataset not found: {merged_path}")
    
    with open(merged_path, "rb") as f:
        data = pickle.load(f)
    
    domains = np.array([d['domain'] for d in data])
    labels = np.array([d['label'] for d in data])
    
    # First split: 70% train, 30% temp
    train_idx, temp_idx = train_test_split(
        range(len(data)), 
        test_size=0.3, 
        stratify=list(zip(domains, labels)),
        random_state=42
    )
    
    # Split temp into val/test (50/50 of temp = 15% each)
    temp_domains = domains[temp_idx]
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=list(zip(temp_domains, temp_labels)),
        random_state=42
    )
    
    # Create splits
    splits = [
        ('02_train', train_idx),
        ('02_val', val_idx),
        ('02_test', test_idx)
    ]
    
    for name, idx in splits:
        split_data = [data[i] for i in idx]
        out_path = ARTIFACTS_DIR / f"{name}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(split_data, f)
        
        # Statistics
        domain_counts = Counter(d['domain'] for d in split_data)
        print(f"   {name}: {len(split_data)} samples | {dict(domain_counts)}")
    
    print(f"\n✅ Split complete! Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or merge hybrid dataset for Guardian")
    parser.add_argument("--total-count", type=int, default=6000, help="Total samples to generate")
    parser.add_argument("--adv-seed-count", type=int, default=0, help="Number of adversarial seeds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--merge-math", action="store_true", help="Merge long-CoT math data")
    parser.add_argument("--split-merged", action="store_true", help="Split merged dataset into train/val/test")
    args = parser.parse_args()
    
    if args.merge_math:
        merge_math_data()
    elif args.split_merged:
        split_merged_data()
    else:
        print(f"\n🧬 Guardian Dataset Generator v12.1")
        print(f"Generating {args.total_count} base + {args.adv_seed_count} seeds")
        
        raw = generate_hybrid_dataset(
            total_count=args.total_count,
            adv_seed_count=args.adv_seed_count,
            seed=args.seed
        )
        
        raw_path = ARTIFACTS_DIR / "01_raw_data.pkl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "wb") as f:
            pickle.dump(raw, f)
        
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS! Saved perfectly balanced dataset to {raw_path}")
        print(f"{'='*80}")
        print(f"   Samples: {len(raw)}")
        print(f"   Domains: {Counter([d['domain'] for d in raw])}")
        print(f"   Labels: {Counter([d['label'] for d in raw])}")
        print(f"   Final hallucination rate: 50.00% (ENFORCED)")
        print(f"{'='*80}\n")