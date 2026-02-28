#!/usr/bin/env python3
"""
pipeline/guardian_pipeline_master.py - PRODUCTION VERSION v1.3 (updated)

Changes in this updated copy:
- Adds a new CLI argument `--train-subset` which accepts a CSV or JSON file path.
  When provided and Step 4 is requested, the script will convert/merge that subset
  into ARTIFACTS_DIR/02_train.pkl so the trainer uses the provided training subset.
- Exposes and forwards common training flags: --batch-size and --num-workers.
- Keeps all original behavior and safety checks.
"""

import os
import sys
import argparse
import json
import pickle
import time
import random
import shutil
import gc
import psutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# ==================== CRITICAL: MODULE-SAFE PATH SETUP ====================
def setup_project_paths():
    """Robust path setup for both direct execution and module mode"""
    try:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
    except (NameError, AttributeError):
        project_root = Path.cwd()
        if not (project_root / "config.py").exists():
            print("❌ ERROR: Could not detect project root.")
            print(f"   Current directory: {project_root}")
            sys.exit(1)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

PROJECT_ROOT = setup_project_paths()

try:
    from config import ARTIFACTS_DIR, DATA_DIR
    from core import guardian_data_hybrid as data_gen
    from core.guardian_vision_core import GuardianVisionNet
    from core.guardian_utils import FeatureExtractor, FocalLoss
    from core.guardian_dataset_live import GuardianLiveDataLoader
    from core.guardian_trainer_moe import CoEvolutionaryTrainer
    from pipeline.snapshot_utils import snapshot_artifacts_on_C
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

ARTIFACTS_DIR = Path(ARTIFACTS_DIR)
DATA_DIR = Path(DATA_DIR)
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

# Default seed; may be overridden by CLI --seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# ==================== DATA VALIDATION UTILITIES ====================

def validate_meta_field(sample: dict, default_source: str = 'unknown') -> dict:
    """
    Ensure sample['meta'] is a proper dict with required fields.
    Fixes corrupted float/nan meta fields.
    """
    meta = sample.get('meta')

    if isinstance(meta, dict):
        # Ensure required keys exist
        meta.setdefault('source', default_source)
        meta.setdefault('category', 'Unknown')
        return sample

    # Meta is corrupted (float, nan, None, etc.)
    # Determine source from other fields if possible
    domain = sample.get('domain', 'unknown')
    if domain == 'math':
        source = 'math_base'
    elif domain == 'code':
        source = 'code_base'
    else:
        source = f"{domain}_base"

    sample['meta'] = {
        'source': source,
        'category': domain.capitalize(),
        'fixed': True  # Flag that this was auto-fixed
    }
    return sample


def validate_dataset(data: List[dict], dataset_name: str = "dataset") -> List[dict]:
    """
    Validate and fix entire dataset.
    Returns cleaned data with proper meta fields.
    """
    print(f"\n🔍 Validating {dataset_name}...")

    original_count = len(data)
    fixed_count = 0

    cleaned_data = []
    for i, sample in enumerate(data):
        # Ensure required fields exist
        if 'question' not in sample or 'answer' not in sample:
            print(f"  ⚠️  Sample {i}: Missing question/answer, skipping")
            continue

        if 'label' not in sample or sample['label'] not in (0, 1):
            print(f"  ⚠️  Sample {i}: Invalid label, skipping")
            continue

        if 'domain' not in sample:
            print(f"  ⚠️  Sample {i}: Missing domain, defaulting to 'math'")
            sample['domain'] = 'math'

        # Fix meta field
        original_meta = sample.get('meta')
        sample = validate_meta_field(sample)

        if not isinstance(original_meta, dict):
            fixed_count += 1

        cleaned_data.append(sample)

    print(f"  ✅ Validated: {len(cleaned_data)}/{original_count} samples")
    if fixed_count > 0:
        print(f"  🔧 Fixed {fixed_count} corrupted meta fields")

    return cleaned_data


def safe_pickle_dump(data: List[dict], path: Path, protocol: int = pickle.HIGHEST_PROTOCOL):
    """
    Safely dump data to pickle with validation.
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Validate before saving
    data = validate_dataset(data, path.name)

    # Use highest protocol for efficiency
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=protocol)

    print(f"  💾 Saved: {path} ({len(data)} samples)")


def safe_pickle_load(path: Path, validate: bool = True) -> List[dict]:
    """
    Safely load pickle with optional validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"Pickle not found: {path}")

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"  📂 Loaded: {path} ({len(data)} samples)")

    if validate:
        data = validate_dataset(data, path.name)

    return data


def purge_artifacts(preserve_files=None):
    """Purge artifacts with safety checks."""
    if not ARTIFACTS_DIR.exists():
        return

    preserve_files = preserve_files or []
    preserve_names = [Path(f).name for f in preserve_files]

    print(f"🧹 Purging {ARTIFACTS_DIR}...")

    for item in ARTIFACTS_DIR.iterdir():
        if item.name in preserve_names:
            print(f"  ⏭️  Preserving {item.name}")
            continue

        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            print(f"  ⚠️  Could not delete {item.name}: {e}")

    print("✓ Purge complete")


def get_memory_usage():
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024**3)
    vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    return ram_gb, vram_gb


# ==================== STEP 1: DATA GENERATION ====================

def step_1_generation(
    count: int = 9000,
    math_synthetic_pairs: int = 300,
    use_adv_math: bool = False,
    adv_math_target: Optional[int] = None,
    auto_generate_adv: bool = True,
    reduce_math: bool = False,
    math_keep_ratio: float = 1.0
):
    """
    Step 1: Generate data with proper meta field handling.
    """
    merged_path = ARTIFACTS_DIR / "01_raw_data_merged.pkl"

    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 1: SPIDER-NATIVE DATA GENERATION{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"   Target base samples: {count}")
    print(f"   Math synthetic v2 pairs: {math_synthetic_pairs}")
    print(f"   Use adversarial math: {use_adv_math}")
    if adv_math_target:
        print(f"   Adversarial math target: {adv_math_target}")
    print(f"   Generator Compliance: ENABLED")

    if merged_path.exists():
        print(f"\n{Colors.GREEN}📦 Loading merged dataset...{Colors.RESET}")
        raw = safe_pickle_load(merged_path, validate=True)
        data_source = "merged"
    else:
        print(f"\n{Colors.YELLOW}⚠️  Generating fresh data ({count} base samples)...{Colors.RESET}")
        purge_artifacts()

        # Generate base dataset
        raw = data_gen.generate_hybrid_dataset(
            total_count=count,
            adv_seed_count=0,
            seed=SEED
        )

        # Validate base data immediately
        raw = validate_dataset(raw, "base_dataset")

        # ============================================================
        # OPTIONAL: REDUCE MATH SAMPLES FOR ABLATION EXPERIMENT
        # ============================================================
        if reduce_math and 0.0 <= math_keep_ratio <= 1.0:
            print(f"\n⚠️ ABLATION: Reducing math samples to {math_keep_ratio*100:.0f}%")
            math_samples = [d for d in raw if d['domain'] == 'math']
            non_math_samples = [d for d in raw if d['domain'] != 'math']

            keep_n = int(len(math_samples) * math_keep_ratio)
            if keep_n < 0:
                keep_n = 0
            if keep_n > 0:
                math_samples = random.sample(math_samples, keep_n)
            else:
                math_samples = []

            raw = math_samples + non_math_samples
            random.shuffle(raw)

            print(f"   → Math reduced to {keep_n} samples")
            print(f"   → New dataset size: {len(raw)}")
        # ============================================================

        # Add math synthetic v2
        if math_synthetic_pairs > 0:
            print(f"\n🔢 Adding math synthetic v2 ({math_synthetic_pairs} pairs)...")
            try:
                from generators.math_synthetic_v2 import generate_math_synthetic_pairs_v2

                math_v2 = generate_math_synthetic_pairs_v2(
                    n_pairs=math_synthetic_pairs,
                    include_cot=True
                )

                for s in math_v2:
                    raw.append({
                        'question': s['question'],
                        'answer': s['answer'],
                        'label': int(s['true_label']),
                        'domain': 'math',
                        'meta': {
                            'source': 'math_synthetic_v2',
                            'category': 'Math-Synthetic-v2',
                            'version': 'v2.0'
                        }
                    })

                print(f"   Added {len(math_v2)} v2 samples")

            except Exception as e:
                print(f"   ⚠️  v2 generation failed: {e}")
                import traceback
                traceback.print_exc()

        # Add adversarial math sources
        if use_adv_math:
            print(f"\n🔢 Adding adversarial math sources...")
            try:
                from generators.generate_all_math_sources import generate_and_merge

                adv_samples = generate_and_merge(
                    n_v3=500, n_v4=500, n_long=400,
                    n_adv=600, n_eq=300,
                    target_merged=adv_math_target
                )

                if adv_samples and isinstance(adv_samples, list):
                    # Validate adversarial samples
                    adv_samples = validate_dataset(adv_samples, "adversarial_math")
                    raw.extend(adv_samples)
                    print(f"   Added {len(adv_samples)} adversarial samples")
                else:
                    print(f"   ⚠️  No adversarial samples generated")

            except Exception as e:
                print(f"   ⚠️  Adversarial generation failed: {e}")
                import traceback
                traceback.print_exc()

        data_source = "fresh_generated"

    # Balance dataset
    print(f"\n{'='*80}")
    print(f"Phase: Post-hoc balancing to enforce 50% hallucination rate...")
    print(f"{'='*80}")

    valid = [d for d in raw if d['label'] == 0]
    hallu = [d for d in raw if d['label'] == 1]

    print(f"  Before balancing: {len(valid)} valid, {len(hallu)} hallucination")

    target = min(len(valid), len(hallu))
    balance_rng = random.Random(SEED + 999)

    if len(valid) > target:
        valid = balance_rng.sample(valid, target)
        print(f"  → Downsampled valid to {target}")
    if len(hallu) > target:
        hallu = balance_rng.sample(hallu, target)
        print(f"  → Downsampled hallucination to {target}")

    raw = valid + hallu
    balance_rng.shuffle(raw)

    print(f"  After balancing: {len(raw)} total samples (50% hallucination rate)")

    # Validate final dataset
    raw = validate_dataset(raw, "final_dataset")

    # Save with validation
    safe_pickle_dump(raw, ARTIFACTS_DIR / "01_raw_data.pkl")

    # Generate manifest
    domain_counts = Counter([d['domain'] for d in raw])
    source_counts = Counter([d['meta'].get('source', 'unknown') for d in raw])

    manifest = {
        "n_samples": len(raw),
        "hallucination_rate": 0.5,
        "domain_counts": dict(domain_counts),
        "source_breakdown": dict(source_counts),
        "timestamp": time.time(),
        "version": "v1.0_production",
        "data_source": data_source
    }

    with open(ARTIFACTS_DIR / "01_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Step 1 complete: {len(raw):,} samples")
    print(f"   Domains: {dict(domain_counts)}")
    print(f"   Sources: {dict(source_counts)}")


# ==================== STEP 2: SPLITTING (FIXED) ====================

def step_2_splitting():
    """
    Step 2: Stratified splitting with proper meta field preservation.
    FIXED: Manual stratification without pd.DataFrame corruption.
    """

    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 2: STRATIFIED SPLITTING (80/10/10){Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    # Load with validation
    data = safe_pickle_load(ARTIFACTS_DIR / "01_raw_data.pkl")

    # CRITICAL Fix: Manual stratification without pd.DataFrame
    print("Creating stratification groups...")

    # Build stratification groups
    domain_label_pairs = [(d['domain'], d['label']) for d in data]
    unique_groups = list(set(domain_label_pairs))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    # Assign group indices
    for d in data:
        d['stratify_group'] = group_to_idx[(d['domain'], d['label'])]

    # Group samples
    group_samples = defaultdict(list)
    for d in data:
        group_samples[d['stratify_group']].append(d)

    train_data, val_data, test_data = [], [], []

    print(f"Splitting {len(unique_groups)} stratification groups...")

    for group_idx, samples in group_samples.items():
        n = len(samples)
        if n < 2:
            train_data.extend(samples)
            continue

        # Shuffle within group
        random.shuffle(samples)

        # Calculate split sizes (80/10/10)
        n_test = max(1, int(n * 0.1))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_test - n_val

        # Ensure at least 1 train sample
        if n_train < 1:
            n_train = 1
            n_val = (n - n_train) // 2
            n_test = n - n_train - n_val

        train_data.extend(samples[:n_train])
        val_data.extend(samples[n_train:n_train + n_val])
        test_data.extend(samples[n_train + n_val:])

    # Shuffle each split
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Save splits
    splits = {'train': train_data, 'val': val_data, 'test': test_data}

    for split_name, split_data in splits.items():
        # Remove temporary field
        for d in split_data:
            d.pop('stratify_group', None)

        # Validate and save
        split_data = validate_dataset(split_data, f"{split_name}_split")
        safe_pickle_dump(split_data, ARTIFACTS_DIR / f"02_{split_name}.pkl")

        # Stats
        domain_counts = Counter([d['domain'] for d in split_data])
        hallu_rate = sum([d['label'] for d in split_data]) / len(split_data)

        print(f"  → {split_name:6}: {len(split_data):5d} samples, "
              f"hallu={hallu_rate:.1%}, domains={dict(domain_counts)}")

    print("✅ Splits saved")


# ==================== STEP 3: EXTRACTION ====================

def step_3_extraction():
    """Step 3: Spider-Triangulation feature extraction."""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 3: SPIDER-TRIANGULATION EXTRACTION{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    for split_name in ["train", "val", "test"]:
        split_path = ARTIFACTS_DIR / f"02_{split_name}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

    from core.guardian_extract_memmap import run_extraction

    for split_name in ["train", "val", "test"]:
        print(f"  → Extracting {split_name} split...")
        try:
            run_extraction(split_name=split_name)
            print(f"  ✅ {split_name} extraction complete")
        except Exception as e:
            print(f"  ❌ {split_name} extraction failed: {e}")
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print("✓ All splits extracted to NVMe")


# ==================== UTILS: EXPERIMENT FOLDER HANDLING ====================

def ensure_experiment_folder(experiment_name: str) -> Path:
    """
    Ensure experiments/<experiment_name>/ exists and return its Path.
    """
    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def infer_experiment_name_from_manifest() -> Optional[str]:
    """
    Try to infer a stable experiment folder name from 01_manifest.json math ratio.
    Returns a folder name like 'math_50' or None if not inferable.
    """
    manifest_path = ARTIFACTS_DIR / "01_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            m = json.load(f)
        domain_counts = m.get("domain_counts", {})
        total = sum(domain_counts.values()) if domain_counts else 0
        math = domain_counts.get("math", 0)
        if total == 0:
            return None
        ratio = float(math) / float(total)
        pct = int(round(ratio * 100))
        # normalize to one of the expected folders (0,10,25,50,100) if close
        for target in [0,10,25,50,100]:
            if abs(pct - target) <= 5:
                return f"math_{target}"
        # otherwise return math_<pct>
        return f"math_{pct}"
    except Exception:
        return None

def copy_artifacts_to_experiment(exp_dir: Path):
    """
    Copy selected artifact files into the experiment folder.
    Overwrites existing files with the same name.
    """
    files_to_copy = [
        "training_log.csv",
        "training_summary.json",
        "guardian_spider_native.pth",
        "guardian_spider_native_full.pth",
        "train_meta.json",
        "vibration_summary.json",
        "vibration_per_sample.csv",
        "01_manifest.json"
    ]
    for fname in files_to_copy:
        src = ARTIFACTS_DIR / fname
        if src.exists():
            dst = exp_dir / src.name
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"⚠️ Could not copy {src} to {dst}: {e}")

# ==================== STEP 4: TRAINING (WRAPPER) ====================

def step_4_training(batch_size=512, num_workers=0, compile_model=True, resume=True,
                    domain_loss_weight=0.1, balance_loss_weight=5.0, entropy_loss_weight=0.001,
                    router_grad_clip=0.5, experiment: Optional[str] = None, output_dir_override: Optional[Path] = None):
    """Step 4: Spider-Native Live Training (with post-run C: snapshot)."""
    if sys.platform == "win32" and num_workers > 0:
        print(f"{Colors.YELLOW}⚠️  WINDOWS: Forcing num_workers=0{Colors.RESET}")
        num_workers = 0

    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 4: SPIDER-NATIVE LIVE TRAINING{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    # Verify data exists
    for split in ["train", "val"]:
        path = ARTIFACTS_DIR / f"02_{split}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Required data missing: {path}")

    # Initialize trainer
    trainer = CoEvolutionaryTrainer(
        artifacts_dir=ARTIFACTS_DIR,
        device="cuda" if torch.cuda.is_available() else "cpu",
        domain_loss_weight=domain_loss_weight
    )
    trainer.balance_loss_weight = balance_loss_weight
    trainer.entropy_loss_weight = entropy_loss_weight
    trainer.router_grad_clip = router_grad_clip
    trainer.resume = resume

    # Load data and train
    trainer.load_data(train_prefix="02_train", val_prefix="02_val")
    trainer.build_model()
    trainer.train(epochs=60, batch_size=batch_size)

    # After training completes, determine experiment folder name
    exp_name = None
    if experiment:
        exp_name = experiment
    else:
        exp_name = infer_experiment_name_from_manifest()
    if exp_name is None:
        ts = int(time.time())
        exp_name = f"run_{ts}"

    # Determine experiment directory (where small artifacts will be copied)
    if output_dir_override:
        exp_dir = Path(output_dir_override)
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        exp_dir = ensure_experiment_folder(exp_name)

    print(f"\n📁 Copying small artifacts to experiment folder: {exp_dir}")
    # copy selected small artifacts (this function already handles safe copying)
    copy_artifacts_to_experiment(exp_dir)

    # -----------------------------
    # Write per-run metrics.json
    # -----------------------------
    try:
        import json as _json
        import os as _os
        import time as _time
        from pathlib import Path as _Path

        # Determine run directory: prefer explicit OUTPUT_DIR, then RUN_ID, else fallback to exp_dir
        out_dir_env = _os.environ.get("OUTPUT_DIR")
        if out_dir_env:
            run_dir = _Path(out_dir_env)
        else:
            run_id_env = _os.environ.get("RUN_ID")
            if run_id_env:
                # if RUN_ID already contains 'run_' prefix, keep it; else prefix
                if str(run_id_env).startswith("run_"):
                    run_dir = _Path("runs") / str(run_id_env)
                else:
                    run_dir = _Path("runs") / f"run_{run_id_env}"
            else:
                # prefer exp_dir if available
                try:
                    run_dir = _Path(exp_dir)
                except Exception:
                    run_dir = _Path("runs") / f"run_{int(_time.time())}"

        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"

        # Try to obtain best_f1 from trainer or training_summary.json
        best_f1_val = None

        # 1) trainer.best_f1 if available
        try:
            best_f1_val = float(getattr(trainer, "best_f1", None))
        except Exception:
            best_f1_val = None

        # 2) fallback: try reading training_summary.json from run_dir, exp_dir or ARTIFACTS_DIR
        if best_f1_val is None:
            ts_candidates = [
                run_dir / "training_summary.json",
                exp_dir / "training_summary.json",
                ARTIFACTS_DIR / "training_summary.json",
            ]
            for ts in ts_candidates:
                try:
                    if ts.exists():
                        j = _json.loads(ts.read_text(encoding="utf-8"))
                        if "best_f1" in j:
                            best_f1_val = float(j["best_f1"])
                            break
                except Exception:
                    continue

        # 3) if still None, leave metrics empty object so analyzer can skip gracefully
        if best_f1_val is not None:
            metrics_obj = {"best_f1": float(best_f1_val)}
        else:
            metrics_obj = {}

        try:
            metrics_path.write_text(_json.dumps(metrics_obj, indent=2), encoding="utf-8")
            print(f"✅ Wrote metrics.json to {metrics_path}")
        except Exception as e:
            print(f"⚠️ Could not write metrics.json to {metrics_path}: {e}")

        # Debug prints to aid deterministic snapshot inclusion
        try:
            print(f"DEBUG: metrics written to: {metrics_path}")
            print(f"DEBUG: exp_dir: {exp_dir}, ARTIFACTS_DIR: {ARTIFACTS_DIR}")
        except Exception:
            pass

    except Exception as e:
        # Non-fatal: log and continue
        print(f"⚠️ Unexpected error while writing metrics.json: {e}")

    # -----------------------------
    # Create a lightweight snapshot on C: using hard links for large files
    # -----------------------------
    snapshot_root = Path(r"C:\guardian_runs")
    try:
        # Determine run_id for snapshot: prefer RUN_ID env var, else derive from exp_dir name
        run_id_env = os.environ.get("RUN_ID")
        if run_id_env:
            # normalize run_id to not include 'run_' prefix when passing to snapshot
            run_id_for_snapshot = str(run_id_env)
            if run_id_for_snapshot.startswith("run_"):
                run_id_for_snapshot = run_id_for_snapshot.replace("run_", "")
        else:
            # if output_dir_override was provided and looks like runs/run_<id>, use that
            try:
                candidate = exp_dir.name
                if candidate.startswith("run_"):
                    run_id_for_snapshot = candidate.replace("run_", "")
                else:
                    # fallback to timestamped run id
                    run_id_for_snapshot = f"{int(time.time())}"
            except Exception:
                run_id_for_snapshot = f"{int(time.time())}"

        # Build the run metrics path we wrote earlier
        out_dir_env = os.environ.get("OUTPUT_DIR")
        if out_dir_env:
            run_dir_for_snapshot = Path(out_dir_env)
        else:
            # prefer the run_dir we created earlier if available
            try:
                run_dir_for_snapshot = run_dir
            except Exception:
                run_dir_for_snapshot = Path("runs") / f"run_{run_id_for_snapshot}"

        run_metrics = run_dir_for_snapshot / "metrics.json"

        # Prepare extra_files list for snapshot_artifacts_on_C
        extra_files = []
        if run_metrics.exists():
            extra_files.append(str(run_metrics))
        else:
            # also try exp_dir location (some code wrote metrics into exp_dir)
            alt = Path(exp_dir) / "metrics.json"
            if alt.exists():
                extra_files.append(str(alt))

        print(f"🔁 Creating C: snapshot for run id: {run_id_for_snapshot} (including extra_files: {extra_files})")
        snapshot_path = snapshot_artifacts_on_C(
            run_id=run_id_for_snapshot,
            artifacts_dir=str(ARTIFACTS_DIR),
            snapshot_root=str(snapshot_root),
            extra_files=extra_files
        )
        print(f"✅ Snapshot created at: {snapshot_path}")
    except Exception as e:
        print(f"{Colors.YELLOW}⚠️ Snapshot creation failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

    print(f"✅ Artifacts copied to {exp_dir}")


# ==================== STEP 5: CALIBRATION ====================

def step_5_calibration():
    """Step 5: Temperature calibration."""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 5: SPIDER-NATIVE CALIBRATION{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    model_path = ARTIFACTS_DIR / "guardian_spider_native.pth"
    val_features = ARTIFACTS_DIR / "val_features.bin"
    val_meta = ARTIFACTS_DIR / "val_metadata.pkl"

    missing = []
    if not model_path.exists():
        missing.append(f"Model: {model_path}")
    if not val_features.exists():
        missing.append(f"Val features: {val_features}")
    if not val_meta.exists():
        missing.append(f"Val metadata: {val_meta}")

    if missing:
        print(f"❌ Missing prerequisites:")
        for m in missing:
            print(f"   - {m}")
        return

    from core.guardian_calib import run_calibration
    calibration_data = run_calibration()

    print(f"\n{Colors.GREEN}✓ Calibration complete!{Colors.RESET}")
    print(f"  {calibration_data}")

# ==================== STEP 6: THRESHOLD DISCOVERY (UPDATED) ====================

def step_6_threshold_discovery():
    """
    Step 6: Threshold discovery.
    UPDATED: Now uses memmap format from Step 3 directly.
    """
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 6: THRESHOLD DISCOVERY{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    # Check for Step 3 memmap files
    required_files = [
        ARTIFACTS_DIR / "val_features.bin",
        ARTIFACTS_DIR / "val_metadata.pkl",
        ARTIFACTS_DIR / "scaler.pkl",
    ]

    # Check for model (multiple possible names)
    model_files = [
        "guardian_spider_native.pth",
        "guardian_moe_final.pth",
        "guardian_best.pth",
        "guardian_final.pth"
    ]

    model_path = None
    for fname in model_files:
        candidate = ARTIFACTS_DIR / fname
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        required_files.append(ARTIFACTS_DIR / "guardian_*.pth (any model file)")

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"❌ Missing prerequisites:")
        for m in missing:
            print(f"   - {m}")
        return

    print(f"✅ Found model: {model_path.name}")

    # Set environment for dtype safety
    os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
    os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["TORCH_DTYPE"] = "float32"

    # Import and run updated threshold discovery
    try:
        from utils.threshold_discovery_features import discover_thresholds
        discover_thresholds()
        print(f"\n{Colors.GREEN}✓ Threshold discovery complete!{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Threshold discovery failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

# ==================== STEP 7: EVALUATION (UPDATED) ====================

def step_7_evaluation():
    """
    Step 7: Stress testing.
    UPDATED: Dtype safety with Unsloth patches disabled.
    """
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 7: SPIDER-NATIVE STRESS TEST{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    model_path = ARTIFACTS_DIR / "guardian_spider_native.pth"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}. Run step 4 first.")
        return

    # Set environment for dtype safety BEFORE importing tester
    os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
    os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["TORCH_DTYPE"] = "float32"

    # Force fp32 globally
    torch.set_default_dtype(torch.float32)

    from utils.stress_test_extended import run_extended_stress

    try:
        run_extended_stress()
        print(f"\n{Colors.GREEN}✓ Stress test complete!{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Stress test failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

# ==================== STEP 8: FULL DEMO ====================

def step_8_demo():
    """Step 8: Full evaluation suite."""
    print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.CYAN}🕷️ STEP 8: FULL GUARDIAN EVALUATION SUITE{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    required_files = [
        ARTIFACTS_DIR / "guardian_spider_native.pth",
        ARTIFACTS_DIR / "calib_temp.json",
        ARTIFACTS_DIR / "thresholds_features.json"
    ]

    missing = [f for f in required_files if not f.exists()]
    if missing:
        print(f"❌ Missing prerequisites:")
        for m in missing:
            print(f"   - {m.name}")
        return

    # Set environment for dtype safety
    os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
    os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
    os.environ["TORCH_DTYPE"] = "float32"
    torch.set_default_dtype(torch.float32)

    from utils.guardian_eval_runner import run_full_evaluation

    try:
        run_full_evaluation()
        print(f"\n{Colors.GREEN}✓ Full evaluation complete!{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Full evaluation failed: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

def validate_step_order(step_num):
    """Validate that prerequisites for step exist."""
    required_files = {
        1: [],
        2: [ARTIFACTS_DIR / "01_raw_data.pkl"],
        3: [
            ARTIFACTS_DIR / "02_train.pkl",
            ARTIFACTS_DIR / "02_val.pkl",
            ARTIFACTS_DIR / "02_test.pkl"
        ],
        4: [
            ARTIFACTS_DIR / "02_train.pkl",
            ARTIFACTS_DIR / "02_val.pkl",
            ARTIFACTS_DIR / "train_features.bin",
            ARTIFACTS_DIR / "val_features.bin"
        ],
        5: [
            ARTIFACTS_DIR / "guardian_spider_native.pth",
            ARTIFACTS_DIR / "val_features.bin",
            ARTIFACTS_DIR / "val_metadata.pkl"
        ],
        6: [
            ARTIFACTS_DIR / "val_features.bin",
            ARTIFACTS_DIR / "val_metadata.pkl",
            ARTIFACTS_DIR / "scaler.pkl"
        ],
        7: [ARTIFACTS_DIR / "guardian_spider_native.pth"],
        8: [
            ARTIFACTS_DIR / "guardian_spider_native.pth",
            ARTIFACTS_DIR / "calib_temp.json",
            ARTIFACTS_DIR / "thresholds_features.json"
        ]
    }

    missing = []
    for required in required_files.get(step_num, []):
        if not required.exists():
            # For step 6, check for any model file
            if step_num == 6 and "guardian" in str(required) and "*.pth" in str(required):
                model_exists = any(
                    (ARTIFACTS_DIR / fname).exists() for fname in ["guardian_spider_native.pth", "guardian_moe_final.pth", "guardian_best.pth", "guardian_final.pth"]
                )
                if model_exists:
                    continue
            missing.append(required)
    return missing

# ==================== NEW: Helper to convert a CSV/JSON subset into 02_train.pkl ====================

def convert_subset_to_train_pickle(subset_path: Path, artifacts_dir: Path, key: str = "label"):
    """
    Convert a CSV/JSON subset into ARTIFACTS_DIR/02_train.pkl and return the Path.
    - If ARTIFACTS_DIR/01_raw_data.pkl exists, merge/replace rows by `key`.
    - Otherwise create 02_train.pkl directly from the subset.
    """
    import json, csv, pickle
    subset_path = Path(subset_path)
    artifacts_dir = Path(artifacts_dir)
    raw_path = artifacts_dir / "01_raw_data.pkl"
    out_path = artifacts_dir / "02_train.pkl"

    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")

    # Load subset rows
    suffix = subset_path.suffix.lower()
    subset_rows = []
    if suffix in [".csv", ".tsv"]:
        try:
            import pandas as pd
            df = pd.read_csv(subset_path)
            subset_rows = df.to_dict(orient="records")
        except Exception:
            with open(subset_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                subset_rows = [dict(r) for r in reader]
    elif suffix in [".json", ".ndjson"]:
        with open(subset_path, "r", encoding="utf-8") as f:
            try:
                subset_rows = json.load(f)
            except Exception:
                subset_rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Unsupported subset file format. Use CSV or JSON.")

    # Minimal normalization
    normalized = []
    for r in subset_rows:
        if not isinstance(r, dict):
            r = dict(r)
        if "question" not in r and "prompt" in r:
            r["question"] = r.get("prompt")
        if "answer" not in r and "target" in r:
            r["answer"] = r.get("target")
        if "label" in r:
            try:
                r["label"] = int(r["label"])
            except Exception:
                pass
        if "domain" not in r:
            r["domain"] = r.get("domain", "math")
        if "meta" not in r:
            r["meta"] = {"source": "subset", "category": r.get("domain", "Unknown")}
        normalized.append(r)

    # Merge with raw if present
    if raw_path.exists():
        try:
            with open(raw_path, "rb") as f:
                raw = pickle.load(f)
        except Exception:
            raw = []
        subset_map = {str(r.get(key, "")): r for r in normalized if key in r}
        if subset_map:
            out = []
            replaced = 0
            raw_keys = {str(s.get(key, "")) for s in raw}
            for s in raw:
                sid = str(s.get(key, ""))
                if sid in subset_map:
                    out.append(subset_map[sid])
                    replaced += 1
                else:
                    out.append(s)
            # append unmatched subset rows
            for k, r in subset_map.items():
                if k not in raw_keys:
                    out.append(r)
            with open(out_path, "wb") as f:
                pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
            return out_path
        else:
            # no key matches -> write normalized as train
            with open(out_path, "wb") as f:
                pickle.dump(normalized, f, protocol=pickle.HIGHEST_PROTOCOL)
            return out_path

    # No raw -> create train directly
    with open(out_path, "wb") as f:
        pickle.dump(normalized, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


# ==================== CLI / Main entrypoint ====================

def main():
    parser = argparse.ArgumentParser(description="Guardian pipeline master controller")
    parser.add_argument("--step", type=int, choices=list(range(1,9)), default=4, help="Pipeline step to run (1-8)")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")
    parser.add_argument("--count", type=int, default=9000, help="Base sample count for step 1")
    parser.add_argument("--math-synthetic-pairs", type=int, default=300)
    parser.add_argument("--use-adv-math", action="store_true")
    parser.add_argument("--adv-math-target", type=int, default=None)
    parser.add_argument("--no-auto-generate-adv", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training (forwarded to step 4)")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of data loader workers (forwarded to step 4)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from checkpoint")
    parser.add_argument("--reduce-math", action="store_true")
    parser.add_argument("--math-keep-ratio", type=float, default=1.0)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--llm", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    # NEW: accept a train-subset path (CSV or JSON) to be merged into ARTIFACTS_DIR/02_train.pkl before step 4
    parser.add_argument("--train-subset", type=str, default=None, help="Path to CSV/JSON subset to use as training split (merged into ARTIFACTS_DIR/02_train.pkl)")
    args = parser.parse_args()

    # Apply seed override\n    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print(f"\n🔁 Overriding SEED -> {SEED}")

    # Record LLM name in environment if provided
    if args.llm:
        os.environ["AB_LLM"] = args.llm
        print(f"🔁 Recording LLM name in env: AB_LLM={args.llm}")

    # If output-dir override provided, use it
    output_dir_override = None
    if args.output_dir:
        output_dir_override = Path(args.output_dir)
        print(f"🔁 Using explicit output directory override: {args.output_dir}")

    # If user requested to run all steps, run them in order
    if args.all:
        # Step 1
        step_1_generation(count=args.count, math_synthetic_pairs=args.math_synthetic_pairs,
                          use_adv_math=args.use_adv_math, adv_math_target=args.adv_math_target,
                          reduce_math=args.reduce_math, math_keep_ratio=args.math_keep_ratio)
        # Step 2
        step_2_splitting()
        # Step 3
        step_3_extraction()
        # Step 4
        # If train-subset provided, convert it into ARTIFACTS_DIR/02_train.pkl before training
        if args.train_subset:
            try:
                convert_subset_to_train_pickle(Path(args.train_subset), ARTIFACTS_DIR)
            except Exception as e:
                print(f"❌ Failed to convert train-subset: {e}")
                raise
        step_4_training(batch_size=args.batch_size, num_workers=args.num_workers, resume=args.resume, output_dir_override=output_dir_override)
        # Steps 5-8
        step_5_calibration()
        step_6_threshold_discovery()
        step_7_evaluation()
        step_8_demo()
        return

    # Run a single requested step
    step = args.step

    # If step 4 is requested and a train-subset is provided, convert it first
    if step == 4 and args.train_subset:
        try:
            convert_subset_to_train_pickle(Path(args.train_subset), ARTIFACTS_DIR)
        except Exception as e:
            print(f"❌ Failed to convert train-subset: {e}")
            raise

    # Validate prerequisites for the requested step
    missing = validate_step_order(step)
    if missing:
        print(f"❌ Missing prerequisites for step {step}:")
        for m in missing:
            print(f"   - {m}")
        # For step 1 we allow running even if some artifacts missing; otherwise exit
        if step != 1:
            return

    # Dispatch steps
    if step == 1:
        step_1_generation(count=args.count, math_synthetic_pairs=args.math_synthetic_pairs,
                          use_adv_math=args.use_adv_math, adv_math_target=args.adv_math_target,
                          reduce_math=args.reduce_math, math_keep_ratio=args.math_keep_ratio)
    elif step == 2:
        step_2_splitting()
    elif step == 3:
        step_3_extraction()
    elif step == 4:
        step_4_training(batch_size=args.batch_size, num_workers=args.num_workers, resume=args.resume, output_dir_override=output_dir_override, experiment=args.experiment)
    elif step == 5:
        step_5_calibration()
    elif step == 6:
        step_6_threshold_discovery()
    elif step == 7:
        step_7_evaluation()
    elif step == 8:
        step_8_demo()
    else:
        print("Unknown step:", step)

if __name__ == "__main__":
    main()

