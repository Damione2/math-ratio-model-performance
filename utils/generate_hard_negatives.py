# generate_hard_negatives.py (v5 - Refined Mining with Exclusion)
import json
import random
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# 1. SETUP PATHS FIRST 
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path: # Now 'sys' will be recognized
    sys.path.append(str(ROOT_DIR))

# 2. NOW IMPORT PROJECT MODULES
try:
    from core import guardian_tester_moe as tester
except ImportError:
    import guardian_tester_moe as tester

try:
    from config import DATA_DIR
except ImportError:
    # Fallback if config isn't found
    DATA_DIR = ROOT_DIR / "data"

# --- Configuration ---
TARGET_HARD_NEGATIVES = 500
CONFUSION_WINDOW = (0.2, 0.8)
# Use DATA_DIR to ensure we are looking on the D: drive
MERGED_FILE = DATA_DIR / "llm_adv_merged.jsonl"        
OUTPUT_FILE = DATA_DIR / "llm_adv_hard_negatives.jsonl"  
KNOWN_HARD_FILE = DATA_DIR / "known_hard_negatives.jsonl"
BATCH_SIZE = 64

def load_merged_samples(file_path: Path):
    """Load all samples from merged JSONL, return as list of dicts."""
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found! Run merge_adv_datasets.py first.")
    
    samples = []
    print(f"📂 Loading samples from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                # Ensure required keys exist
                if "question" not in sample or "answer" not in sample:
                    continue
                samples.append(sample)
            except json.JSONDecodeError:
                print(f"⚠️  Invalid JSON at line {line_num}, skipping.")
                continue
    
    print(f"   Loaded {len(samples)} samples.")
    return samples

def load_known_hard(file_path: Path):
    known = set()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        key = (item.get("question", ""), item.get("answer", ""))
                        known.add(key)
                    except json.JSONDecodeError:
                        continue
    print(f"Loaded {len(known)} known hard negatives for exclusion.")
    return known

def main():
    print("🧠 Loading Guardian Model and resources...")
    tester.load_all_resources()
    
    extractor = tester._RESOURCES["extractor"]
    scaler = tester._RESOURCES["scaler"]
    model = tester._RESOURCES["model"]
    device = tester._RESOURCES["device"]

    # Load all adversarial samples
    all_samples = load_merged_samples(MERGED_FILE)
    if len(all_samples) == 0:
        print("❌ No samples loaded. Exiting.")
        return

    # Load known hard for exclusion
    known_hard = load_known_hard(KNOWN_HARD_FILE)

    # Shuffle for randomness
    random.shuffle(all_samples)
    sample_stream = all_samples.copy()

    hard_negatives_found = 0
    attempts = 0

    pbar = tqdm(total=TARGET_HARD_NEGATIVES, desc="Mining", unit="hard neg")

    domain_map = {0: "math", 1: "code", 2: "real_world"}

    while hard_negatives_found < TARGET_HARD_NEGATIVES:
        if not sample_stream:
            sample_stream = all_samples.copy()
            random.shuffle(sample_stream)
            attempts += len(all_samples)

        batch_samples = []
        for _ in range(BATCH_SIZE):
            if sample_stream:
                candidate = sample_stream.pop()
                key = (candidate.get("question", ""), candidate.get("answer", ""))
                if key not in known_hard:
                    batch_samples.append(candidate)
            else:
                break

        if not batch_samples:
            print("[WARN] No more unseen samples available. Stopping early.")
            break

        attempts += len(batch_samples)

        try:
            # 1. Create a list of tuples (Question, Answer)
            qa_pairs = [(s['question'], s['answer']) for s in batch_samples]
            
            # 2. Unzip those pairs into two separate lists
            batch_qs, batch_as = zip(*qa_pairs) 
            
            # 3. Pass the lists to the extractor
            raw_features = extractor.extract(list(batch_qs), list(batch_as))

            # 1. Reshape to (Batch * Layers, 1536) so the scaler sees the 1536 it expects
            reshaped_for_scaler = raw_features.reshape(-1, 1536)

            # 2. Transform the features
            scaled_flat = scaler.transform(reshaped_for_scaler)

            # 3. Reshape back to the (Batch, 28, 1536) format the model needs
            scaled_features = scaled_flat.reshape(-1, 28, 1536)

            features_tensor = torch.tensor(scaled_features, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits, gate_logits = model(features_tensor)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            hallu_probs = probs[:, 1]

            if gate_logits is not None:
                gate_probs = torch.softmax(gate_logits, dim=-1).cpu().numpy()
                domain_preds = np.argmax(gate_probs, axis=1)
            else:
                domain_preds = [-1] * len(batch_samples)

            for i, sample in enumerate(batch_samples):
                prob = float(hallu_probs[i])
                if CONFUSION_WINDOW[0] <= prob <= CONFUSION_WINDOW[1]:
                    sample_copy = sample.copy()
                    sample_copy["mining_prob"] = prob
                    sample_copy["guardian_status"] = "HALLUCINATION" if prob > 0.5 else "VALID"
                    sample_copy["guardian_domain"] = domain_map.get(domain_preds[i], "unknown")

                    key = (sample.get("question", ""), sample.get("answer", ""))

                    # Append to both output and known
                    with open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                        f_out.write(json.dumps(sample_copy) + "\n")
                    with open(KNOWN_HARD_FILE, "a", encoding="utf-8") as f_known:
                        f_known.write(json.dumps(sample_copy) + "\n")

                    known_hard.add(key)  # Update in-memory set

                    hard_negatives_found += 1
                    pbar.update(1)

                    if hard_negatives_found >= TARGET_HARD_NEGATIVES:
                        break

        except Exception as e:
            print(f"[ERROR] Batch failed (attempts: {attempts}): {e}")

        if attempts % (BATCH_SIZE * 10) == 0:
            rate = (hard_negatives_found / attempts * 100) if attempts > 0 else 0
            print(f"[PROGRESS] Attempts: {attempts} | Found: {hard_negatives_found}/{TARGET_HARD_NEGATIVES} | Yield: {rate:.3f}%")

        if hard_negatives_found >= TARGET_HARD_NEGATIVES:
            break

    pbar.close()
    final_rate = (hard_negatives_found / attempts * 100) if attempts > 0 else 0
    print(f"\n✅ Mining Complete!")
    print(f"Found {hard_negatives_found} new unseen hard negatives from {attempts} attempts.")
    print(f"Final Yield Rate: {final_rate:.2f}%")
    print(f"New hard negatives appended to: {OUTPUT_FILE}")
    print(f"Known hard set updated in: {KNOWN_HARD_FILE}")

if __name__ == "__main__":
    main()