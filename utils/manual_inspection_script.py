# manual_inspection_script.py
"""
Manual Inspection Script for 100 Mixed Test Examples

This script:
1. Loads the test split (03_test_features.pkl from pipeline_artifacts)
2. Randomly samples 100 examples (reproducible with seed 42)
3. Runs full inference using the trained Guardian model (with calibration)
4. Outputs a readable JSON with:
   - Question & Answer
   - True domain and label
   - Predicted domain, status (VALID/HALLUCINATION), hallucination probability, gate confidence
5. Prints a summary of accuracy on these 100 samples

Run this after a full pipeline execution (steps 1-6 completed).
"""

import random
import json
import torch
import numpy as np
from pathlib import Path

# Add root to path to find core and config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules from the core package
from core.guardian_tester_moe import load_all_resources, _RESOURCES
from core.guardian_vision_core import GuardianVisionNet
from core.guardian_utils import GuardianScaler

# Import centralized paths from root config
try:
    from config import ARTIFACTS_DIR
except ImportError:
    # Fallback for manual runs
    ARTIFACTS_DIR = Path(__file__).parent.parent / "pipeline_artifacts"

# File paths now use the centralized ARTIFACTS_DIR (usually on D: drive)
TEST_PKL = ARTIFACTS_DIR / "03_test_features.pkl"
CALIB_JSON = ARTIFACTS_DIR / "calib_temp.json"
OUTPUT_JSON = ARTIFACTS_DIR / "manual_inspection_100_results.json"

# Domain mapping (consistent with pipeline)
DOMAIN_MAP = {0: "math", 1: "code", 2: "real_world"}

def main():
    print("Loading resources and model...")
    load_all_resources()  # Loads extractor, scaler, model
    model = _RESOURCES["model"]
    scaler = _RESOURCES["scaler"]
    extractor = _RESOURCES["extractor"]
    device = _RESOURCES["device"]

    # Load calibration temperatures
    with open(CALIB_JSON, "r", encoding="utf-8") as f:
        temps = json.load(f)["temperatures"]
    print(f"Loaded temperatures: {temps}")

    # Load test data
    print(f"Loading test data from {TEST_PKL}...")
    with open(TEST_PKL, "rb") as f:
        test_data = pickle.load(f)
    print(f"Loaded {len(test_data)} test samples.")

    # Sample 100 mixed examples (reproducible)
    random.seed(42)
    sampled = random.sample(test_data, 100)
    print("Sampled 100 examples.")

    results = []
    correct = 0

    print("Running inference on 100 samples...")
    for i, item in enumerate(sampled):
        q = item["question"]
        a = item["answer"]
        true_label = item["label"]  # 0 = VALID, 1 = HALLUCINATION
        true_domain = item["domain"]

        # Extract features
        feat = extractor.extract([q], [a])[0]  # (28, 1536)

        # Normalize
        flat = feat.reshape(-1, 1536)
        scaled = scaler.transform(flat)
        tensor = torch.tensor(scaled.reshape(1, 28, 1536), dtype=torch.float32, device=device)

        # Inference
        with torch.no_grad():
            logits, gate_logits = model(tensor)
            logits_np = logits.cpu().numpy().flatten()
            gate_probs = torch.softmax(gate_logits, dim=-1).cpu().numpy().flatten()
            gate_pred = int(np.argmax(gate_probs))
            gate_conf = float(np.max(gate_probs))

        pred_domain = DOMAIN_MAP.get(gate_pred, "unknown")

        # Temperature scaling
        temp = temps.get(pred_domain, 1.0)
        scaled_logits = logits_np / temp
        probs = torch.softmax(torch.tensor(scaled_logits), dim=-1).numpy()
        hallu_prob = float(probs[1])
        pred_label = 1 if hallu_prob > 0.5 else 0
        pred_status = "HALLUCINATION" if pred_label == 1 else "VALID"

        is_correct = pred_label == true_label
        if is_correct:
            correct += 1

        results.append({
            "index": i + 1,
            "question": q,
            "answer": a,
            "true_domain": true_domain,
            "true_label": "HALLUCINATION" if true_label == 1 else "VALID",
            "predicted_domain": pred_domain,
            "predicted_status": pred_status,
            "hallucination_probability": round(hallu_prob, 4),
            "gate_confidence": round(gate_conf, 4),
            "correct": is_correct
        })

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100")

    # Save results
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    accuracy = correct / 100
    print("\n" + "="*50)
    print("MANUAL INSPECTION COMPLETE")
    print(f"Results saved to: {OUTPUT_JSON}")
    print(f"Accuracy on these 100 samples: {accuracy:.1%} ({correct}/100)")
    print("="*50)

    # Print a few examples
    print("\nFirst 3 examples:")
    for res in results[:3]:
        print(f"\n--- #{res['index']} ---")
        print(f"Domain: {res['true_domain']} → Predicted: {res['predicted_domain']}")
        print(f"True: {res['true_label']} → Pred: {res['predicted_status']} (Prob: {res['hallucination_probability']})")
        print(f"Q: {res['question']}")
        print(f"A: {res['answer'][:300]}...")

if __name__ == "__main__":
    import pickle  # Ensure pickle is imported
    main()