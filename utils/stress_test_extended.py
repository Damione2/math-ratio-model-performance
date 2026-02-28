#!/usr/bin/env python3
"""
utils/stress_test_extended.py — Option A Refactor + Threshold Integration + Dtype Safety
--------------------------------------------------------------------------------------
This version uses:
  - diagnose_sample() as the ONLY inference path
  - calib_temp.json for temperature scaling
  - thresholds_features.json for per-domain thresholds
  - FIXED: Unpacks 4 values from diagnose_sample (was 3)
  - FIXED: Dtype safety with Unsloth patches disabled

This is the correct evaluation pipeline for the new Guardian architecture.
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix

# Disable Unsloth patches BEFORE importing Guardian components
os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_DTYPE"] = "float32"

from config import ARTIFACTS_DIR

# Import after setting env vars
from core.guardian_tester_moe import diagnose_sample, load_all_resources

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
SEED = 42
DOMAIN_CONFIDENCE_THRESHOLD = 0.75

REPORT_JSON = ARTIFACTS_DIR / "stress_test_spider_native_report.json"
REPORT_TXT = ARTIFACTS_DIR / "stress_test_spider_native_report.txt"

DOMAIN_MAP = {0: "math", 1: "code", 2: "real_world"}

# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def calculate_ece(probs, labels, n_bins=10):
    """Calculate Expected Calibration Error."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if mask.any():
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += abs(acc - conf) * (mask.sum() / len(probs))
    return float(ece)


# ---------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------
def get_test_suite():
    """Generate comprehensive test cases."""
    tests = []
    
    # Chain-of-Thought tests
    tests.extend([
        {
            "category": "Chain-of-Thought",
            "name": "cot_correct",
            "question": "Solve step-by-step: 23 + 19.",
            "answer": "23 + 19 = 42. Step 1: 23 + 10 = 33. Step 2: 33 + 9 = 42.",
            "true_label": 0
        },
        {
            "category": "Chain-of-Thought",
            "name": "cot_drift",
            "question": "Solve step-by-step: 23 + 19.",
            "answer": "23 + 19 = 42. Step 1: 23 + 10 = 33. Step 2: 33 + 9 = 42. Final: 43.",
            "true_label": 1
        },
    ])
    
    # Math subtle tests
    tests.extend([
        {
            "category": "Math-Subtle",
            "name": "subtraction_correct",
            "question": "What is 1001 - 1?",
            "answer": "1000",
            "true_label": 0
        },
        {
            "category": "Math-Subtle",
            "name": "subtraction_wrong",
            "question": "What is 1001 - 1?",
            "answer": "999",
            "true_label": 1
        },
    ])
    
    # Code syntax tests
    tests.extend([
        {
            "category": "Code-Syntax",
            "name": "listcomp_correct",
            "question": "Write list comprehension for squares 0-9.",
            "answer": "[x**2 for x in range(10)]",
            "true_label": 0
        },
        {
            "category": "Code-Syntax",
            "name": "listcomp_wrong",
            "question": "Write list comprehension for squares 0-9.",
            "answer": "[x^2 for x in range(10)]",
            "true_label": 1
        },
    ])
    
    # Real-world tests
    tests.extend([
        {
            "category": "Real-World",
            "name": "author_correct",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Jane Austen.",
            "true_label": 0
        },
        {
            "category": "Real-World",
            "name": "author_wrong",
            "question": "Who wrote 'Pride and Prejudice'?",
            "answer": "Charles Dickens.",
            "true_label": 1
        },
    ])
    
    # Math contamination tests (NEW)
    tests.extend([
        {
            "category": "Math-Contamination",
            "name": "clean_code",
            "question": "Write a Python function to greet a user.",
            "answer": "def greet(name):\n    return f'Hello, {name}!'",
            "true_label": 0
        },
        {
            "category": "Math-Contamination",
            "name": "contaminated_code",
            "question": "Write a Python function to calculate area.",
            "answer": "import math\ndef area(r):\n    return math.pi * r ** 2",
            "true_label": 1  # Math-heavy code may trigger hallucination
        },
        {
            "category": "Math-Contamination",
            "name": "externalized_code",
            "question": "Write a Python function to calculate area.",
            "answer": "def area(r):\n    return [MATH_TOOL](r)",  # Externalized math
            "true_label": 0
        },
    ])
    
    return tests


# ---------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------
def run_extended_stress():
    """Run extended stress test with proper error handling."""
    print("\n🧪 Running Extended Stress Test (Option A + Dtype Safe)...")
    print(f"{'='*60}")

    # Load tester resources (forces fp32)
    try:
        load_all_resources()
        print("✅ Resources loaded (fp32 mode)")
    except Exception as e:
        print(f"❌ Failed to load resources: {e}")
        raise

    # -----------------------------
    # Load calibration temperatures
    # -----------------------------
    calib_path = ARTIFACTS_DIR / "calib_temp.json"
    if calib_path.exists():
        calib = json.load(open(calib_path, "r"))
        temps = calib.get("temperatures", {d: 1.0 for d in DOMAIN_MAP.values()})
        print(f"✅ Loaded calibration temps: {temps}")
    else:
        temps = {d: 1.0 for d in DOMAIN_MAP.values()}
        print(f"⚠️ No calibration found, using default temps")

    # -----------------------------
    # Load thresholds (NEW)
    # -----------------------------
    thresh_path = ARTIFACTS_DIR / "thresholds_features.json"
    if thresh_path.exists():
        thresh_data = json.load(open(thresh_path, "r"))
        domain_thresholds = thresh_data.get("thresholds", {})
        global_thresh = thresh_data.get("global_threshold", 0.45)
        print(f"✅ Loaded thresholds from {thresh_path}")
        print(f"   Per-domain: {domain_thresholds}")
        print(f"   Global: {global_thresh}")
    else:
        print("⚠️ thresholds_features.json not found — using fallback 0.45")
        domain_thresholds = {"math": 0.45, "code": 0.45, "real_world": 0.45}
        global_thresh = 0.45

    # -----------------------------
    # Run test suite
    # -----------------------------
    tests = get_test_suite()
    results = []
    
    print(f"\n📝 Running {len(tests)} test cases...")

    y_true_all, y_pred_all, y_prob_all = [], [], []
    category_buckets = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})

    for idx, test in enumerate(tests, 1):
        q, a = test["question"], test["answer"]
        true_label = test["true_label"]
        category = test["category"]

        try:
            # Use diagnose_sample() as the ONLY inference path
            # FIXED: Unpack 4 values (was 3)
            status, conf, domain, diagnostics = diagnose_sample(q, a)
        except Exception as e:
            print(f"❌ Test {idx} failed: {e}")
            # Log error but continue
            results.append({
                "index": idx,
                "category": category,
                "name": test["name"],
                "question": q,
                "answer": a,
                "true_label": true_label,
                "predicted_label": -1,
                "prob_hallucination": 0.0,
                "domain_assigned": "error",
                "threshold_used": 0.45,
                "is_correct": False,
                "error": str(e),
                "diagnostics": {}
            })
            continue

        # Convert to hallucination probability
        prob_hallu = conf if status == "HALLUCINATION" else (1 - conf)

        # Threshold selection
        threshold = domain_thresholds.get(domain, global_thresh)

        pred_label = int(prob_hallu > threshold)

        # Collect metrics
        y_true_all.append(true_label)
        y_pred_all.append(pred_label)
        y_prob_all.append(prob_hallu)

        category_buckets[category]["y_true"].append(true_label)
        category_buckets[category]["y_pred"].append(pred_label)
        category_buckets[category]["y_prob"].append(prob_hallu)

        # Store result with full diagnostics
        results.append({
            "index": idx,
            "category": category,
            "name": test["name"],
            "question": q,
            "answer": a,
            "true_label": true_label,
            "predicted_label": pred_label,
            "prob_hallucination": round(prob_hallu, 4),
            "domain_assigned": domain,
            "threshold_used": round(threshold, 4),
            "is_correct": pred_label == true_label,
            "diagnostics": diagnostics
        })
        
        # Progress indicator
        status_icon = "✅" if pred_label == true_label else "❌"
        print(f"  {status_icon} Test {idx:2d}/{len(tests)}: {test['name']:20s} | "
              f"Domain: {domain:8s} | Pred: {pred_label} | True: {true_label}")

    # Summary statistics
    def calc_metrics(y_true, y_pred, y_prob):
        """Calculate metrics safely."""
        if len(y_true) == 0:
            return {"accuracy": 0.0, "f1": 0.0, "ece": 0.0}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        
        accuracy = float(np.mean(y_true == y_pred))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        ece = calculate_ece(y_prob, y_true) if len(y_prob) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "f1": f1,
            "ece": ece,
        }

    # Build report
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "n_samples": len(results),
        "n_failed": sum(1 for r in results if r.get("error")),
        "calibration_temps": temps,
        "thresholds": domain_thresholds,
        "global_threshold": global_thresh,
        "summary": calc_metrics(y_true_all, y_pred_all, y_prob_all),
        "per_category": {},
        "individual_results": results,
    }
    
    # Add confusion matrix if we have valid predictions
    if len(y_true_all) > 0 and all(y != -1 for y in y_pred_all):
        report["confusion_matrix"] = confusion_matrix(
            y_true_all, y_pred_all, labels=[0, 1]
        ).tolist()

    # Per-category metrics
    for cat, bucket in category_buckets.items():
        if len(bucket["y_true"]) > 0:
            report["per_category"][cat] = {
                "n_samples": len(bucket["y_true"]),
                **calc_metrics(bucket["y_true"], bucket["y_pred"], bucket["y_prob"]),
            }

    # Save JSON report
    json.dump(report, open(REPORT_JSON, "w"), indent=2)

    # Generate text report
    with open(REPORT_TXT, "w", encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("EXTENDED STRESS TEST REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Samples: {report['n_samples']} (Failed: {report['n_failed']})\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy: {report['summary']['accuracy']:.3f}\n")
        f.write(f"  F1 Score: {report['summary']['f1']:.3f}\n")
        f.write(f"  ECE:      {report['summary']['ece']:.3f}\n\n")
        
        f.write("PER-CATEGORY METRICS:\n")
        for cat, metrics in report["per_category"].items():
            f.write(f"  {cat:20s}: Acc={metrics['accuracy']:.3f}, "
                   f"F1={metrics['f1']:.3f}, N={metrics['n_samples']}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("="*60 + "\n")
        for r in results:
            if "error" in r:
                f.write(f"\n❌ {r['name']}: ERROR - {r['error']}\n")
            else:
                status = "✅" if r["is_correct"] else "❌"
                f.write(f"\n{status} {r['name']} ({r['category']})\n")
                f.write(f"   Q: {r['question']}\n")
                f.write(f"   A: {r['answer']}\n")
                f.write(f"   Domain: {r['domain_assigned']} | "
                       f"Prob: {r['prob_hallucination']:.3f} | "
                       f"Threshold: {r['threshold_used']}\n")
                if r["diagnostics"]:
                    f.write(f"   Diagnostics: {r['diagnostics']}\n")

    print(f"\n{'='*60}")
    print("STRESS TEST COMPLETE")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {report['summary']['accuracy']:.3f}")
    print(f"Overall F1:       {report['summary']['f1']:.3f}")
    print(f"Overall ECE:      {report['summary']['ece']:.3f}")
    print(f"\nReports saved:")
    print(f"  JSON: {REPORT_JSON}")
    print(f"  TXT:  {REPORT_TXT}")


if __name__ == "__main__":
    run_extended_stress()