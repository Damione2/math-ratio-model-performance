# utils/guardian_eval_runner.py
"""
Unified evaluation runner for Guardian.
Loads:
  - Manual tests
  - Synthetic tests
Applies:
  - diagnose_sample()
  - calibrated temperatures
  - per-domain thresholds
Outputs:
  - JSON report
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import f1_score, confusion_matrix

from config import ARTIFACTS_DIR
from core.guardian_tester_moe import diagnose_sample, load_all_resources

from utils.guardian_eval_manual import get_manual_tests
from utils.guardian_eval_synthetic import get_synthetic_tests



def calculate_ece(probs, labels, n_bins=10):
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


def run_full_evaluation():
    print("\n🧪 Running FULL Guardian Evaluation Suite…")

    load_all_resources()

    # Load calibration temps
    calib_path = ARTIFACTS_DIR / "calib_temp.json"
    temps = json.load(open(calib_path))["temperatures"]

    # Load thresholds
    thresh_path = ARTIFACTS_DIR / "thresholds_features.json"
    thresh_data = json.load(open(thresh_path))
    domain_thresholds = thresh_data["thresholds"]
    global_thresh = thresh_data["global_threshold"]

    # Load tests
    manual_tests = get_manual_tests()
    synthetic_tests = get_synthetic_tests()
    tests = manual_tests + synthetic_tests

    results = []
    y_true_all, y_pred_all, y_prob_all = [], [], []
    category_buckets = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_prob": []})

    for idx, test in enumerate(tests, 1):
        q, a = test["question"], test["answer"]
        true_label = test["true_label"]
        category = test["category"]

        # ✅ FIXED: Unpack 4 values from diagnose_sample (was 3)
        status, conf, domain, diagnostics = diagnose_sample(q, a)
        prob_hallu = conf if status == "HALLUCINATION" else (1 - conf)

        threshold = domain_thresholds.get(domain, global_thresh)
        pred_label = int(prob_hallu > threshold)

        y_true_all.append(true_label)
        y_pred_all.append(pred_label)
        y_prob_all.append(prob_hallu)

        category_buckets[category]["y_true"].append(true_label)
        category_buckets[category]["y_pred"].append(pred_label)
        category_buckets[category]["y_prob"].append(prob_hallu)

        results.append({
            "index": idx,
            "category": category,
            "question": q,
            "answer": a,
            "true_label": true_label,
            "predicted_label": pred_label,
            "prob_hallucination": round(prob_hallu, 4),
            "domain_assigned": domain,
            "threshold_used": round(threshold, 4),
            "is_correct": pred_label == true_label,
            "diagnostics": diagnostics,  # ✅ NEW: Include diagnostics in results
        })

    def calc_metrics(y_true, y_pred, y_prob):
        return {
            "accuracy": float(np.mean(np.array(y_true) == np.array(y_pred))),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "ece": calculate_ece(y_prob, y_true),
        }

    report = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": len(results),
        "calibration_temps": temps,
        "thresholds": domain_thresholds,
        "global_threshold": global_thresh,
        "summary": calc_metrics(y_true_all, y_pred_all, y_prob_all),
        "per_category": {},
        "individual_results": results,
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
    }

    for cat, bucket in category_buckets.items():
        report["per_category"][cat] = {
            "n_samples": len(bucket["y_true"]),
            **calc_metrics(bucket["y_true"], bucket["y_pred"], bucket["y_prob"]),
        }

    out_path = ARTIFACTS_DIR / "guardian_full_eval_report.json"
    json.dump(report, open(out_path, "w"), indent=2)

    print(f"\n📄 Full evaluation report saved to:\n  {out_path}")