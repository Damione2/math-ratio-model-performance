#!/usr/bin/env python3
"""
validate_and_record.py

Usage:
  python validate_and_record.py --run-dir experiments/Qwen2.5-Math-1.5B/math_25_seed0

Checks expected artifacts and appends a summary row to experiments/ablation_summary.json.
"""

import argparse
import json
from pathlib import Path
import sys
import datetime

EXPECTED_ARTIFACTS = [
    "training_summary.json",
    "training_log.csv",
    "guardian_spider_native.pth"
]

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Path to the run output directory")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run-dir not found: {run_dir}")
        return 2

    # Check artifacts
    missing = []
    for fname in EXPECTED_ARTIFACTS:
        if not (run_dir / fname).exists():
            missing.append(fname)
    if missing:
        print(f"ERROR: Missing artifacts in {run_dir}: {missing}")
        # still proceed to record partial info, but return non-zero
        status_code = 3
    else:
        status_code = 0

    # Load training_summary.json if present
    training_summary = load_json(run_dir / "training_summary.json")
    manifest = load_json(run_dir / "run_manifest.json")

    record = {
        "run_dir": str(run_dir.resolve()),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "status": "partial" if missing else "ok"
    }

    # Prefer training_summary fields
    if training_summary:
        record.update({
            "best_f1": training_summary.get("best_f1"),
            "best_epoch": training_summary.get("best_epoch"),
            "final_epoch": training_summary.get("final_epoch"),
            "math_ratio_observed": training_summary.get("math_ratio")
        })
    elif manifest:
        # fallback to manifest
        record.update({
            "best_f1": None,
            "best_epoch": None,
            "final_epoch": None,
            "math_ratio_observed": manifest.get("math_ratio")
        })
    else:
        record.update({
            "best_f1": None,
            "best_epoch": None,
            "final_epoch": None,
            "math_ratio_observed": None
        })

    # Parse model/seed from path or manifest
    # Expect path like experiments/<modelName>/math_<ratio>_seed<seed>
    parts = run_dir.parts
    try:
        model_name = parts[-3]
        run_name = parts[-1]
    except Exception:
        model_name = manifest.get("llm") if manifest else None
        run_name = run_dir.name

    record["model"] = model_name
    record["run_name"] = run_name

    # Append to experiments/ablation_summary.json
    summary_path = Path("experiments") / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing summary
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    else:
        data = []

    data.append(record)
    summary_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Recorded run summary to {summary_path}")

    return status_code

if __name__ == "__main__":
    sys.exit(main())
