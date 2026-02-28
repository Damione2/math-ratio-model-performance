#!/usr/bin/env python3
"""
run_ablation_experiment.py

Wrapper to run a single ablation run (one math ratio, one seed, one LLM)
by invoking pipeline.guardian_pipeline_master steps 1..4 with consistent args.

Usage:
    python run_ablation_experiment.py --math-ratio 0.25 --seed 0 --out experiments/Qwen2.5-Math-1.5B/math_25_seed0 --llm unsloth/Qwen2.5-Math-1.5B
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import datetime
import shutil

PYTHON = sys.executable

def call(cmd, cwd=None, env=None):
    print("\n>>>", cmd)
    rc = subprocess.call(cmd, shell=True, cwd=cwd, env=env)
    return rc

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    p = argparse.ArgumentParser(description="Run one ablation experiment (math ratio, seed, llm)")
    p.add_argument("--math-ratio", type=float, required=True, help="math ratio (0.0-1.0)")
    p.add_argument("--seed", type=int, required=True, help="random seed")
    p.add_argument("--out", type=str, required=True, help="output directory for this run")
    p.add_argument("--llm", type=str, required=True, help="LLM model name (HuggingFace or local path)")
    p.add_argument("--no-steps", action="store_true", help="dry-run: print commands but do not execute")
    args = p.parse_args()

    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # Basic manifest for reproducibility
    manifest = {
        "math_ratio": args.math_ratio,
        "seed": args.seed,
        "llm": args.llm,
        "out_dir": str(out_dir.resolve()),
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "python": PYTHON,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Common base command to call pipeline.guardian_pipeline_master
    # The pipeline module should accept the flags used below; adjust if your pipeline uses different names.
    # Step 1: dataset generation with fixed math ratio (no synthetic math / no adv)
    step1_cmd = (
        f'{PYTHON} -m pipeline.guardian_pipeline_master '
        f'--step 1 '
        f'--count 9000 '
        f'--math-synthetic-pairs 0 '
        f'--no-auto-generate-adv '
        f'--reduce-math '
        f'--math-keep-ratio {args.math_ratio} '
        f'--experiment {out_dir.name} '
        f'--seed {args.seed} '
        f'--llm {args.llm} '
        f'--output-dir "{out_dir}"'
    )

    # Step 2: preprocessing / feature extraction
    step2_cmd = (
        f'{PYTHON} -m pipeline.guardian_pipeline_master '
        f'--step 2 '
        f'--experiment {out_dir.name} '
        f'--seed {args.seed} '
        f'--llm {args.llm} '
        f'--output-dir "{out_dir}"'
    )

    # Step 3: build datasets / splits
    step3_cmd = (
        f'{PYTHON} -m pipeline.guardian_pipeline_master '
        f'--step 3 '
        f'--experiment {out_dir.name} '
        f'--seed {args.seed} '
        f'--llm {args.llm} '
        f'--output-dir "{out_dir}"'
    )

    # Step 4: training (no-resume to ensure fresh run)
    step4_cmd = (
        f'{PYTHON} -m pipeline.guardian_pipeline_master '
        f'--step 4 '
        f'--no-resume '
        f'--batch-size 512 '
        f'--experiment {out_dir.name} '
        f'--seed {args.seed} '
        f'--llm {args.llm} '
        f'--output-dir "{out_dir}"'
    )

    commands = [step1_cmd, step2_cmd, step3_cmd, step4_cmd]

    # If pipeline.guardian_pipeline_master is not present, try fallback to run_ablation_fixed_math.py
    module_path = Path(__file__).parent / "pipeline" / "guardian_pipeline_master.py"
    if not module_path.exists():
        print("Warning: pipeline.guardian_pipeline_master module not found at:", module_path)
        # fallback: try to call run_ablation_fixed_math.py if present (older pipeline)
        fallback = Path(__file__).parent / "run_ablation_fixed_math.py"
        if fallback.exists():
            print("Fallback: using run_ablation_fixed_math.py (note: this script may not accept --llm/--seed/out).")
            # call the fixed math script with the keep ratio only
            fallback_cmd = f'{PYTHON} "{fallback}"'
            # run fallback with environment variables to pass parameters
            os.environ["AB_EXPERIMENT_NAME"] = out_dir.name
            os.environ["AB_MATH_KEEP_RATIO"] = str(args.math_ratio)
            os.environ["AB_SEED"] = str(args.seed)
            if args.no_steps:
                print("Dry-run fallback:", fallback_cmd)
                return 0
            rc = call(fallback_cmd)
            if rc != 0:
                print("Fallback script failed with rc=", rc)
                return rc
            # After fallback, exit successfully (training likely done by fallback)
            manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return 0
        else:
            print("Error: neither pipeline.guardian_pipeline_master nor run_ablation_fixed_math.py found. Aborting.")
            return 2

    # Execute commands sequentially
    for i, cmd in enumerate(commands, start=1):
        print(f"\n--- Executing step {i} ---")
        if args.no_steps:
            print("Dry-run:", cmd)
            continue
        rc = call(cmd)
        if rc != 0:
            print(f"Step {i} failed with exit code {rc}. Aborting.")
            manifest["failed_step"] = i
            manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return rc

    # If we reach here, assume training completed. Try to collect summary files if present.
    # Common artifacts to look for (adjust names if your pipeline uses different filenames)
    artifacts = {}
    for fname in ["training_summary.json", "training_log.csv", "guardian_spider_native.pth", "guardian_spider_native_full.pth"]:
        candidate = out_dir / fname
        if candidate.exists():
            artifacts[fname] = str(candidate.resolve())

    manifest["artifacts"] = artifacts
    manifest["finished_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nRun completed successfully. Manifest written to:", out_dir / "run_manifest.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
