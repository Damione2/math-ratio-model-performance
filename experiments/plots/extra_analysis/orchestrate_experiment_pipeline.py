#!/usr/bin/env python3
"""
experiments/plots/extra_analysis/orchestrate_experiment_pipeline.py

Orchestrate pilot runs using pipeline/guardian_pipeline_master.py.

Behavior:
- Imports ARTIFACTS_DIR from config.py and passes it to merge_and_run.py
- For each run: merges matched subset into ARTIFACTS_DIR/02_train.pkl, runs the pipeline (step 4),
  captures stdout/stderr into runs/run_{run_id}/stdout.txt and stderr.txt, and extracts best_f1.
- Writes seed_assignments.csv and experiment_results.csv under experiments/plots/extra_analysis.

Now fully supports --batch-size and --num-workers (forwarded all the way down to the trainer).

Usage:
  python -m experiments.plots.extra_analysis.orchestrate_experiment_pipeline \
    --n-per-arm 10 --batch-size 512 --num-workers 0
"""

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

# Import ARTIFACTS_DIR from project config so we always use the canonical path
from config import ARTIFACTS_DIR as CONFIG_ARTIFACTS_DIR

OUT = Path("experiments/plots/extra_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# LLM name recorded in manifest / results
LLM_NAME = "Qwen2.5-Math-1.5B"

# Path to the merge-and-run helper (must exist)
MERGE_AND_RUN = Path("experiments/plots/extra_analysis/merge_and_run.py").as_posix()


def extract_best_f1_from_text(text):
    """Look for a line like: BEST_F1 0.81234"""
    for line in text.splitlines()[::-1]:
        if "BEST_F1" in line:
            parts = line.strip().split()
            try:
                return float(parts[-1])
            except:
                continue
    return None


def run_merge_and_pipeline(
    matched_csv: str,
    seed: int,
    run_id: str,
    llm: str,
    artifacts_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """
    Calls merge_and_run.py which:
      - writes ARTIFACTS_DIR/02_train.pkl (merged train split)
      - runs pipeline/guardian_pipeline_master.py --step 4
      - writes stdout/stderr to runs/run_{run_id}/stdout.txt and stderr.txt
    """
    run_dir = Path("runs") / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f'python "{MERGE_AND_RUN}" '
        f'--matched-csv "{matched_csv}" '
        f'--seed {seed} --run-id {run_id} '
        f'--artifacts-dir "{artifacts_dir}" '
        f'--batch-size {batch_size} '
        f'--num-workers {num_workers} '
        f'--llm "{llm}"'
    )
    print("Executing:", cmd)

    # FIXED: Use encoding='utf-8' to handle UnicodeDecodeError on Windows
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'  # Replace any undecodable bytes to avoid crashes
    )

    # Prefer files written by merge_and_run.py if they exist
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    stdout_text = stdout_path.read_text(encoding='utf-8', errors='replace') if stdout_path.exists() else proc.stdout
    stderr_text = stderr_path.read_text(encoding='utf-8', errors='replace') if stderr_path.exists() else proc.stderr

    return proc.returncode, stdout_text, stderr_text


def main():
    p = argparse.ArgumentParser(description="Orchestrate Guardian pilot runs with full training flag support")
    p.add_argument("--n-per-arm", type=int, default=10)
    p.add_argument("--sleep-between", type=float, default=1.0)
    p.add_argument("--control-file", default=str(OUT / "control_examples.csv"))
    p.add_argument("--treatment-file", default=str(OUT / "treatment_examples.csv"))
    p.add_argument("--llm", default=LLM_NAME)

    # NEW: Training flags forwarded all the way to guardian_pipeline_master.py
    p.add_argument("--batch-size", type=int, default=16, help="Batch size for training (passed to guardian_pipeline_master)")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers (passed to guardian_pipeline_master)")

    args = p.parse_args()

    control = Path(args.control_file)
    treat = Path(args.treatment_file)
    if not control.exists() or not treat.exists():
        print("ERROR: control or treatment file missing. Run prepare_symbol_experiment.py first.")
        return

    total = args.n_per_arm * 2
    seeds = random.sample(range(10000, 10000 + total * 10), total)
    arms = ["Control"] * args.n_per_arm + ["Treatment"] * args.n_per_arm
    random.shuffle(arms)

    assignments = []
    for i, (seed, arm) in enumerate(zip(seeds, arms)):
        run_id = f"{int(time.time())}_{i}"
        subset = str(control) if arm == "Control" else str(treat)
        outdir = str(Path("runs") / f"run_{run_id}")
        assignments.append({
            "run_id": run_id,
            "seed": seed,
            "arm": arm,
            "subset": subset,
            "outdir": outdir
        })

    # Save assignments
    with open(OUT / "seed_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "arm", "subset", "outdir"])
        writer.writeheader()
        writer.writerows(assignments)

    results = []
    for a in assignments:
        # Step 1: merge matched subset + run pipeline step 4 with custom batch size / workers
        rc, stdout_text, stderr_text = run_merge_and_pipeline(
            matched_csv=a["subset"],
            seed=a["seed"],
            run_id=a["run_id"],
            llm=args.llm,
            artifacts_dir=str(CONFIG_ARTIFACTS_DIR),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        # Print a short tail of stderr/stdout for quick debugging
        if stderr_text:
            print(f"--- stderr (tail) for run {a['run_id']} ---")
            print(stderr_text[-2000:])
        if stdout_text:
            print(f"--- stdout (tail) for run {a['run_id']} ---")
            print(stdout_text[-2000:])

        # Try to extract best_f1
        best_f1 = extract_best_f1_from_text(stdout_text or "") or extract_best_f1_from_text(stderr_text or "")
        if best_f1 is None:
            metrics_path = Path(a["outdir"]) / "metrics.json"
            if metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text(encoding='utf-8'))
                    best_f1 = m.get("best_f1", None)
                except:
                    best_f1 = None

        results.append({
            "run_id": a["run_id"],
            "seed": a["seed"],
            "arm": a["arm"],
            "model": args.llm,
            "best_f1": best_f1
        })

        if rc != 0:
            print(f"Run {a['run_id']} exited with code {rc}. Check runs/run_{a['run_id']}/stderr.txt for details.")

        time.sleep(args.sleep_between)

    # Save results
    with open(OUT / "experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "arm", "model", "best_f1"])
        writer.writeheader()
        writer.writerows(results)

    print("✅ Orchestration complete. Results saved to", OUT / "experiment_results.csv")


if __name__ == "__main__":
    main()
