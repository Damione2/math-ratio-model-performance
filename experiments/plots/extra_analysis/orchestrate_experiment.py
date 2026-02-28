#!/usr/bin/env python3
"""
Orchestrate a randomized pilot for the symbol manipulation experiment.

What it does
- Reads treatment_examples.csv and control_examples.csv
- Randomly assigns seeds to Control and Treatment arms
- Launches training runs sequentially using a user-specified training command template
- Collects best_f1 per run from either stdout marker or a metrics file
- Writes experiments/plots/extra_analysis/experiment_results.csv

Usage
- Edit TRAIN_CMD_TEMPLATE to match your training CLI. Use placeholders:
    {seed}      -> integer seed
    {subset}    -> path to CSV subset (control or treatment)
    {run_id}    -> unique run id used for run directory
- Run:
    python experiments/plots/extra_analysis/orchestrate_experiment.py --n-per-arm 10
"""
import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path

OUT = Path("experiments/plots/extra_analysis")
OUT.mkdir(parents=True, exist_ok=True)

# EDIT THIS TEMPLATE to match your training command.
# Ensure the command writes a metrics file runs/run_{run_id}/metrics.json with key "best_f1"
# or prints a line containing "BEST_F1 <value>" to stdout.
TRAIN_CMD_TEMPLATE = "python train.py --seed {seed} --train-subset {subset} --out-dir runs/run_{run_id}"

def extract_best_f1_from_text(text):
    # Look for a line like: BEST_F1 0.81234
    for line in text.splitlines()[::-1]:
        if "BEST_F1" in line:
            parts = line.strip().split()
            try:
                return float(parts[-1])
            except:
                continue
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-arm", type=int, default=10)
    p.add_argument("--sleep-between", type=float, default=1.0)
    p.add_argument("--mode", choices=["append","mask"], default="append")
    p.add_argument("--control-file", default=str(OUT / "control_examples.csv"))
    p.add_argument("--treatment-file", default=str(OUT / "treatment_examples.csv"))
    p.add_argument("--train-cmd-template", default=TRAIN_CMD_TEMPLATE)
    args = p.parse_args()

    # Validate files
    control = Path(args.control_file)
    treat = Path(args.treatment_file)
    if not control.exists() or not treat.exists():
        print("ERROR: control or treatment file missing. Run prepare script first.")
        return

    total = args.n_per_arm * 2
    seeds = random.sample(range(10000, 10000 + total * 10), total)
    arms = ["Control"] * args.n_per_arm + ["Treatment"] * args.n_per_arm
    random.shuffle(arms)

    assignments = []
    for i, (seed, arm) in enumerate(zip(seeds, arms)):
        run_id = f"{int(time.time())}_{i}"
        subset = str(control) if arm == "Control" else str(treat)
        assignments.append({"run_id": run_id, "seed": seed, "arm": arm, "subset": subset})

    # Save assignments
    with open(OUT / "seed_assignments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "arm", "subset"])
        writer.writeheader()
        writer.writerows(assignments)

    results = []
    for a in assignments:
        run_id = a["run_id"]
        seed = a["seed"]
        subset = a["subset"]
        cmd = args.train_cmd_template.format(seed=seed, subset=subset, run_id=run_id)
        print("Running:", cmd)
        try:
            proc = subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out_text = proc.stdout + "\n" + proc.stderr
            best_f1 = extract_best_f1_from_text(out_text)
            # fallback: read metrics file
            metrics_path = Path(f"runs/run_{run_id}/metrics.json")
            if best_f1 is None and metrics_path.exists():
                try:
                    m = json.loads(metrics_path.read_text())
                    best_f1 = m.get("best_f1", None)
                except:
                    best_f1 = None
            results.append({"run_id": run_id, "seed": seed, "arm": a["arm"], "model": "Qwen2.5-Math-1.5B", "best_f1": best_f1})
        except subprocess.CalledProcessError as e:
            print("Run failed for", run_id, "seed", seed)
            results.append({"run_id": run_id, "seed": seed, "arm": a["arm"], "model": "Qwen2.5-Math-1.5B", "best_f1": None})
        time.sleep(args.sleep_between)

    # Save results
    with open(OUT / "experiment_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "seed", "arm", "model", "best_f1"])
        writer.writeheader()
        writer.writerows(results)

    print("Orchestration complete. Results saved to", OUT / "experiment_results.csv")

if __name__ == "__main__":
    main()
