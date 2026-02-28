#!/usr/bin/env python3
"""
merge_and_run.py - Helper for orchestrate_experiment_pipeline

Merges a matched subset CSV/JSON into ARTIFACTS_DIR/02_train.pkl
then runs pipeline/guardian_pipeline_master.py --step 4 with full flag forwarding
(including --batch-size, --num-workers, --seed, --llm, --output-dir, etc.).
"""

import argparse
import sys
from pathlib import Path
import subprocess
import os

# Make sure we can import from the pipeline module
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.guardian_pipeline_master import convert_subset_to_train_pickle


def main():
    parser = argparse.ArgumentParser(description="Merge subset and run Guardian training step")
    parser.add_argument("--matched-csv", required=True, help="Path to the matched examples CSV/JSON")
    parser.add_argument("--artifacts-dir", required=True, help="Path to ARTIFACTS_DIR")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--run-id", required=True, help="Unique run identifier")
    parser.add_argument("--llm", default="Qwen2.5-Math-1.5B", help="LLM name for logging")
    parser.add_argument("--step", type=int, default=4, help="Pipeline step (default 4)")
    parser.add_argument("--pipeline", default="pipeline/guardian_pipeline_master.py", help="Path to master pipeline script")
    parser.add_argument("--key", default="label", help="Key used for matching samples")

    # === NEW: Training flags forwarded to guardian_pipeline_master.py ===
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional run output directory (overrides runs/run_<run_id>)")

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting merge_and_run for run_id={args.run_id}")
    print(f"   Matched CSV : {args.matched_csv}")
    print(f"   Batch size  : {args.batch_size}")
    print(f"   Num workers : {args.num_workers}")
    print(f"   Seed        : {args.seed}")
    print(f"   LLM         : {args.llm}")

    # Step 1: Convert/merge the subset into 02_train.pkl (reuses the function from master)
    try:
        train_pkl_path = convert_subset_to_train_pickle(
            subset_path=Path(args.matched_csv),
            artifacts_dir=artifacts_dir,
            key=args.key
        )
        print(f"✅ Successfully created/updated: {train_pkl_path}")
    except Exception as e:
        print(f"❌ Failed to convert subset: {e}")
        sys.exit(1)

    # Step 2: Run the main pipeline (step 4) with all flags forwarded
    pipeline_script = Path(args.pipeline).resolve()

    cmd = [
        sys.executable,
        str(pipeline_script),
        "--step", str(args.step),
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--no-resume",                 # fresh training for each experiment run
        "--experiment", f"symbol_run_{args.run_id}",
        "--llm", args.llm,
    ]

    # Output directory override so logs/metrics go into runs/run_XXXX/
    if args.output_dir:
        run_output_dir = Path(args.output_dir)
    else:
        run_output_dir = Path("runs") / f"run_{args.run_id}"
    run_output_dir.mkdir(parents=True, exist_ok=True)
    cmd.extend(["--output-dir", str(run_output_dir)])

    print(f"🔄 Running pipeline: {' '.join(cmd)}")

    # Capture output to files (so orchestrate can read them)
    stdout_path = run_output_dir / "stdout.txt"
    stderr_path = run_output_dir / "stderr.txt"

    try:
        with open(stdout_path, "w", encoding="utf-8") as outf, \
             open(stderr_path, "w", encoding="utf-8") as errf:

            # ensure downstream processes can see RUN_ID and optional OUTPUT_DIR
            env = os.environ.copy()
            env["RUN_ID"] = args.run_id
            env["OUTPUT_DIR"] = str(run_output_dir)

            result = subprocess.run(
                cmd,
                stdout=outf,
                stderr=errf,
                text=True,
                check=False,
                env=env
            )

        print(f"Pipeline finished with return code: {result.returncode}")
        print(f"   Logs saved → {run_output_dir}")

        if result.returncode != 0:
            print("⚠️  Pipeline had errors. Check stderr.txt above.")
            sys.exit(result.returncode)

    except Exception as e:
        print(f"❌ Failed to execute pipeline: {e}")
        sys.exit(1)

    print(f"✅ Run {args.run_id} completed successfully!")


if __name__ == "__main__":
    main()
