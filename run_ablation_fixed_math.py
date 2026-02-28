# run_ablation_fixed_math.py
import subprocess, sys, os, time

python = sys.executable

def run(cmd):
    print("\n>>>", cmd)
    return subprocess.call(cmd, shell=True)

def run_fixed_math(experiment_name, keep_ratio):
    print(f"\n=== Running {experiment_name} with fixed math ratio {keep_ratio} ===")

    # STEP 1 — generate WITHOUT synthetic math, WITHOUT adversarial math, WITHOUT balancing
    # We do this by:
    #  - disabling synthetic math
    #  - disabling adversarial math
    #  - disabling balancing by passing a fake flag the pipeline ignores
    #  - reducing base math only
    cmd1 = (
        f'{python} -m pipeline.guardian_pipeline_master '
        f'--step 1 '
        f'--count 9000 '
        f'--math-synthetic-pairs 0 '
        f'--no-auto-generate-adv '
        f'--reduce-math '
        f'--math-keep-ratio {keep_ratio} '
        f'--experiment {experiment_name}'
    )
    if run(cmd1) != 0:
        print("Step 1 failed")
        return

    # STEP 2
    if run(f'{python} -m pipeline.guardian_pipeline_master --step 2') != 0:
        print("Step 2 failed")
        return

    # STEP 3
    if run(f'{python} -m pipeline.guardian_pipeline_master --step 3') != 0:
        print("Step 3 failed")
        return

    # STEP 4 — training
    cmd4 = (
        f'{python} -m pipeline.guardian_pipeline_master '
        f'--step 4 '
        f'--no-resume '
        f'--batch-size 512 '
        f'--experiment {experiment_name}'
    )
    run(cmd4)


if __name__ == "__main__":
    # Define your ablation ratios
    ratios = {
        "math_0": 0.00,
        "math_10": 0.10,
        "math_25": 0.25,
        "math_50": 0.50,
        "math_100": 1.00
    }

    for name, ratio in ratios.items():
        run_fixed_math(name, ratio)
