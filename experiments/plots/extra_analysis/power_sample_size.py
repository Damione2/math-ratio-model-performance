#!/usr/bin/env python3
"""
Compute power and recommended seeds per arm for the symbol manipulation experiment.

This script:
- Loads seed-level best_f1 from experiments/ablation_summary.csv
- Computes observed within-run SD of best_f1 and the observed slope effect (if --effect-estimate auto)
- Runs a simple power calculation for a two-sample t-test (Control vs Treatment)
- Optionally runs a small simulation for mixed-model power (if requested)

Outputs:
- experiments/plots/extra_analysis/power_results.json
- prints recommended seeds per arm for 80% and 90% power
"""
import argparse
import json
import numpy as np
import pandas as pd
from math import ceil
from scipy import stats

def two_sample_n_per_group(delta, sd, alpha=0.05, power=0.8):
    # Cohen d = delta / sd
    d = abs(delta) / sd
    # Use approximation for two-sample t (equal n)
    # n = 2 * ( (z_{1-alpha/2} + z_{power})^2 ) / d^2
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_power = stats.norm.ppf(power)
    n = 2 * ((z_alpha + z_power)**2) / (d**2)
    return ceil(n)

def simulate_mixed_power(delta, sd, n_per_arm, n_sims=1000, alpha=0.05):
    # Simple simulation: generate best_f1 per run ~ N(mu, sd^2)
    # Control mean = 0, Treatment mean = -delta
    # Test with two-sample t as proxy for mixed model power
    rng = np.random.default_rng(123)
    successes = 0
    for _ in range(n_sims):
        control = rng.normal(0.0, sd, size=n_per_arm)
        treat = rng.normal(-delta, sd, size=n_per_arm)
        t, p = stats.ttest_ind(control, treat, equal_var=False)
        if p < alpha:
            successes += 1
    return successes / n_sims

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="experiments/ablation_summary.csv")
    p.add_argument("--effect-estimate", default="auto",
                   help="Use 'auto' to estimate delta from data (regression slope * mean math_pct range) or provide numeric delta (absolute F1 change).")
    args = p.parse_args()

    df = pd.read_csv(args.summary)
    # Use best_f1 across seeds to estimate sd
    best_f1 = df["best_f1"].dropna().values
    sd = float(np.std(best_f1, ddof=1))
    mean_math_pct = float((df["math_ratio"].dropna() * 100.0).mean())
    # If auto, estimate delta as observed mixedlm slope * 10% change (practical)
    if args.effect_estimate == "auto":
        # Use published mixedlm slope if present in project files; fallback to -0.0025 per 1%
        try:
            import json, pathlib
            mfile = pathlib.Path("experiments/plots/mixedlm_result.json")
            if mfile.exists():
                m = json.loads(mfile.read_text())
                slope = m.get("fixed_effect_math_pct_coef", -0.0025)
            else:
                slope = -0.0025
        except Exception:
            slope = -0.0025
        # delta = slope * 10 (effect for 10 percentage points)
        delta = abs(slope) * 10.0
    else:
        delta = float(args.effect_estimate)

    results = {}
    results["observed_sd_best_f1"] = sd
    results["assumed_effect_delta"] = delta
    # compute n for 80% and 90% power
    results["n_80"] = two_sample_n_per_group(delta, sd, alpha=0.05, power=0.8)
    results["n_90"] = two_sample_n_per_group(delta, sd, alpha=0.05, power=0.9)
    # quick simulation for recommended n_80
    sim_power = simulate_mixed_power(delta, sd, results["n_80"], n_sims=1000)
    results["simulated_power_at_n_80"] = sim_power

    out = "experiments/plots/extra_analysis/power_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Wrote power results to", out)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
