#!/usr/bin/env python3
"""
analyze_experiment_results.py

Single script to perform steps 1-6 of post-experiment analysis:
  1) Collect per-run best_f1 values from runs/run_<id> folders
  2) Merge with seed_assignments.csv to label Control vs Treatment
  3) Compute descriptive stats and 95% CIs
  4) Compute Cohen's d (effect size)
  5) Bootstrap 95% CI for difference and permutation p-value
  6) Welch's t-test
  + Produce plots, robustness checks, simple error analysis sampling,
    optional holdout evaluation (best-effort), and reproducibility artifacts.

Usage:
  python experiments/plots/extra_analysis/analyze_experiment_results.py \
    --runs-dir runs --assignments experiments/plots/extra_analysis/seed_assignments.csv

Dependencies:
  - Python 3.8+
  - numpy, pandas, scipy, matplotlib, seaborn
If missing, install with:
  pip install numpy pandas scipy matplotlib seaborn

Outputs:
  - analysis/report_<timestamp>/ : CSV summary, plots, bootstrap arrays, README, excerpts
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try imports and provide helpful message if missing
try:
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Missing Python packages. Please install required packages:")
    print("  pip install numpy pandas scipy matplotlib seaborn")
    raise

# -------------------------
# Utilities
# -------------------------
BEST_F1_PATTERNS = [
    re.compile(r"BEST_F1\s+([0-9]*\.?[0-9]+)"),
    re.compile(r"Best F1[:\s]+([0-9]*\.?[0-9]+)"),
    re.compile(r"best_f1[:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE),
]

def extract_best_f1_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    for pat in BEST_F1_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    # fallback: look for a line like "BEST F1: 0.8123"
    m = re.search(r"BEST.*F1.*?([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def safe_read_text(path: Path, tail_lines: int = 2000) -> str:
    try:
        # read entire file but guard memory by reading tail if large
        size = path.stat().st_size
        if size < 2_000_000:  # <2MB
            return path.read_text(encoding="utf-8", errors="replace")
        # else read tail
        with path.open("rb") as f:
            # read last ~200KB
            f.seek(max(0, size - 200_000))
            data = f.read().decode("utf-8", errors="replace")
            return data
    except Exception:
        return ""

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -------------------------
# Core collection logic
# -------------------------
def collect_runs(runs_dir: Path) -> Dict[str, Dict]:
    """
    Scan runs_dir for run_<id> folders and extract best_f1 and metadata.
    Returns dict keyed by run_id with fields:
      - run_id, path, best_f1 (float or None), stdout, stderr, metrics_json (dict or None)
    """
    runs = {}
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    for run_folder in sorted(runs_dir.glob("run_*")):
        run_id = run_folder.name.replace("run_", "")
        info = {"run_id": run_id, "path": run_folder, "best_f1": None,
                "metrics": None, "stdout": None, "stderr": None}
        # metrics.json
        metrics_path = run_folder / "metrics.json"
        if metrics_path.exists():
            try:
                m = json.loads(metrics_path.read_text(encoding="utf-8", errors="replace"))
                info["metrics"] = m
                # common key names
                for k in ("best_f1", "best-f1", "bestF1", "BEST_F1"):
                    if k in m:
                        try:
                            info["best_f1"] = float(m[k])
                            break
                        except Exception:
                            pass
            except Exception:
                info["metrics"] = None

        # stdout/stderr
        stdout_path = run_folder / "stdout.txt"
        stderr_path = run_folder / "stderr.txt"
        if stdout_path.exists():
            s = safe_read_text(stdout_path)
            info["stdout"] = s
            if info["best_f1"] is None:
                val = extract_best_f1_from_text(s)
                if val is not None:
                    info["best_f1"] = val
        if stderr_path.exists():
            s = safe_read_text(stderr_path)
            info["stderr"] = s
            if info["best_f1"] is None:
                val = extract_best_f1_from_text(s)
                if val is not None:
                    info["best_f1"] = val

        # fallback: try any .log or training_summary
        if info["best_f1"] is None:
            for alt in run_folder.glob("*.log"):
                txt = safe_read_text(alt)
                val = extract_best_f1_from_text(txt)
                if val is not None:
                    info["best_f1"] = val
                    break

        runs[run_id] = info
    return runs

def load_assignments(assignments_csv: Path) -> pd.DataFrame:
    if not assignments_csv.exists():
        raise FileNotFoundError(f"Assignments CSV not found: {assignments_csv}")
    df = pd.read_csv(assignments_csv, dtype=str)
    # normalize columns
    expected = {"run_id", "seed", "arm", "subset", "outdir"}
    # try to infer if run_id has run_ prefix or not
    if "run_id" not in df.columns and "run" in df.columns:
        df = df.rename(columns={"run": "run_id"})
    # ensure run_id has no 'run_' prefix
    df["run_id"] = df["run_id"].astype(str).apply(lambda x: x.replace("run_", ""))
    return df

# -------------------------
# Statistics & tests
# -------------------------
def describe_group(values: List[float]) -> Dict:
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    median = float(np.median(arr))
    se = std / math.sqrt(n) if n > 1 else 0.0
    # t-based 95% CI
    if n > 1:
        t = stats.t.ppf(0.975, df=n-1)
        ci = (mean - t * se, mean + t * se)
    else:
        ci = (mean, mean)
    return {"n": n, "mean": mean, "std": std, "median": median, "se": se, "95ci": ci}

def cohens_d(x: List[float], y: List[float]) -> Optional[float]:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return None
    sx2 = np.var(x, ddof=1)
    sy2 = np.var(y, ddof=1)
    sp = math.sqrt(((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2))
    if sp == 0:
        return None
    return (np.mean(x) - np.mean(y)) / sp

def bootstrap_diff_means(x: List[float], y: List[float], n_boot: int = 10000, seed: Optional[int] = None) -> Tuple[float, Tuple[float, float], np.ndarray]:
    rng = np.random.RandomState(seed)
    diffs = np.empty(n_boot, dtype=float)
    x = np.array(x)
    y = np.array(y)
    nx, ny = len(x), len(y)
    for i in range(n_boot):
        sx = rng.choice(x, size=nx, replace=True)
        sy = rng.choice(y, size=ny, replace=True)
        diffs[i] = sx.mean() - sy.mean()
    mean_diff = float(diffs.mean())
    ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
    return mean_diff, ci, diffs

def permutation_test_mean_diff(x: List[float], y: List[float], n_perm: int = 10000, seed: Optional[int] = None) -> float:
    rng = np.random.RandomState(seed)
    x = np.array(x)
    y = np.array(y)
    obs = float(np.mean(x) - np.mean(y))
    pooled = np.concatenate([x, y])
    n = len(x)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        px = pooled[:n]
        py = pooled[n:]
        if abs(px.mean() - py.mean()) >= abs(obs):
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return pval

# -------------------------
# Plots
# -------------------------
def plot_means_ci(df_summary: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(6, 5))
    sns.barplot(x="arm", y="mean", data=df_summary, yerr=df_summary["se"] * stats.t.ppf(0.975, df=df_summary["n"] - 1))
    plt.ylabel("Mean best_f1")
    plt.title("Mean best_f1 by arm (95% CI)")
    plt.tight_layout()
    p = outdir / "means_ci.png"
    plt.savefig(p, dpi=200)
    plt.close()

def plot_violin(values_df: pd.DataFrame, outdir: Path):
    plt.figure(figsize=(8, 5))
    sns.violinplot(x="arm", y="best_f1", data=values_df, inner="box", cut=0)
    sns.swarmplot(x="arm", y="best_f1", data=values_df, color="k", alpha=0.6, size=3)
    plt.ylabel("best_f1")
    plt.title("Distribution of best_f1 by arm")
    plt.tight_layout()
    p = outdir / "violin_best_f1.png"
    plt.savefig(p, dpi=200)
    plt.close()

def plot_bootstrap_hist(diffs: np.ndarray, outdir: Path):
    plt.figure(figsize=(7, 4))
    sns.histplot(diffs, bins=60, kde=True)
    plt.axvline(np.mean(diffs), color="red", linestyle="--", label=f"mean={np.mean(diffs):.4f}")
    plt.title("Bootstrap distribution of mean differences (Treatment - Control)")
    plt.xlabel("Mean difference")
    plt.legend()
    plt.tight_layout()
    p = outdir / "bootstrap_diff_hist.png"
    plt.savefig(p, dpi=200)
    plt.close()

# -------------------------
# Error analysis sampling
# -------------------------
def sample_error_excerpts(runs_info: Dict[str, Dict], assignments_df: pd.DataFrame, outdir: Path, per_arm: int = 10):
    """
    For up to per_arm runs per arm, extract tail excerpts of stdout/stderr and save to files.
    Also attempt to find predictions.json or outputs/ files.
    """
    ensure_dir(outdir)
    merged = []
    # join runs_info with assignments
    for _, row in assignments_df.iterrows():
        rid = str(row["run_id"]).replace("run_", "")
        arm = row.get("arm", "Unknown")
        if rid in runs_info:
            info = runs_info[rid]
            merged.append({"run_id": rid, "arm": arm, "path": info["path"], "stdout": info.get("stdout"), "stderr": info.get("stderr")})
    df = pd.DataFrame(merged)
    results = []
    for arm in df["arm"].unique():
        sub = df[df["arm"] == arm]
        # sample up to per_arm
        sample = sub.sample(n=min(per_arm, len(sub)), random_state=42) if len(sub) > 0 else sub
        for _, r in sample.iterrows():
            rid = r["run_id"]
            run_dir = Path(r["path"])
            excerpt = {}
            # stdout excerpt
            if r["stdout"]:
                excerpt_text = "\n".join(r["stdout"].splitlines()[-200:])
            else:
                p = run_dir / "stdout.txt"
                excerpt_text = safe_read_text(p)[-2000:] if p.exists() else ""
            excerpt["stdout_excerpt"] = excerpt_text
            # stderr excerpt
            if r["stderr"]:
                excerpt_text = "\n".join(r["stderr"].splitlines()[-200:])
            else:
                p = run_dir / "stderr.txt"
                excerpt_text = safe_read_text(p)[-2000:] if p.exists() else ""
            excerpt["stderr_excerpt"] = excerpt_text
            # try predictions
            preds = None
            for cand in ["predictions.json", "outputs.json", "preds.json"]:
                p = run_dir / cand
                if p.exists():
                    try:
                        preds = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                        break
                    except Exception:
                        preds = None
            excerpt["predictions"] = preds
            # write excerpt file
            fname = outdir / f"{arm}_run_{rid}_excerpt.txt"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(f"Run: {rid}\nArm: {arm}\nPath: {run_dir}\n\n--- STDOUT (tail) ---\n")
                f.write(excerpt["stdout_excerpt"] or "<no stdout>\n")
                f.write("\n\n--- STDERR (tail) ---\n")
                f.write(excerpt["stderr_excerpt"] or "<no stderr>\n")
                if preds:
                    f.write("\n\n--- PREDICTIONS (first 3) ---\n")
                    try:
                        f.write(json.dumps(preds if isinstance(preds, list) else list(preds.items())[:3], indent=2))
                    except Exception:
                        f.write(str(preds)[:2000])
            results.append({"run_id": rid, "arm": arm, "excerpt_file": str(fname)})
    # save CSV index
    out_csv = outdir / "error_excerpts_index.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    return out_csv

# -------------------------
# Holdout evaluation (best-effort)
# -------------------------
def holdout_evaluate(holdout_path: Optional[Path], artifacts_dir: Optional[Path], outdir: Path):
    """
    Best-effort: if holdout provided and artifacts_dir contains model and features,
    attempt to run a lightweight evaluation. This function is intentionally conservative:
    it will not attempt to load heavy models. Instead it checks for val_features.bin or
    metrics and reports if evaluation is possible. If not possible, it writes a short note.
    """
    note = []
    if holdout_path is None:
        note.append("No holdout provided; skipping holdout evaluation.")
        (outdir / "holdout_note.txt").write_text("\n".join(note))
        return
    note.append(f"Holdout provided: {holdout_path}")
    if artifacts_dir is None:
        note.append("No artifacts_dir provided; cannot load model or features.")
        (outdir / "holdout_note.txt").write_text("\n".join(note))
        return
    # check for features or model
    features = artifacts_dir / "val_features.bin"
    model = artifacts_dir / "guardian_spider_native.pth"
    if features.exists() and model.exists():
        note.append("Found val_features.bin and guardian_spider_native.pth. You can run full evaluation using pipeline step 7/8.")
    else:
        note.append("No suitable features/model found for automated holdout evaluation. Skipping.")
    (outdir / "holdout_note.txt").write_text("\n".join(note))

# -------------------------
# Main orchestration
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results (Control vs Treatment)")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run_<id> folders")
    parser.add_argument("--assignments", type=str, default="experiments/plots/extra_analysis/seed_assignments.csv", help="CSV with run assignments")
    parser.add_argument("--outdir", type=str, default="analysis", help="Base output folder for analysis")
    parser.add_argument("--n-boot", type=int, default=10000, help="Bootstrap resamples")
    parser.add_argument("--n-perm", type=int, default=10000, help="Permutation resamples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--holdout", type=str, default=None, help="Optional holdout CSV/JSON for evaluation")
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Optional ARTIFACTS_DIR for holdout/model evaluation")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    runs_dir = Path(args.runs_dir)
    assignments_csv = Path(args.assignments)
    base_out = Path(args.outdir)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_dir = base_out / f"report_{timestamp}"
    ensure_dir(report_dir)

    print(f"Collecting runs from: {runs_dir}")
    runs_info = collect_runs(runs_dir)
    print(f"Found {len(runs_info)} runs.")

    print(f"Loading assignments from: {assignments_csv}")
    assignments_df = load_assignments(assignments_csv)

    # Merge runs_info with assignments
    rows = []
    for _, row in assignments_df.iterrows():
        run_id = str(row["run_id"]).replace("run_", "")
        arm = row.get("arm", "Unknown")
        seed = row.get("seed", "")
        subset = row.get("subset", "")
        outdir = row.get("outdir", "")
        info = runs_info.get(run_id, None)
        best_f1 = None
        metrics = None
        if info:
            best_f1 = info.get("best_f1")
            metrics = info.get("metrics")
        rows.append({"run_id": run_id, "arm": arm, "seed": seed, "subset": subset, "outdir": outdir, "best_f1": best_f1, "metrics": metrics})
    df = pd.DataFrame(rows)

    # Some runs may exist in runs_dir but not in assignments; include them as unknown
    extra_runs = [rid for rid in runs_info.keys() if rid not in set(df["run_id"].astype(str))]
    for rid in extra_runs:
        info = runs_info[rid]
        df = pd.concat([df, pd.DataFrame([{"run_id": rid, "arm": "Unknown", "seed": "", "subset": "", "outdir": str(info["path"]), "best_f1": info.get("best_f1"), "metrics": info.get("metrics")}])], ignore_index=True)

    # Save merged table
    merged_csv = report_dir / "merged_runs.csv"
    df.to_csv(merged_csv, index=False)
    print(f"Merged runs saved to: {merged_csv}")

    # Filter to runs with best_f1
    df_valid = df[df["best_f1"].notnull()].copy()
    df_valid["best_f1"] = df_valid["best_f1"].astype(float)
    print(f"Usable runs with best_f1: {len(df_valid)} / {len(df)}")

    # Group by arm
    arms = df_valid["arm"].unique().tolist()
    groups = {arm: df_valid[df_valid["arm"] == arm]["best_f1"].tolist() for arm in arms}

    # Descriptive stats
    summary_rows = []
    for arm, vals in groups.items():
        desc = describe_group(vals)
        summary_rows.append({"arm": arm, **desc})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "summary_stats.csv", index=False)
    print("Summary stats:")
    print(summary_df.to_string(index=False))

    # If exactly two arms (Control and Treatment), compute comparative stats
    if len(groups) >= 2:
        # Try to pick Control and Treatment names
        # Prefer exact names "Control" and "Treatment"
        if "Control" in groups and "Treatment" in groups:
            control_vals = groups["Control"]
            treat_vals = groups["Treatment"]
            control_name = "Control"
            treat_name = "Treatment"
        else:
            # pick first two arms
            arm_names = list(groups.keys())
            control_name, treat_name = arm_names[0], arm_names[1]
            control_vals = groups[control_name]
            treat_vals = groups[treat_name]

        print(f"\nComparing arms: {treat_name} (treatment) vs {control_name} (control)")
        # Welch's t-test
        try:
            tstat, pval_t = stats.ttest_ind(treat_vals, control_vals, equal_var=False, nan_policy="omit")
        except Exception:
            tstat, pval_t = float("nan"), float("nan")
        # Cohen's d
        d = cohens_d(treat_vals, control_vals)
        # Bootstrap
        mean_diff, boot_ci, boot_diffs = bootstrap_diff_means(treat_vals, control_vals, n_boot=args.n_boot, seed=args.seed)
        # Permutation
        perm_p = permutation_test_mean_diff(treat_vals, control_vals, n_perm=args.n_perm, seed=args.seed)
        # Save bootstrap array
        np.save(report_dir / "bootstrap_diffs.npy", boot_diffs)
        # Save comparative summary
        comp = {
            "treatment": treat_name,
            "control": control_name,
            "n_treatment": len(treat_vals),
            "n_control": len(control_vals),
            "mean_treatment": float(np.mean(treat_vals)) if len(treat_vals) else None,
            "mean_control": float(np.mean(control_vals)) if len(control_vals) else None,
            "mean_diff": mean_diff,
            "bootstrap_95ci_low": boot_ci[0],
            "bootstrap_95ci_high": boot_ci[1],
            "cohens_d": d,
            "welch_t_pvalue": pval_t,
            "permutation_pvalue": perm_p
        }
        comp_df = pd.DataFrame([comp])
        comp_df.to_csv(report_dir / "comparison_summary.csv", index=False)
        print("\nComparison summary:")
        print(comp_df.to_string(index=False))

        # Plots
        try:
            plot_means_ci(summary_df, report_dir)
            plot_violin(df_valid[["arm", "best_f1"]].rename(columns={"best_f1": "best_f1"}), report_dir)
            plot_bootstrap_hist(boot_diffs, report_dir)
            print(f"Plots saved to {report_dir}")
        except Exception as e:
            print("Plotting failed:", e)

        # Robustness: sensitivity by leave-one-out (recompute mean diff removing each run)
        sens = []
        all_t = np.array(treat_vals)
        all_c = np.array(control_vals)
        for i in range(len(all_t)):
            t_sub = np.delete(all_t, i)
            if len(t_sub) == 0:
                continue
            diff = t_sub.mean() - all_c.mean()
            sens.append(diff)
        for j in range(len(all_c)):
            c_sub = np.delete(all_c, j)
            if len(c_sub) == 0:
                continue
            diff = all_t.mean() - c_sub.mean()
            sens.append(diff)
        sens = np.array(sens) if sens else np.array([])
        if sens.size > 0:
            sens_low, sens_high = np.percentile(sens, [2.5, 97.5])
            with open(report_dir / "robustness_sensitivity.txt", "w", encoding="utf-8") as f:
                f.write(f"Sensitivity (leave-one-out) 95% range for mean diff: [{sens_low:.6f}, {sens_high:.6f}]\n")
            print("Robustness sensitivity saved.")
    else:
        print("Not enough arms with data to perform comparative tests (need at least 2).")

    # Error analysis sampling
    excerpts_dir = report_dir / "excerpts"
    ensure_dir(excerpts_dir)
    try:
        idx_csv = sample_error_excerpts(runs_info, assignments_df, excerpts_dir, per_arm=10)
        print(f"Error excerpts index saved to: {idx_csv}")
    except Exception as e:
        print("Error sampling excerpts failed:", e)

    # Holdout evaluation (best-effort)
    holdout_path = Path(args.holdout) if args.holdout else None
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else None
    try:
        holdout_evaluate(holdout_path, artifacts_dir, report_dir)
    except Exception as e:
        print("Holdout evaluation step failed:", e)

    # Reproducibility README
    readme = report_dir / "README.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write("Analysis report\n")
        f.write("================\n")
        f.write(f"Generated: {datetime.utcnow().isoformat()} UTC\n\n")
        f.write("Files in this report:\n")
        for p in sorted(report_dir.rglob("*")):
            if p.is_file():
                f.write(f" - {p.relative_to(report_dir)}\n")
        f.write("\nReproducibility\n")
        f.write("----------------\n")
        f.write("To reproduce this analysis run:\n\n")
        f.write("python experiments/plots/extra_analysis/analyze_experiment_results.py \\\n")
        f.write(f"  --runs-dir {runs_dir} \\\n")
        f.write(f"  --assignments {assignments_csv} \\\n")
        f.write(f"  --outdir {base_out} \\\n")
        f.write(f"  --n-boot {args.n_boot} --n-perm {args.n_perm} --seed {args.seed}\n\n")
        f.write("Notes:\n - This script extracts best_f1 from metrics.json or stdout/stderr.\n - If some runs are missing best_f1 they are excluded from statistical tests.\n - For holdout evaluation, provide --holdout and --artifacts-dir if available.\n")
    print(f"Report written to: {report_dir}")

if __name__ == "__main__":
    main()
