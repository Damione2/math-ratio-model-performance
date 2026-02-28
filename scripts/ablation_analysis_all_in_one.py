#!/usr/bin/env python3
# scripts/ablation_analysis_all_in_one.py
"""
Purpose
-------
Comprehensive ablation analysis pipeline (single file). This script:
  - Loads seed-level summary from experiments/ablation_summary.csv (explicit path)
  - Computes aggregated group means by (model, math_ratio)
  - Runs aggregated OLS regression (best_f1_mean ~ math_pct) on groups with min_seeds
  - Runs seed-level MixedLM (best_f1 ~ math_pct with random intercepts for model)
  - Computes bootstrap 95% CI for aggregated slope
  - Computes Cook's distance and flags high-influence aggregated groups
  - Performs sensitivity checks:
      * Exclude the three flagged high-influence aggregated groups (exact math_ratio values)
      * Leave-one-out aggregated sensitivity (drop each aggregated group in turn)
  - Extra analyses (added):
      * Weighted Least Squares (WLS) aggregated regression (weights = n_seeds)
      * MixedLM with interaction (math_pct * C(model)) to test slope heterogeneity by model
      * Slopes comparison table and figure (base OLS, sensitivity OLS, WLS, MixedLM base, MixedLM sensitivity)
  - Writes CSV/JSON/PNG/TXT/MD outputs to experiments/plots/ and subfolders
  - All file/folder names and locations are explicit in the script
Usage (from project root, venv active):
  python scripts/ablation_analysis_all_in_one.py --summary experiments/ablation_summary.csv --out experiments/plots --min-seeds 2
Notes
  - Requires: pandas, numpy, matplotlib, seaborn, scipy, statsmodels
  - The script will create additional outputs under:
      experiments/plots/
      experiments/plots/sensitivity_excluding_high_influence/
      experiments/plots/leave_one_out/
      experiments/plots/extra_analysis/
"""

import argparse
from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.outliers_influence import OLSInfluence

sns.set(style="whitegrid")

# ---------- Argument parsing ----------
parser = argparse.ArgumentParser(description="Aggregate ablation analyses (regression, mixed model, bootstrap, influence, sensitivity, extra analyses).")
parser.add_argument("--summary", default="experiments/ablation_summary.csv", help="Path to seed-level summary CSV (explicit).")
parser.add_argument("--out", default="experiments/plots", help="Output directory for plots and reports (explicit).")
parser.add_argument("--min-seeds", type=int, default=2, help="Minimum seeds per (model,math_ratio) to include in aggregated regression.")
parser.add_argument("--bootstrap-iters", type=int, default=2000, help="Bootstrap iterations for slope CI.")
args = parser.parse_args()

SUMMARY_CSV = Path(args.summary).resolve()
OUT_DIR = Path(args.out).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRA_DIR = OUT_DIR / "extra_analysis"
EXTRA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Helper functions ----------
def parse_ratio(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str) and x.strip().endswith("%"):
        try:
            return float(x.strip().rstrip("%")) / 100.0
        except:
            return float("nan")
    try:
        return float(x)
    except:
        return float("nan")

def bootstrap_slope(data_df, iters=2000, seed=42):
    rng = np.random.default_rng(seed)
    slopes = []
    n = len(data_df)
    Xcol = "math_pct"
    Ycol = "best_f1_mean"
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        sample = data_df.iloc[idx]
        Xs = sm.add_constant(sample[Xcol].values)
        ys = sample[Ycol].values
        try:
            m = sm.OLS(ys, Xs).fit()
            slopes.append(m.params[1])
        except:
            slopes.append(np.nan)
    slopes = np.array(slopes)
    slopes = slopes[~np.isnan(slopes)]
    lower = float(np.percentile(slopes, 2.5)) if len(slopes) else None
    upper = float(np.percentile(slopes, 97.5)) if len(slopes) else None
    median = float(np.median(slopes)) if len(slopes) else None
    return {"median": median, "ci_lower": lower, "ci_upper": upper, "n_boot": int(len(slopes))}

def safe_write_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def safe_plot_save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ---------- Core analysis function ----------
def run_full_analysis(summary_csv: Path, out_dir: Path, min_seeds: int = 2, bootstrap_iters: int = 2000, report_prefix: str = ""):
    """
    Run the full analysis pipeline on a given seed-level summary CSV and write outputs to out_dir.
    Returns a dict with key results and DataFrames for further programmatic checks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    df["math_ratio"] = df["math_ratio"].apply(parse_ratio)
    df = df.dropna(subset=["math_ratio", "best_f1"])
    df["math_pct"] = df["math_ratio"] * 100.0

    # Aggregation
    group_cols = ["model", "math_ratio"]
    grouped = df.groupby(group_cols)
    agg = grouped["best_f1"].agg(["mean", "std", "count"]).reset_index().rename(columns={"mean":"best_f1_mean","std":"best_f1_std","count":"n_seeds"})
    agg["math_pct"] = agg["math_ratio"] * 100.0
    agg_csv = out_dir / f"{report_prefix}aggregated_by_model_ratio.csv"
    agg.to_csv(agg_csv, index=False, encoding="utf-8")

    # Filter groups with at least min_seeds
    agg_filtered = agg[agg["n_seeds"] >= min_seeds].copy()
    if agg_filtered.empty:
        raise ValueError(f"No aggregated groups with n_seeds >= {min_seeds} in {summary_csv}")

    # Aggregated OLS regression
    X = sm.add_constant(agg_filtered["math_pct"].values)
    y = agg_filtered["best_f1_mean"].values
    ols_model = sm.OLS(y, X).fit()
    regression_result = {
        "slope": float(ols_model.params[1]),
        "intercept": float(ols_model.params[0]),
        "r_squared": float(ols_model.rsquared),
        "p_value": float(ols_model.pvalues[1]),
        "stderr": float(ols_model.bse[1]),
        "n": int(len(agg_filtered))
    }
    safe_write_json(out_dir / f"{report_prefix}regression_bestf1_vs_mathratio.json", regression_result)

    # Plot aggregated regression
    fig = plt.figure(figsize=(9,6))
    sns.scatterplot(data=agg_filtered, x="math_pct", y="best_f1_mean", size="n_seeds", legend=False)
    x_vals = np.linspace(agg_filtered["math_pct"].min(), agg_filtered["math_pct"].max(), 200)
    y_vals = regression_result["intercept"] + regression_result["slope"] * x_vals
    plt.plot(x_vals, y_vals, color="red", label=f"OLS fit (slope={regression_result['slope']:.6f})")
    plt.xlabel("Observed math ratio (%)")
    plt.ylabel("Mean Best Val F1 (per group)")
    plt.title("Aggregated regression: mean Best F1 vs Observed math ratio")
    plt.legend()
    safe_plot_save(fig, out_dir / f"{report_prefix}agg_regression_plot.png")

    # Cook's distance
    influence = OLSInfluence(ols_model)
    cooks = influence.cooks_distance[0]
    agg_filtered = agg_filtered.reset_index(drop=True)
    agg_filtered["cooks_d"] = cooks
    cooks_csv = out_dir / f"{report_prefix}cooks_distance.csv"
    agg_filtered[["model","math_ratio","math_pct","best_f1_mean","n_seeds","cooks_d"]].to_csv(cooks_csv, index=False, encoding="utf-8")
    threshold = 4.0 / len(agg_filtered)
    high_influence = agg_filtered[agg_filtered["cooks_d"] > threshold].copy()

    # Bootstrap slope CI
    bootstrap_res = bootstrap_slope(agg_filtered, iters=bootstrap_iters)
    safe_write_json(out_dir / f"{report_prefix}bootstrap_slope_ci.json", bootstrap_res)

    # MixedLM (seed-level)
    df_mixed = df.dropna(subset=["math_pct","best_f1","model"]).copy()
    df_mixed["model"] = df_mixed["model"].astype(str)
    md = smf.mixedlm("best_f1 ~ math_pct", df_mixed, groups=df_mixed["model"])
    try:
        mdf = md.fit(reml=False, method="lbfgs")
        mixed_summary = mdf.summary().as_text()
        with open(out_dir / f"{report_prefix}mixedlm_summary.txt", "w", encoding="utf-8") as f:
            f.write(mixed_summary)
        mixed_result = {
            "fixed_effect_math_pct_coef": float(mdf.params.get("math_pct", np.nan)),
            "fixed_effect_math_pct_se": float(mdf.bse.get("math_pct", np.nan)),
            "pvalue": float(mdf.pvalues.get("math_pct", np.nan)) if hasattr(mdf, "pvalues") else None,
            "converged": bool(getattr(mdf, "mle_retvals", {}).get("converged", True))
        }
    except Exception as e:
        mixed_summary = f"MixedLM failed: {e}\n\n{traceback.format_exc()}"
        with open(out_dir / f"{report_prefix}mixedlm_summary.txt", "w", encoding="utf-8") as f:
            f.write(mixed_summary)
        mixed_result = {"error": str(e)}

    safe_write_json(out_dir / f"{report_prefix}mixedlm_result.json", mixed_result)

    # Residual plot for mixed model if available
    try:
        if "mdf" in locals() and hasattr(mdf, "fittedvalues"):
            resid = mdf.resid
            fitted = mdf.fittedvalues
            fig = plt.figure(figsize=(8,5))
            plt.scatter(fitted, resid, alpha=0.6)
            plt.axhline(0, color="red", linestyle="--")
            plt.xlabel("Fitted values (mixed model)")
            plt.ylabel("Residuals")
            plt.title("MixedLM residuals vs fitted")
            safe_plot_save(fig, out_dir / f"{report_prefix}seed_mixedlm_resid_plot.png")
    except Exception:
        pass

    # Markdown report
    md_lines = []
    md_lines.append("# Ablation Analysis Report\n")
    md_lines.append("## Inputs and locations\n")
    md_lines.append(f"- Seed-level summary CSV: **{summary_csv}**")
    md_lines.append(f"- Output folder: **{out_dir}**\n")
    md_lines.append(f"## Aggregated regression (groups with min_seeds >= {min_seeds})\n")
    md_lines.append("Regression results (OLS on aggregated group means):\n")
    md_lines.append(json.dumps(regression_result, indent=2))
    md_lines.append("\nBootstrap 95% CI for aggregated slope (percent scale):\n")
    md_lines.append(json.dumps(bootstrap_res, indent=2))
    md_lines.append("\nHigh-influence aggregated groups (Cook's D > 4/n):\n")
    if not high_influence.empty:
        for _, row in high_influence.iterrows():
            md_lines.append(f"- model: {row['model']}, math_ratio: {row['math_ratio']:.12f}, math_pct: {row['math_pct']:.2f}%, best_f1_mean: {row['best_f1_mean']:.6f}, n_seeds: {int(row['n_seeds'])}, cooks_d: {row['cooks_d']:.6e}")
    else:
        md_lines.append("- None (no high-influence points found)\n")
    md_lines.append("\n## Mixed-effects model (seed-level)\n")
    md_lines.append(f"MixedLM summary saved to: **{out_dir.name}/{report_prefix}mixedlm_summary.txt**")
    md_lines.append("\nMixedLM result (JSON):\n")
    md_lines.append(json.dumps(mixed_result, indent=2))
    md_lines.append("\n## Files produced\n")
    md_lines.append(f"- {out_dir.name}/{report_prefix}aggregated_by_model_ratio.csv")
    md_lines.append(f"- {out_dir.name}/{report_prefix}agg_regression_plot.png")
    md_lines.append(f"- {out_dir.name}/{report_prefix}regression_bestf1_vs_mathratio.json")
    md_lines.append(f"- {out_dir.name}/{report_prefix}bootstrap_slope_ci.json")
    md_lines.append(f"- {out_dir.name}/{report_prefix}cooks_distance.csv")
    md_lines.append(f"- {out_dir.name}/{report_prefix}mixedlm_summary.txt")
    md_lines.append(f"- {out_dir.name}/{report_prefix}mixedlm_result.json")
    md_lines.append("\n## Notes and interpretation guidance\n")
    md_lines.append("- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.")
    md_lines.append("- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.")
    md_lines.append("- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.")
    md_lines.append("- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.\n")

    with open(out_dir / f"{report_prefix}ablation_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n\n".join(md_lines))

    # Return results for programmatic use
    return {
        "df": df,
        "agg": agg,
        "agg_filtered": agg_filtered,
        "regression_result": regression_result,
        "bootstrap_res": bootstrap_res,
        "high_influence": high_influence,
        "mixed_result": mixed_result,
        "out_dir": out_dir
    }

# ---------- Extra analyses ----------
def run_wls_on_aggregated(agg_csv: Path, out_dir: Path, min_seeds: int = 2):
    """
    Weighted Least Squares on aggregated groups (weights = n_seeds).
    Writes JSON and plot to out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = pd.read_csv(agg_csv)
    agg["math_ratio"] = agg["math_ratio"].apply(parse_ratio)
    agg["math_pct"] = agg["math_ratio"] * 100.0
    agg_filtered = agg[agg["n_seeds"] >= min_seeds].copy()
    if agg_filtered.empty:
        raise ValueError("No aggregated groups with required min_seeds for WLS.")
    X = sm.add_constant(agg_filtered["math_pct"].values)
    y = agg_filtered["best_f1_mean"].values
    weights = agg_filtered["n_seeds"].values
    try:
        wls_model = sm.WLS(y, X, weights=weights).fit()
        result = {
            "slope": float(wls_model.params[1]),
            "intercept": float(wls_model.params[0]),
            "r_squared": float(wls_model.rsquared) if hasattr(wls_model, "rsquared") else None,
            "p_value": float(wls_model.pvalues[1]) if hasattr(wls_model, "pvalues") else None,
            "stderr": float(wls_model.bse[1]) if hasattr(wls_model, "bse") else None,
            "n": int(len(agg_filtered))
        }
        safe_write_json(out_dir / "wls_regression.json", result)
        # Plot
        fig = plt.figure(figsize=(9,6))
        sns.scatterplot(data=agg_filtered, x="math_pct", y="best_f1_mean", size="n_seeds", legend=False)
        x_vals = np.linspace(agg_filtered["math_pct"].min(), agg_filtered["math_pct"].max(), 200)
        y_vals = result["intercept"] + result["slope"] * x_vals
        plt.plot(x_vals, y_vals, color="green", label=f"WLS fit (slope={result['slope']:.6f})")
        plt.xlabel("Observed math ratio (%)")
        plt.ylabel("Mean Best Val F1 (per group)")
        plt.title("WLS aggregated regression (weights = n_seeds)")
        plt.legend()
        safe_plot_save(fig, out_dir / "wls_regression_plot.png")
        return result
    except Exception as e:
        raise

def run_mixedlm_interaction(summary_csv: Path, out_dir: Path):
    """
    Fit MixedLM with interaction: best_f1 ~ math_pct * C(model) with random intercepts for model.
    If MixedLM cannot fit interaction, fall back to OLS with interaction.
    Save summary text and a compact JSON of interaction terms.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(summary_csv)
    df["math_ratio"] = df["math_ratio"].apply(parse_ratio)
    df = df.dropna(subset=["math_ratio", "best_f1", "model"])
    df["math_pct"] = df["math_ratio"] * 100.0
    df["model"] = df["model"].astype(str)

    # Try MixedLM with interaction by creating interaction term manually and using random intercepts
    # MixedLM formula with interaction is tricky; attempt via patsy formula
    try:
        md = smf.mixedlm("best_f1 ~ math_pct * C(model)", df, groups=df["model"])
        mdf = md.fit(reml=False, method="lbfgs")
        summary_text = mdf.summary().as_text()
        with open(out_dir / "mixedlm_interaction_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_text)
        # Extract interaction terms from params
        interaction_terms = {}
        for term in mdf.params.index:
            if ":" in term or "C(model)" in term:
                coef = float(mdf.params[term])
                se = float(mdf.bse.get(term, np.nan))
                pval = float(mdf.pvalues.get(term, np.nan)) if hasattr(mdf, "pvalues") else None
                interaction_terms[term] = {"coef": coef, "se": se, "pvalue": pval}
        result = {"converged": True, "interaction_terms": interaction_terms}
        safe_write_json(out_dir / "mixedlm_interaction_result.json", result)
        return result
    except Exception as e_mixed:
        # Fallback to OLS with interaction (exploratory)
        try:
            ols = smf.ols("best_f1 ~ math_pct * C(model)", data=df).fit()
            summary_text = ols.summary().as_text()
            with open(out_dir / "mixedlm_interaction_summary.txt", "w", encoding="utf-8") as f:
                f.write("MixedLM failed; falling back to OLS with interaction.\n\n")
                f.write(summary_text)
            interaction_terms = {}
            for term in ols.params.index:
                if ":" in term or "C(model)" in term:
                    coef = float(ols.params[term])
                    se = float(ols.bse.get(term, np.nan))
                    pval = float(ols.pvalues.get(term, np.nan))
                    interaction_terms[term] = {"coef": coef, "se": se, "pvalue": pval}
            result = {"converged": False, "interaction_terms": interaction_terms, "fallback": "OLS"}
            safe_write_json(out_dir / "mixedlm_interaction_result.json", result)
            return result
        except Exception as e_ols:
            # write error
            err = {"error": "Both MixedLM and OLS interaction failed", "mixed_error": str(e_mixed), "ols_error": str(e_ols), "trace_mixed": traceback.format_exc()}
            safe_write_json(out_dir / "mixedlm_interaction_result.json", err)
            raise

def collect_and_plot_slopes(extra_out: Path, base_json: Path, sensitivity_json: Path, wls_result: dict, mixed_base_json: Path, mixed_sens_json: Path = None):
    """
    Collect slope estimates from various sources and produce a comparison CSV and bar chart.
    Writes slopes_comparison.csv and slopes_comparison.png to extra_out.
    """
    rows = []

    def load_json_safe(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

    base = load_json_safe(base_json) or {}
    sens = load_json_safe(sensitivity_json) or {}
    mixed_base = load_json_safe(mixed_base_json) or {}
    mixed_sens = load_json_safe(mixed_sens_json) if mixed_sens_json and mixed_sens_json.exists() else None

    # base OLS
    rows.append({
        "method": "base_OLS",
        "slope": base.get("slope"),
        "stderr": base.get("stderr"),
        "ci_lower": None,
        "ci_upper": None,
        "source_file": str(base_json)
    })
    # sensitivity OLS
    rows.append({
        "method": "sensitivity_OLS",
        "slope": sens.get("slope"),
        "stderr": sens.get("stderr"),
        "ci_lower": None,
        "ci_upper": None,
        "source_file": str(sensitivity_json)
    })
    # WLS
    rows.append({
        "method": "WLS",
        "slope": wls_result.get("slope") if wls_result else None,
        "stderr": wls_result.get("stderr") if wls_result else None,
        "ci_lower": None,
        "ci_upper": None,
        "source_file": str(extra_out / "wls_regression.json")
    })
    # mixedlm base
    rows.append({
        "method": "mixedlm_base",
        "slope": mixed_base.get("fixed_effect_math_pct_coef"),
        "stderr": mixed_base.get("fixed_effect_math_pct_se"),
        "ci_lower": None,
        "ci_upper": None,
        "source_file": str(mixed_base_json)
    })
    # mixedlm sensitivity (if present)
    if mixed_sens:
        rows.append({
            "method": "mixedlm_sensitivity",
            "slope": mixed_sens.get("fixed_effect_math_pct_coef"),
            "stderr": mixed_sens.get("fixed_effect_math_pct_se"),
            "ci_lower": None,
            "ci_upper": None,
            "source_file": str(mixed_sens_json)
        })

    df_slopes = pd.DataFrame(rows)
    df_slopes.to_csv(extra_out / "slopes_comparison.csv", index=False)

    # Plot bar chart with error bars where stderr available
    fig, ax = plt.subplots(figsize=(8,4))
    methods = df_slopes["method"].tolist()
    slopes = df_slopes["slope"].fillna(0).astype(float).tolist()
    stderrs = df_slopes["stderr"].fillna(0).astype(float).tolist()
    x = np.arange(len(methods))
    bars = ax.bar(x, slopes, yerr=stderrs, capsize=5, color="C0")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Slope (change in best_f1 per 1% math_pct)")
    ax.set_title("Comparison of slope estimates across methods")
    safe_plot_save(fig, extra_out / "slopes_comparison.png")
    return df_slopes

# ---------- Main execution ----------
def main():
    try:
        print("Starting ablation analysis pipeline")
        if not SUMMARY_CSV.exists():
            raise FileNotFoundError(f"Seed-level summary not found: {SUMMARY_CSV}")

        # 1) Run analysis on original summary
        print("Running analysis on original summary CSV...")
        base_results = run_full_analysis(SUMMARY_CSV, OUT_DIR, min_seeds=args.min_seeds, bootstrap_iters=args.bootstrap_iters, report_prefix="")

        # 2) Create filtered CSV excluding the three flagged aggregated groups (exact math_ratio values)
        filtered_csv = SUMMARY_CSV.parent / "ablation_summary_filtered_excl_high_influence.csv"
        df_seed = base_results["df"].copy()

        mask_drop = (
            ((df_seed["model"] == "Qwen2.5-1.5B") & (np.isclose(df_seed["math_ratio"].astype(float), 0.1220982142857142, atol=1e-9))) |
            ((df_seed["model"] == "Qwen2.5-Coder-1.5B") & (np.isclose(df_seed["math_ratio"].astype(float), 0.0, atol=1e-9))) |
            ((df_seed["model"] == "Qwen2.5-Math-1.5B") & (np.isclose(df_seed["math_ratio"].astype(float), 0.0, atol=1e-9)))
        )
        filtered_df = df_seed[~mask_drop].copy()
        filtered_df.to_csv(filtered_csv, index=False)
        print("Filtered CSV written:", filtered_csv)

        # 3) Run analysis on filtered CSV (exclude high-influence groups)
        sensitivity_out = OUT_DIR / "sensitivity_excluding_high_influence"
        print("Running analysis on filtered CSV (excluding flagged high-influence groups)...")
        sensitivity_results = run_full_analysis(filtered_csv, sensitivity_out, min_seeds=args.min_seeds, bootstrap_iters=args.bootstrap_iters, report_prefix="sensitivity_")

        # 4) Leave-one-out aggregated sensitivity (on aggregated groups used in base aggregated regression)
        loo_out = OUT_DIR / "leave_one_out"
        loo_out.mkdir(parents=True, exist_ok=True)
        agg_used = base_results["agg_filtered"].copy().reset_index(drop=True)
        loo_summary_rows = []
        print("Running leave-one-out aggregated sensitivity (dropping each aggregated group in turn)...")
        for idx, row in agg_used.iterrows():
            model_name = row["model"]
            math_ratio_val = float(row["math_ratio"])
            df_seed = base_results["df"].copy()
            drop_mask = ( (df_seed["model"] == model_name) & (np.isclose(df_seed["math_ratio"].astype(float), math_ratio_val, atol=1e-9)) )
            df_loo = df_seed[~drop_mask].copy()
            tmp_csv = loo_out / f"ablation_summary_loo_drop_{idx}_{model_name.replace('/','_')}_math_{math_ratio_val:.12f}.csv"
            df_loo.to_csv(tmp_csv, index=False)
            try:
                res = run_full_analysis(tmp_csv, loo_out, min_seeds=args.min_seeds, bootstrap_iters=500, report_prefix=f"loo_{idx}_")
                slope = res["regression_result"]["slope"]
                r2 = res["regression_result"]["r_squared"]
                n_groups = res["regression_result"]["n"]
                loo_summary_rows.append({
                    "dropped_idx": int(idx),
                    "dropped_model": model_name,
                    "dropped_math_ratio": math_ratio_val,
                    "slope": slope,
                    "r_squared": r2,
                    "n_groups": n_groups
                })
            except Exception as e:
                loo_summary_rows.append({
                    "dropped_idx": int(idx),
                    "dropped_model": model_name,
                    "dropped_math_ratio": math_ratio_val,
                    "slope": None,
                    "r_squared": None,
                    "n_groups": None,
                    "error": str(e)
                })

        loo_df = pd.DataFrame(loo_summary_rows)
        loo_df.to_csv(loo_out / "leave_one_out_summary.csv", index=False)

        # ---------- EXTRA ANALYSES ----------
        print("Running extra analyses (WLS, MixedLM interaction, slopes comparison)...")
        agg_csv = OUT_DIR / "aggregated_by_model_ratio.csv"
        # 1) WLS
        try:
            wls_result = run_wls_on_aggregated(agg_csv, EXTRA_DIR, min_seeds=args.min_seeds)
            print("WLS result written to:", EXTRA_DIR / "wls_regression.json")
        except Exception as e:
            wls_result = None
            safe_write_json(EXTRA_DIR / "error.json", {"step": "WLS", "error": str(e), "trace": traceback.format_exc()})
            print("WLS failed; error written to extra_analysis/error.json")

        # 2) MixedLM with interaction (seed-level)
        try:
            mixed_inter_result = run_mixedlm_interaction(SUMMARY_CSV, EXTRA_DIR)
            print("MixedLM interaction result written to:", EXTRA_DIR / "mixedlm_interaction_result.json")
        except Exception as e:
            mixed_inter_result = None
            safe_write_json(EXTRA_DIR / "error.json", {"step": "mixedlm_interaction", "error": str(e), "trace": traceback.format_exc()})
            print("MixedLM interaction failed; error written to extra_analysis/error.json")

        # 3) Slopes comparison
        try:
            base_json = OUT_DIR / "regression_bestf1_vs_mathratio.json"
            sens_json = sensitivity_out / "sensitivity_regression_bestf1_vs_mathratio.json"
            mixed_base_json = OUT_DIR / "mixedlm_result.json"
            mixed_sens_json = sensitivity_out / "sensitivity_mixedlm_result.json"
            df_slopes = collect_and_plot_slopes(EXTRA_DIR, base_json, sens_json, wls_result or {}, mixed_base_json, mixed_sens_json)
            print("Slopes comparison written to:", EXTRA_DIR / "slopes_comparison.csv")
        except Exception as e:
            safe_write_json(EXTRA_DIR / "error.json", {"step": "slopes_comparison", "error": str(e), "trace": traceback.format_exc()})
            print("Slopes comparison failed; error written to extra_analysis/error.json")

        # 5) Final summary print
        print("\n=== Ablation analysis complete ===")
        print(f"Seed summary read from: {SUMMARY_CSV}")
        print(f"Aggregated table written to: {OUT_DIR / 'aggregated_by_model_ratio.csv'}")
        print(f"Aggregated regression JSON: {OUT_DIR / 'regression_bestf1_vs_mathratio.json'}")
        print(f"Bootstrap slope CI JSON: {OUT_DIR / 'bootstrap_slope_ci.json'}")
        print(f"Cook's distance CSV: {OUT_DIR / 'cooks_distance.csv'}")
        print(f"MixedLM summary: {OUT_DIR / 'mixedlm_summary.txt'}")
        print(f"Markdown report: {OUT_DIR / 'ablation_analysis_report.md'}")
        print(f"\nFiltered CSV (excluded high-influence groups): {filtered_csv}")
        print(f"Sensitivity outputs (excluding high-influence groups): {sensitivity_out}")
        print(f"Leave-one-out outputs: {loo_out}")
        print(f"Leave-one-out summary CSV: {loo_out / 'leave_one_out_summary.csv'}")
        print(f"Extra analysis outputs: {EXTRA_DIR}")

        # Print high-influence groups from base run
        hi = base_results["high_influence"]
        if not hi.empty:
            print("\nHigh influence groups (Cook's D > 4/n) from base aggregated regression:")
            for _, r in hi.iterrows():
                print(f" - model={r['model']}, math_pct={r['math_pct']:.2f}%, best_f1_mean={r['best_f1_mean']:.6f}, cooks_d={r['cooks_d']:.6e}")
        else:
            print("\nNo high-influence aggregated groups found in base run.")

    except Exception as e_main:
        # Write top-level error file
        err = {"error": str(e_main), "trace": traceback.format_exc()}
        safe_write_json(EXTRA_DIR / "error.json", err)
        print("Fatal error during analysis. See experiments/plots/extra_analysis/error.json for details.")
        raise

if __name__ == "__main__":
    main()
