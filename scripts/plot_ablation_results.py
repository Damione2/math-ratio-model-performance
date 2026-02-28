#!/usr/bin/env python3
# scripts/plot_ablation_results.py
"""
Recursive ablation plotter (Option C, corrected)

- Recursively scans experiments/ for folders containing both:
    training_summary.json and training_log.csv
- Reads math_ratio from training_summary.json (fallback to folder name)
- Treats each such folder as a seed-level experiment
- Aggregates by (model_name, math_ratio) to compute mean/std across seeds
- Produces:
    - per-seed scatter plots
    - aggregated mean ± std plots with error bars
    - regression best_f1 ~ math_ratio
    - epoch curves (val_f1, math_acc, code_acc, real_acc)
    - summary CSV/JSON and Markdown report
Usage:
    python scripts/plot_ablation_results.py --root experiments --out experiments/plots
"""
from pathlib import Path
import json
import argparse
import sys
import traceback
import warnings
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

sns.set(style="whitegrid")

# ---------- Helpers ----------
def safe_load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        warnings_list.append(f"Failed to load JSON {path}: {e}")
        return None

def safe_load_csv(path: Path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        warnings_list.append(f"Failed to load CSV {path}: {e}")
        return None

def infer_math_ratio_from_name(name: str):
    lower = name.lower()
    for sep in ["math_", "math-", "math", "m_"]:
        if lower.startswith(sep):
            tail = lower[len(sep):]
            digits = ''.join(ch for ch in tail if ch.isdigit())
            if digits:
                try:
                    pct = int(digits)
                    return float(pct) / 100.0
                except Exception:
                    pass
    m = re.search(r"(\d{1,3})", name)
    if m:
        try:
            pct = int(m.group(1))
            return float(pct) / 100.0
        except Exception:
            pass
    return None

def flatten_name(folder: Path):
    parts = folder.parts
    try:
        idx = parts.index(EXPERIMENTS_DIR.name)
        rel = parts[idx+1:]
    except Exception:
        rel = parts[-3:]
    label = "_".join(rel)
    return label.replace(" ", "_")

def parse_model_and_ratio_from_path(folder: Path, summary_obj):
    # Prefer math_ratio from training_summary.json
    ratio = None
    if isinstance(summary_obj, dict):
        ratio = summary_obj.get("math_ratio", None)
    # model detection from path
    parts = folder.parts
    model = None
    for p in parts[::-1]:
        if re.search(r"qwen|unsloth", p, re.IGNORECASE):
            model = p
            break
    if model is None and len(parts) >= 2:
        model = parts[-2]
    # fallback ratio from path if not present in summary
    if ratio is None:
        ratio = infer_math_ratio_from_name(folder.name)
        if ratio is None:
            for p in parts[::-1]:
                ratio = infer_math_ratio_from_name(p)
                if ratio is not None:
                    break
    return model, ratio

def save_fig(fig, path: Path, w=12, h=8):
    fig.set_size_inches(w, h)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ---------- Argument parsing ----------
p = argparse.ArgumentParser()
p.add_argument("--root", default="experiments", help="Root experiments directory")
p.add_argument("--out", default="experiments/plots", help="Output plots directory")
p.add_argument("--min-seeds", type=int, default=1, help="Minimum seeds required to compute aggregated stats")
args = p.parse_args()

EXPERIMENTS_DIR = Path(args.root).resolve()
PLOTS_DIR = Path(args.out).resolve()
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = EXPERIMENTS_DIR / "ablation_summary.csv"
SUMMARY_JSON = EXPERIMENTS_DIR / "ablation_summary.json"
REGRESSION_JSON = PLOTS_DIR / "regression_bestf1_vs_mathratio.json"
REPORT_MD = PLOTS_DIR / "ablation_report.md"

warnings_list = []
seed_runs = []  # list of dicts: {folder, model, ratio, label, summary, log_df}

# ---------- Discover runs recursively ----------
for folder in sorted(EXPERIMENTS_DIR.rglob("*")):
    if not folder.is_dir():
        continue
    summary_path = folder / "training_summary.json"
    log_path = folder / "training_log.csv"
    if summary_path.exists() and log_path.exists():
        summary = safe_load_json(summary_path)
        log_df = safe_load_csv(log_path)
        if summary is None or log_df is None:
            continue
        log_df.columns = [c.strip() for c in log_df.columns]
        if "epoch" not in log_df.columns:
            log_df = log_df.reset_index().rename(columns={"index": "epoch"})
            log_df["epoch"] = log_df["epoch"].astype(int)
        model, ratio = parse_model_and_ratio_from_path(folder, summary)
        label = flatten_name(folder)
        seed_runs.append({
            "folder": folder,
            "label": label,
            "model": model,
            "ratio": ratio,
            "summary": summary,
            "log_df": log_df.sort_values("epoch").reset_index(drop=True)
        })

if not seed_runs:
    print(f"No valid runs found under {EXPERIMENTS_DIR}. Each run must contain training_summary.json and training_log.csv")
    sys.exit(1)

# ---------- Build seed-level summary rows ----------
rows = []
for r in seed_runs:
    s = r["summary"]
    log_df = r["log_df"]
    best_epoch = s.get("best_epoch")
    best_f1 = s.get("best_f1")
    if best_epoch is None or best_f1 is None:
        if "val_f1" in log_df.columns:
            idx = log_df["val_f1"].idxmax()
            best_epoch = int(log_df.loc[idx, "epoch"])
            best_f1 = float(log_df.loc[idx, "val_f1"])
        else:
            warnings_list.append(f"Run {r['label']} missing val_f1 in log; skipping")
            continue
    row_match = log_df[log_df["epoch"] == best_epoch]
    if row_match.empty:
        idx = (log_df["epoch"] - best_epoch).abs().idxmin()
        row = log_df.loc[idx]
    else:
        row = row_match.iloc[0]
    def safe_get(col):
        return float(row[col]) if col in row and not pd.isna(row[col]) else float("nan")
    math_acc = safe_get("math_acc") if "math_acc" in log_df.columns else float("nan")
    code_acc = safe_get("code_acc") if "code_acc" in log_df.columns else float("nan")
    real_acc = safe_get("real_acc") if "real_acc" in log_df.columns else float("nan")
    val_ece = safe_get("val_ece") if "val_ece" in log_df.columns else float("nan")
    n_epochs = int(len(log_df))
    rows.append({
        "label": r["label"],
        "folder": str(r["folder"].resolve()),
        "model": r["model"],
        "math_ratio": r["ratio"],
        "best_f1": float(best_f1),
        "best_epoch": int(best_epoch),
        "math_acc_at_best_epoch": math_acc,
        "code_acc_at_best_epoch": code_acc,
        "real_acc_at_best_epoch": real_acc,
        "val_ece_at_best_epoch": val_ece,
        "n_epochs": n_epochs
    })

summary_df = pd.DataFrame(rows)
summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8")
with SUMMARY_JSON.open("w", encoding="utf-8") as f:
    json.dump(summary_df.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

# ---------- Aggregation by (model, ratio) ----------
agg_groups = []
grouped = summary_df.groupby(["model", "math_ratio"])
for (model, ratio), g in grouped:
    if g.shape[0] < args.min_seeds:
        warnings_list.append(f"Group (model={model}, ratio={ratio}) has {g.shape[0]} seeds (< min_seeds); still included but flagged.")
    agg = {
        "model": model,
        "math_ratio": ratio,
        "n_seeds": int(g.shape[0]),
        "best_f1_mean": float(g["best_f1"].mean()),
        "best_f1_std": float(g["best_f1"].std(ddof=0)) if g.shape[0] > 1 else 0.0,
        "math_acc_mean": float(g["math_acc_at_best_epoch"].mean(skipna=True)),
        "code_acc_mean": float(g["code_acc_at_best_epoch"].mean(skipna=True)),
        "real_acc_mean": float(g["real_acc_at_best_epoch"].mean(skipna=True))
    }
    agg_groups.append(agg)
agg_df = pd.DataFrame(agg_groups).sort_values(["math_ratio", "model"]).reset_index(drop=True)

# ---------- Plotting ----------
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def pct(x):
    return int(round(x * 100)) if x is not None and not pd.isna(x) else None

# Plot 1: per-seed scatter (math_pct vs best_f1)
try:
    fig, ax = plt.subplots()
    df = summary_df.copy()
    df["math_pct"] = df["math_ratio"].apply(lambda x: pct(x))
    sns.scatterplot(data=df, x="math_pct", y="best_f1", hue="model", style="model", s=80, ax=ax)
    for _, r in df.iterrows():
        if pd.notnull(r["math_pct"]):
            ax.text(r["math_pct"], r["best_f1"], f" {r['label']}", fontsize=8, verticalalignment="center")
    ax.set_xlabel("Math ratio (%)")
    ax.set_ylabel("Best Val F1")
    ax.set_title("Per-seed: Math ratio vs Best Val F1")
    save_fig(fig, PLOTS_DIR / "per_seed_mathratio_vs_bestf1.png")
except Exception as e:
    warnings_list.append(f"Failed per-seed scatter: {e}")

# Plot 2: aggregated mean ± std per (model, ratio)
try:
    fig, ax = plt.subplots()
    if not agg_df.empty:
        agg_df["math_pct"] = agg_df["math_ratio"].apply(lambda x: pct(x))
        for model_name, grp in agg_df.groupby("model"):
            ax.errorbar(grp["math_pct"], grp["best_f1_mean"], yerr=grp["best_f1_std"], label=str(model_name), marker="o", capsize=4)
        ax.set_xlabel("Math ratio (%)")
        ax.set_ylabel("Mean Best Val F1 (± std)")
        ax.set_title("Aggregated: Mean Best F1 ± Std by Model and Math Ratio")
        ax.legend(loc="best", fontsize="small")
        save_fig(fig, PLOTS_DIR / "agg_meanstd_mathratio_vs_bestf1.png")
except Exception as e:
    warnings_list.append(f"Failed aggregated mean±std plot: {e}")

# Plot 3: regression best_f1 ~ math_pct (using aggregated means if available, else seeds)
regression_result = {}
try:
    reg_df = agg_df if not agg_df.empty else summary_df
    if not reg_df.empty:
        if "best_f1_mean" in reg_df.columns:
            x = reg_df["math_ratio"].apply(lambda v: pct(v)).astype(float).values
            y = reg_df["best_f1_mean"].astype(float).values
        else:
            x = reg_df["math_ratio"].apply(lambda v: pct(v)).astype(float).values
            y = reg_df["best_f1"].astype(float).values
        if len(x) >= 2:
            res = linregress(x, y)
            regression_result = {
                "slope": res.slope,
                "intercept": res.intercept,
                "r_value": res.rvalue,
                "r_squared": res.rvalue ** 2,
                "p_value": res.pvalue,
                "stderr": res.stderr,
                "n": int(len(x))
            }
        else:
            regression_result = {"error": "Not enough points for regression", "n": int(len(x))}
    else:
        regression_result = {"error": "No numeric points for regression"}
except Exception as e:
    regression_result = {"error": str(e)}
with open(REGRESSION_JSON, "w", encoding="utf-8") as f:
    json.dump(regression_result, f, indent=2, ensure_ascii=False)

# Plot 4: epoch curves (val_f1) — one line per seed (subset if too many)
try:
    fig, ax = plt.subplots()
    max_lines = 40
    for i, item in enumerate(seed_runs):
        if i >= max_lines:
            break
        df = item["log_df"]
        if "val_f1" in df.columns:
            ax.plot(df["epoch"], df["val_f1"], label=item["label"], linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1")
    ax.set_title("Validation F1 over Epochs (per-seed)")
    if len(seed_runs) <= max_lines:
        ax.legend(loc="best", fontsize="small")
    save_fig(fig, PLOTS_DIR / "val_f1_epochs_per_seed.png", w=14, h=9)
except Exception as e:
    warnings_list.append(f"Failed val_f1 epoch plot: {e}")

# Plot 5: math_acc over epochs (per-seed)
try:
    fig, ax = plt.subplots()
    for i, item in enumerate(seed_runs):
        if i >= 40:
            break
        df = item["log_df"]
        if "math_acc" in df.columns:
            ax.plot(df["epoch"], df["math_acc"], label=item["label"], linewidth=1.0, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Math Accuracy")
    ax.set_title("Math Accuracy over Epochs (per-seed)")
    if len(seed_runs) <= 40:
        ax.legend(loc="best", fontsize="small")
    save_fig(fig, PLOTS_DIR / "math_acc_epochs_per_seed.png", w=14, h=9)
except Exception as e:
    warnings_list.append(f"Failed math_acc epoch plot: {e}")

# Plot 6: per-model aggregated curves (mean ± std) for val_f1 across epochs
try:
    model_epoch_agg = {}
    for item in seed_runs:
        model = item["model"] or "unknown"
        ratio = item["ratio"]
        key = (model, ratio)
        cols = ["val_f1", "math_acc", "code_acc", "real_acc"]
        available = ["epoch"] + [c for c in cols if c in item["log_df"].columns]
        df = item["log_df"][available].copy().set_index("epoch")
        model_epoch_agg.setdefault(key, []).append(df)
    for key, frames in model_epoch_agg.items():
        concat = pd.concat(frames, axis=1, keys=range(len(frames)))
        # use transpose grouping to avoid deprecated axis=1 usage
        mean_df = concat.T.groupby(level=1).mean().T
        std_df = concat.T.groupby(level=1).std(ddof=0).T
        model_epoch_agg[key] = (mean_df, std_df)
    keys = list(model_epoch_agg.keys())[:9]
    fig, ax = plt.subplots()
    for key in keys:
        mean_df, std_df = model_epoch_agg[key]
        if "val_f1" in mean_df.columns:
            x = mean_df.index.values
            y = mean_df["val_f1"].values
            yerr = std_df["val_f1"].values if "val_f1" in std_df.columns else None
            label = f"{key[0]}_math{pct(key[1]) if key[1] is not None else 'NA'}"
            ax.plot(x, y, label=label)
            if yerr is not None:
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val F1 (mean ± std)")
    ax.set_title("Aggregated Val F1 over Epochs (per model, ratio)")
    ax.legend(loc="best", fontsize="small")
    save_fig(fig, PLOTS_DIR / "agg_val_f1_epochs_by_model_ratio.png", w=14, h=9)
except Exception as e:
    warnings_list.append(f"Failed aggregated epoch curves: {e}")

# ---------- Markdown report ----------
try:
    md_lines = []
    md_lines.append("# Ablation Summary Report\n")
    md_lines.append("## Seed-level summary\n")
    md_table = summary_df[[
        "label", "model", "math_ratio", "best_f1", "best_epoch",
        "math_acc_at_best_epoch", "code_acc_at_best_epoch", "real_acc_at_best_epoch", "n_epochs"
    ]].copy()
    md_table["math_ratio"] = md_table["math_ratio"].apply(lambda x: f"{pct(x)}%" if pd.notnull(x) else "N/A")
    md_lines.append(md_table.to_markdown(index=False))
    md_lines.append("\n## Aggregated summary (by model, math_ratio)\n")
    if not agg_df.empty:
        agg_table = agg_df[["model", "math_ratio", "n_seeds", "best_f1_mean", "best_f1_std"]].copy()
        agg_table["math_ratio"] = agg_table["math_ratio"].apply(lambda x: f"{pct(x)}%" if pd.notnull(x) else "N/A")
        md_lines.append(agg_table.to_markdown(index=False))
    md_lines.append("\n## Regression: best_f1 ~ math_ratio\n")
    if "error" in regression_result:
        md_lines.append(f"Regression not available: {regression_result.get('error')}\n")
    else:
        md_lines.append(f"- **slope**: {regression_result['slope']:.6f}\n")
        md_lines.append(f"- **intercept**: {regression_result['intercept']:.6f}\n")
        md_lines.append(f"- **r_value**: {regression_result['r_value']:.6f}\n")
        md_lines.append(f"- **r_squared**: {regression_result['r_squared']:.6f}\n")
        md_lines.append(f"- **p_value**: {regression_result['p_value']:.6e}\n")
        md_lines.append(f"- **stderr**: {regression_result['stderr']:.6f}\n")
        md_lines.append(f"- **n**: {regression_result['n']}\n")
    md_lines.append("\n## Generated plots\n")
    md_lines.append("- `plots/per_seed_mathratio_vs_bestf1.png`")
    md_lines.append("- `plots/agg_meanstd_mathratio_vs_bestf1.png`")
    md_lines.append("- `plots/val_f1_epochs_per_seed.png`")
    md_lines.append("- `plots/math_acc_epochs_per_seed.png`")
    md_lines.append("- `plots/agg_val_f1_epochs_by_model_ratio.png`\n")
    md_lines.append("## Notes and warnings\n")
    if warnings_list:
        md_lines.append("Warnings encountered during processing:\n")
        for w in warnings_list:
            md_lines.append(f"- {w}\n")
    else:
        md_lines.append("No warnings.\n")
    with REPORT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
except Exception as e:
    warnings_list.append(f"Failed to write markdown report: {e}")

# ---------- Final print ----------
print("\n=== Ablation aggregation complete ===")
print(f"Seed runs processed: {len(seed_runs)}")
print(f"Seed-level summary CSV: {SUMMARY_CSV}")
print(f"Seed-level summary JSON: {SUMMARY_JSON}")
print(f"Plots directory: {PLOTS_DIR}")
print(f"Regression JSON: {REGRESSION_JSON}")
print(f"Markdown report: {REPORT_MD}")
if warnings_list:
    print("\nWarnings:")
    for w in warnings_list:
        print(" -", w)
