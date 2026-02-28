#!/usr/bin/env python3
"""
Compute symbol_score for each example, produce diagnostics, and create matched halves for augmentation.

Outputs:
- experiments/plots/extra_analysis/symbol_scores.csv
- experiments/plots/extra_analysis/matched_pool_A.csv
- experiments/plots/extra_analysis/matched_pool_B.csv
- experiments/plots/extra_analysis/symbol_diagnostics.txt
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re
import json

SYMBOLS = list("0123456789+-*/=()[]{}:`\\") + ["\\(", "\\)", "\\frac"]

def symbol_score_text(text, symbol_regex):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    tokens = text.split()
    if len(tokens) == 0:
        return 0.0
    # proportion of tokens containing any symbol
    has_symbol = [1 if symbol_regex.search(tok) else 0 for tok in tokens]
    return float(sum(has_symbol)) / len(tokens)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="experiments/ablation_summary.csv")
    p.add_argument("--out", default="experiments/plots/extra_analysis")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.summary)
    # Choose a text field to score. If you have raw input text column, replace 'label' with that column name.
    # For diagnostic purposes we will use 'label' and 'folder' as proxies if raw text not available.
    text_col = None
    for candidate in ["text", "prompt", "input", "label", "folder"]:
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        raise RuntimeError("No text-like column found in summary CSV. Add a column with example text or prompts.")

    # Build regex for symbols
    # Escape special chars and include LaTeX tokens
    symbol_patterns = [re.escape(s) for s in ["0","1","2","3","4","5","6","7","8","9","+","-","*","/","=","(",")","[","]","{","}","`",":","\\"]]
    # include \frac and \( \)
    symbol_patterns += [r"\\frac", r"\\\(", r"\\\)"]
    symbol_regex = re.compile("|".join(symbol_patterns))

    df["symbol_score"] = df[text_col].fillna("").astype(str).apply(lambda t: symbol_score_text(t, symbol_regex))
    df["has_symbol"] = (df["symbol_score"] > 0.05).astype(int)  # threshold adjustable

    # Save symbol scores
    df[["label","folder","model","math_ratio","best_f1","symbol_score","has_symbol"]].to_csv(outdir / "symbol_scores.csv", index=False)

    # Diagnostics: distribution and stratified F1
    diag = {}
    diag["n_total"] = int(len(df))
    diag["mean_symbol_score"] = float(df["symbol_score"].mean())
    diag["pct_has_symbol"] = float(df["has_symbol"].mean())
    diag["mean_f1_symbol"] = float(df.loc[df["has_symbol"]==1, "best_f1"].mean())
    diag["mean_f1_no_symbol"] = float(df.loc[df["has_symbol"]==0, "best_f1"].mean())
    diag["count_symbol"] = int(df["has_symbol"].sum())
    diag["count_no_symbol"] = int((df["has_symbol"]==0).sum())

    with open(outdir / "symbol_diagnostics.txt", "w") as f:
        f.write(json.dumps(diag, indent=2))

    # Create matched pool for augmentation
    # Use only examples with labels and best_f1 present
    pool = df.dropna(subset=["best_f1"]).copy()
    # For matching, use columns: model, label (if available), length bucket
    pool["length"] = pool[text_col].astype(str).apply(lambda s: len(s.split()))
    pool["len_bucket"] = pd.qcut(pool["length"], q=4, duplicates="drop").astype(str)

    # Create a key for matching
    match_cols = []
    if "model" in pool.columns:
        match_cols.append("model")
    if "label" in pool.columns:
        match_cols.append("label")
    match_cols.append("len_bucket")

    # Shuffle and group
    pool = pool.sample(frac=1, random_state=42).reset_index(drop=True)
    # Group by match key and split each group into two halves
    pool["group_key"] = pool[match_cols].astype(str).agg("|".join, axis=1)
    matched_A = []
    matched_B = []
    for key, g in pool.groupby("group_key"):
        n = len(g)
        half = n // 2
        matched_A.append(g.iloc[:half])
        matched_B.append(g.iloc[half:half*2])  # ensure equal sizes
    if matched_A:
        matched_A = pd.concat(matched_A, ignore_index=True)
    else:
        matched_A = pool.iloc[0:0]
    if matched_B:
        matched_B = pd.concat(matched_B, ignore_index=True)
    else:
        matched_B = pool.iloc[0:0]

    matched_A.to_csv(outdir / "matched_pool_A.csv", index=False)
    matched_B.to_csv(outdir / "matched_pool_B.csv", index=False)

    print("Wrote symbol diagnostics and matched pools to", outdir)

if __name__ == "__main__":
    main()
