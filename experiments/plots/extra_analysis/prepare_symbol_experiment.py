#!/usr/bin/env python3
"""
prepare_symbol_experiment.py

One-step reproducible preparation for the symbol manipulation experiment.

What it does (idempotent):
1. Loads experiments/ablation_summary.csv
2. Computes a conservative symbol_score per example
3. Writes symbol_scores.csv and symbol_diagnostics.txt
4. Creates matched_pool_A.csv and matched_pool_B.csv using a safe fallback:
   - Primary: group-aware matching (model + length bucket)
   - Fallback: random split if grouping yields no pairs
5. Produces treatment_examples.csv and control_examples.csv using either:
   - append mode (append neutral symbol sequence), or
   - mask mode (replace symbols with <SYM>)
6. Prints a small preview (first 5 original vs treatment rows)

Usage (from project root):
python experiments/plots/extra_analysis/prepare_symbol_experiment.py --summary experiments/ablation_summary.csv --out experiments/plots/extra_analysis --mode append --text-col label

Options:
--summary   Path to summary CSV (default experiments/ablation_summary.csv)
--out       Output directory (default experiments/plots/extra_analysis)
--mode      'append' or 'mask' (default append)
--text-col  Column name that contains the example text (default: auto-detect)
--seed      Random seed for reproducibility (default 42)
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys

def compute_symbol_score(text, multi_tokens=None, single_chars=None):
    s = str(text) if pd.notna(text) else ""
    # Replace multi-token occurrences first
    for tok in (multi_tokens or []):
        s = s.replace(tok, " <SYM> ")
    # Count tokens and tokens containing any single-char symbol
    tokens = s.split()
    if len(tokens) == 0:
        return 0.0
    has_sym = 0
    for tok in tokens:
        if any(ch in tok for ch in (single_chars or "")):
            has_sym += 1
    return float(has_sym) / len(tokens)

def safe_group_match(df, match_cols, q=4):
    # Add length bucket
    df = df.copy()
    df["length"] = df["__text__"].astype(str).apply(lambda s: len(s.split()))
    try:
        df["len_bucket"] = pd.qcut(df["length"], q=q, duplicates="drop").astype(str)
    except Exception:
        df["len_bucket"] = pd.cut(df["length"], bins=2).astype(str)
    # Build group key
    keys = []
    for c in match_cols:
        if c in df.columns:
            keys.append(c)
    if not keys:
        keys = ["len_bucket"]
    df["group_key"] = df[keys].astype(str).agg("|".join, axis=1)
    A_parts = []
    B_parts = []
    for _, g in df.groupby("group_key"):
        n = len(g)
        if n >= 2:
            half = n // 2
            A_parts.append(g.iloc[:half])
            B_parts.append(g.iloc[half:half*2])
    if A_parts and B_parts:
        A = pd.concat(A_parts, ignore_index=True)
        B = pd.concat(B_parts, ignore_index=True)
    else:
        # fallback: random split
        df_shuf = df.sample(frac=1, random_state=42).reset_index(drop=True)
        half = len(df_shuf) // 2
        A = df_shuf.iloc[:half].reset_index(drop=True)
        B = df_shuf.iloc[half:half*2].reset_index(drop=True)
    return A, B

def apply_mask_single(text, single_chars, multi_map):
    s = str(text)
    for k, v in multi_map.items():
        s = s.replace(k, v)
    # replace single characters by <SYM>
    out_chars = []
    for ch in s:
        out_chars.append("<SYM>" if ch in single_chars else ch)
    return "".join(out_chars)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="experiments/ablation_summary.csv")
    p.add_argument("--out", default="experiments/plots/extra_analysis")
    p.add_argument("--mode", choices=["append","mask"], default="append")
    p.add_argument("--symbol-seq", default=" [ ( ) : ]")
    p.add_argument("--text-col", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load summary
    try:
        df = pd.read_csv(args.summary)
    except Exception as e:
        print("ERROR: cannot read summary CSV:", e, file=sys.stderr)
        sys.exit(1)

    # Detect text column
    text_col = args.text_col
    if text_col is None:
        for cand in ["text","prompt","input","label","folder"]:
            if cand in df.columns:
                text_col = cand
                break
    if text_col is None:
        print("ERROR: no text column found; pass --text-col", file=sys.stderr)
        sys.exit(1)

    # Prepare working copy with unified text column name
    df_work = df.copy()
    df_work["__text__"] = df_work[text_col].astype(str)

    # Symbol definitions
    single_chars = "0123456789+-*/=()[]{}:`:\\"
    multi_tokens = ["\\frac", "\\(", "\\)"]
    multi_map = {tok: " <SYM> " for tok in multi_tokens}

    # Compute symbol_score
    df_work["symbol_score"] = df_work["__text__"].apply(lambda t: compute_symbol_score(t, multi_tokens=multi_tokens, single_chars=single_chars))
    df_work["has_symbol"] = (df_work["symbol_score"] > 0.05).astype(int)

    # Write symbol_scores and diagnostics
    scores_cols = [c for c in df.columns] + ["symbol_score", "has_symbol"]
    df_work[scores_cols].to_csv(Path(outdir) / "symbol_scores.csv", index=False)

    diag = {
        "n_total": int(len(df_work)),
        "mean_symbol_score": float(df_work["symbol_score"].mean()),
        "pct_has_symbol": float(df_work["has_symbol"].mean()),
        "mean_f1_symbol": float(df_work.loc[df_work["has_symbol"]==1, "best_f1"].mean()) if df_work["has_symbol"].sum()>0 else None,
        "mean_f1_no_symbol": float(df_work.loc[df_work["has_symbol"]==0, "best_f1"].mean()) if (df_work["has_symbol"]==0).sum()>0 else None,
        "count_symbol": int(df_work["has_symbol"].sum()),
        "count_no_symbol": int((df_work["has_symbol"]==0).sum())
    }
    with open(Path(outdir) / "symbol_diagnostics.txt", "w") as f:
        json.dump(diag, f, indent=2)

    # Create matched pools (group-aware, fallback to random split)
    match_cols = []
    if "model" in df_work.columns:
        match_cols.append("model")
    # include len_bucket via function
    A, B = safe_group_match(df_work, match_cols, q=4)

    # Save matched pools
    A.to_csv(Path(outdir) / "matched_pool_A.csv", index=False)
    B.to_csv(Path(outdir) / "matched_pool_B.csv", index=False)

    # If matched pools are empty, fallback to symbol-rich split
    if len(A) == 0 or len(B) == 0:
        sym = df_work[df_work["symbol_score"] > 0.05].sample(frac=1, random_state=args.seed).reset_index(drop=True)
        half = len(sym) // 2
        A = sym.iloc[:half].reset_index(drop=True)
        B = sym.iloc[half:half*2].reset_index(drop=True)
        A.to_csv(Path(outdir) / "matched_pool_A.csv", index=False)
        B.to_csv(Path(outdir) / "matched_pool_B.csv", index=False)

    # Create treatment and control files
    # Control = B (unchanged), Treatment = A (modified)
    control = B.copy().reset_index(drop=True)
    treatment = A.copy().reset_index(drop=True)

    if args.mode == "mask":
        treatment[text_col] = treatment["__text__"].apply(lambda t: apply_mask_single(t, single_chars, multi_map))
    else:
        treatment[text_col] = treatment["__text__"].astype(str) + args.symbol_seq

    # Keep original columns layout for outputs (use original df columns)
    out_cols = [c for c in df.columns]
    # If text_col not in out_cols (unlikely), ensure it's present
    if text_col not in out_cols:
        out_cols = [text_col] + out_cols

    treatment[out_cols].to_csv(Path(outdir) / "treatment_examples.csv", index=False)
    control[out_cols].to_csv(Path(outdir) / "control_examples.csv", index=False)

    # Preview first 5 examples side-by-side
    n_preview = min(5, len(A), len(treatment))
    print(f"Wrote files to {outdir}")
    print(f"Matched pool A rows: {len(A)}; B rows: {len(B)}; treatment rows: {len(treatment)}; control rows: {len(control)}")
    print("Symbol diagnostics:", json.dumps(diag, indent=2))
    print("\nPreview (first {} examples):".format(n_preview))
    for i in range(n_preview):
        orig = A.iloc[i][text_col]
        mod = treatment.iloc[i][text_col]
        print("---- example", i, "----")
        print("ORIG :", orig)
        print("TREAT:", mod)
        print()

if __name__ == "__main__":
    main()
