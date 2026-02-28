#!/usr/bin/env python3
"""
Robust symbol augmentation / masking using translate + dictionary replacements.
Writes:
- experiments/plots/extra_analysis/treatment_examples.csv
- experiments/plots/extra_analysis/control_examples.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import sys

def build_trans_table(mask_token="<SYM>"):
    # Characters to mask or replace (single-char)
    single_chars = "0123456789+-*/=()[]{}:`:\\"
    # For translate, map each char to mask_token or to itself when appending mode
    trans_mask = {ord(c): mask_token for c in single_chars}
    return trans_mask

def apply_mask(text, trans_table, multi_map):
    s = str(text)
    # First replace multi-char tokens (longer tokens first)
    for k, v in multi_map.items():
        s = s.replace(k, v)
    # Then apply single-char translate
    s = s.translate(trans_table)
    return s

def apply_append(text, append_seq):
    return str(text) + append_seq

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matched-a", default="experiments/plots/extra_analysis/matched_pool_A.csv")
    p.add_argument("--matched-b", default="experiments/plots/extra_analysis/matched_pool_B.csv")
    p.add_argument("--out", default="experiments/plots/extra_analysis")
    p.add_argument("--mode", choices=["append","mask"], default="append")
    p.add_argument("--symbol-seq", default=" [ ( ) : ]")
    p.add_argument("--text-col", default=None, help="Column name containing text (required if auto-detect fails)")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        a = pd.read_csv(args.matched_a)
        b = pd.read_csv(args.matched_b)
    except Exception as e:
        print("Error reading matched_pool files:", e, file=sys.stderr)
        sys.exit(1)

    text_col = args.text_col
    if text_col is None:
        for c in ["text","prompt","input","label","folder"]:
            if c in a.columns:
                text_col = c
                break
    if text_col is None:
        print("No text-like column found. Provide --text-col", file=sys.stderr)
        sys.exit(1)

    # Multi-character tokens to replace (map to <SYM> for mask mode)
    multi_map = {
        "\\frac": "<SYM>",
        "\\(": "<SYM>",
        "\\)": "<SYM>"
    }

    trans_table = build_trans_table(mask_token="<SYM>")

    treatment = a.copy()
    control = b.copy()

    if args.mode == "mask":
        treatment[text_col] = treatment[text_col].apply(lambda t: apply_mask(t, trans_table, multi_map))
    else:
        treatment[text_col] = treatment[text_col].astype(str).apply(lambda s: apply_append(s, args.symbol_seq))

    treatment.to_csv(outdir / "treatment_examples.csv", index=False)
    control.to_csv(outdir / "control_examples.csv", index=False)
    print("Wrote treatment_examples.csv and control_examples.csv to", outdir)

if __name__ == "__main__":
    main()
