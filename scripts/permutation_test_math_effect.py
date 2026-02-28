# scripts/permutation_test_math_effect.py
import sys
import numpy as np
import pandas as pd
from pathlib import Path

csv_path = Path("experiments/ablation_summary.csv")
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found.")
    sys.exit(1)

df = pd.read_csv(csv_path)

# find math column automatically
math_cols = [c for c in df.columns if "math" in c.lower()]
if len(math_cols) == 0:
    print("No column with 'math' in name found. Available columns:", list(df.columns))
    sys.exit(1)

math_col = math_cols[0]
# find best_f1 column
best_cols = [c for c in df.columns if "best" in c.lower() and "f1" in c.lower()]
if len(best_cols) == 0:
    # try common alternatives
    if "best_f1" in df.columns:
        best_col = "best_f1"
    elif "best" in df.columns:
        best_col = "best"
    else:
        print("No best_f1-like column found. Available columns:", list(df.columns))
        sys.exit(1)
else:
    best_col = best_cols[0]

x = pd.to_numeric(df[math_col], errors="coerce").values
y = pd.to_numeric(df[best_col], errors="coerce").values

mask = ~np.isnan(x) & ~np.isnan(y)
x = x[mask]; y = y[mask]

if len(x) < 3:
    print("Not enough valid rows for permutation test.")
    sys.exit(1)

obs_slope = np.polyfit(x, y, 1)[0]
nperm = 5000
count = 0
rng = np.random.default_rng(0)

for _ in range(nperm):
    y_perm = rng.permutation(y)
    slope = np.polyfit(x, y_perm, 1)[0]
    if abs(slope) >= abs(obs_slope):
        count += 1

p_perm = (count + 1) / (nperm + 1)
out = {
    "math_column": math_col,
    "best_f1_column": best_col,
    "n_rows": int(len(x)),
    "obs_slope": float(obs_slope),
    "nperm": nperm,
    "p_perm": float(p_perm)
}

print("Permutation test result:", out)
Path("experiments").mkdir(parents=True, exist_ok=True)
with open("experiments/permutation_test_result.txt", "w", encoding="utf8") as f:
    for k, v in out.items():
        f.write(f"{k}: {v}\n")
print("Wrote experiments/permutation_test_result.txt")
