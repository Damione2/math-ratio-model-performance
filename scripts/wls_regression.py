# scripts/wls_regression.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path

df = pd.read_csv("experiments/ablation_summary.csv")
# auto-detect columns
math_col = [c for c in df.columns if "math" in c.lower()][0]
best_col = [c for c in df.columns if "best" in c.lower() and "f1" in c.lower()][0]

X = pd.to_numeric(df[math_col], errors="coerce")
y = pd.to_numeric(df[best_col], errors="coerce")
mask = ~X.isna() & ~y.isna()
X = X[mask]; y = y[mask]

# add intercept
Xmat = sm.add_constant(X)
# optional weights: inverse variance by group if you have group-level sd; otherwise OLS
model = sm.WLS(y, Xmat)  # or sm.OLS(y, Xmat).fit(cov_type='HC3') for robust SEs
res = model.fit(cov_type='HC3')
print(res.summary())
out = {
    "math_col": math_col,
    "best_col": best_col,
    "n": int(len(y)),
    "slope": float(res.params[1]),
    "slope_se": float(res.bse[1]),
    "pvalue": float(res.pvalues[1])
}
Path("experiments").mkdir(parents=True, exist_ok=True)
with open("experiments/wls_regression_result.txt","w") as f:
    f.write(str(out))
print("Wrote experiments/wls_regression_result.txt")
