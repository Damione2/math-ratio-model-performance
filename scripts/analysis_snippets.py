# scripts/analysis_snippets.py
import pandas as pd
import numpy as np
from scipy.stats import linregress, spearmanr
from math import sqrt

df = pd.read_csv("experiments/ablation_summary.csv")
df = df.dropna(subset=["math_ratio","best_f1"])
df["math_pct"] = (df["math_ratio"]*100).astype(float)

# Linear regression
res = linregress(df["math_pct"], df["best_f1"])
print("Linear regression best_f1 ~ math_pct")
print(f"slope={res.slope:.6f}, intercept={res.intercept:.6f}, r2={res.rvalue**2:.4f}, p={res.pvalue:.3e}")

# Spearman correlation
rho, p_s = spearmanr(df["math_pct"], df["best_f1"])
print(f"Spearman rho={rho:.3f}, p={p_s:.3e}")

# Cohen's d between math_100 and math_0 (if both present)
g100 = df[df["math_pct"]==100]["best_f1"].values
g0 = df[df["math_pct"]==0]["best_f1"].values
if len(g100)>0 and len(g0)>0:
    m1, m2 = g100.mean(), g0.mean()
    s1, s2 = g100.std(ddof=1), g0.std(ddof=1)
    n1, n2 = len(g100), len(g0)
    pooled = sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    cohens_d = (m1 - m2) / pooled
    print(f"Cohen's d (100 vs 0) = {cohens_d:.3f}")
else:
    print("Not enough points for Cohen's d (need both math_100 and math_0).")

# Simple bootstrap CI for slope
def bootstrap_slope(x,y, nboot=5000):
    rng = np.random.default_rng(0)
    slopes = []
    n = len(x)
    for _ in range(nboot):
        idx = rng.integers(0,n,n)
        xi, yi = x[idx], y[idx]
        slopes.append(linregress(xi, yi).slope)
    return np.percentile(slopes, [2.5,97.5])
ci = bootstrap_slope(df["math_pct"].values, df["best_f1"].values)
print("Bootstrap 95% CI for slope:", ci)
