# scripts/bootstrap_slope_direct.py
import pandas as pd, numpy as np
from pathlib import Path
df = pd.read_csv("experiments/ablation_summary.csv")
x = pd.to_numeric(df['math_ratio'], errors='coerce').values
y = pd.to_numeric(df['best_f1'], errors='coerce').values
mask = ~np.isnan(x) & ~np.isnan(y)
x, y = x[mask], y[mask]
rng = np.random.default_rng(0)
nboot = 5000
slopes = np.empty(nboot)
n = len(x)
for i in range(nboot):
    idx = rng.integers(0, n, n)
    slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
np.savetxt("experiments/bootstrap_slopes_direct.txt", slopes, fmt="%.18e")
with open("experiments/bootstrap_slopes_direct_summary.txt","w") as f:
    f.write(f"n={n}\nmedian={np.median(slopes)}\nci_2.5={np.percentile(slopes,2.5)}\nci_97.5={np.percentile(slopes,97.5)}\n")
print("Wrote experiments/bootstrap_slopes_direct.txt and summary")
