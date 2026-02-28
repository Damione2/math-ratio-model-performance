# scripts/plot_residuals.py
import pandas as pd, statsmodels.api as sm, matplotlib.pyplot as plt
df = pd.read_csv('experiments/ablation_summary.csv')
X = sm.add_constant(df['math_ratio'])
y = df['best_f1']
res = sm.OLS(y, X).fit(cov_type='HC3')
fitted = res.fittedvalues
resid = res.resid
plt.figure(figsize=(5,3))
plt.scatter(fitted, resid, s=30)
plt.axhline(0, color='k', linestyle='--')
plt.xlabel('Fitted best_f1')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.tight_layout()
plt.savefig('experiments/plots/residuals_vs_fitted.png', dpi=200)
print('Saved experiments/plots/residuals_vs_fitted.png')
