# scripts/mixedlm_random_intercept.py
import pandas as pd, statsmodels.formula.api as smf
df = pd.read_csv("experiments/ablation_summary.csv")
# ensure model column exists
if 'model' not in df.columns:
    df['model'] = df.get('model_name', df.index.astype(str))
m = smf.mixedlm("best_f1 ~ math_ratio", df, groups=df["model"], re_formula="1")
res = m.fit(reml=False, method='lbfgs')
print(res.summary())
with open("experiments/mixedlm_random_intercept_summary.txt","w") as f:
    f.write(res.summary().as_text())
print("Wrote experiments/mixedlm_random_intercept_summary.txt")
