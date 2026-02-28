# scripts/plot_math_vs_bestf1_labeled.py
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv("experiments/ablation_summary.csv")
plt.figure(figsize=(6,4))
sns.regplot(x='math_ratio', y='best_f1', data=df, ci=95, scatter_kws={'s':40})
for _, r in df.iterrows():
    plt.text(r['math_ratio'], r['best_f1'], str(r.get('model','')[:12]), fontsize=7, alpha=0.7)
plt.xlabel('math_ratio (fraction)')
plt.ylabel('best_f1')
plt.title('best_f1 vs math_ratio with 95% CI and model labels')
plt.tight_layout()
plt.savefig('experiments/plots/math_vs_bestf1_labeled.png', dpi=200)
print('Saved experiments/plots/math_vs_bestf1_labeled.png')
