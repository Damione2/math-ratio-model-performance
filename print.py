import pandas as pd
s = pd.read_csv("experiments/ablation_summary.csv")
g = s.groupby(["model","math_ratio"]).size().reset_index(name="n_seeds")
print(g[g["n_seeds"]==1])
