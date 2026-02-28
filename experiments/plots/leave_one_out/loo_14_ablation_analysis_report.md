# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_14_Qwen2.5-Math-1.5B_math_0.356431292118.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.002297470922283153,
  "intercept": 0.9835325071872429,
  "r_squared": 0.8255342183300342,
  "p_value": 6.900005493514805e-06,
  "stderr": 0.00030489267528129813,
  "n": 14
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.002286670454675963,
  "ci_lower": -0.0027399423331307252,
  "ci_upper": -0.0011782992021144255,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.122098214286, math_pct: 12.21%, best_f1_mean: 0.928230, n_seeds: 3, cooks_d: 3.332876e-01

- model: Qwen2.5-Coder-1.5B, math_ratio: 0.000000000000, math_pct: 0.00%, best_f1_mean: 0.998038, n_seeds: 4, cooks_d: 3.694482e-01

- model: Qwen2.5-Math-1.5B, math_ratio: 0.000000000000, math_pct: 0.00%, best_f1_mean: 0.998038, n_seeds: 4, cooks_d: 3.694482e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_14_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.002527156456984935,
  "fixed_effect_math_pct_se": 6.499148775255185e-05,
  "pvalue": 0.0,
  "converged": true
}


## Files produced


- leave_one_out/loo_14_aggregated_by_model_ratio.csv

- leave_one_out/loo_14_agg_regression_plot.png

- leave_one_out/loo_14_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_14_bootstrap_slope_ci.json

- leave_one_out/loo_14_cooks_distance.csv

- leave_one_out/loo_14_mixedlm_summary.txt

- leave_one_out/loo_14_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
