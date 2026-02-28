# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_7_Qwen2.5-Coder-1.5B_math_0.356431292118.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0015457463031097254,
  "intercept": 0.95906437933615,
  "r_squared": 0.4815646463107488,
  "p_value": 0.01784487446351352,
  "stderr": 0.0005346098652701584,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0015885559014443724,
  "ci_lower": -0.005032415452325813,
  "ci_upper": 6.96779689219941e-05,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 9.363241e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_7_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.002728487173957784,
  "fixed_effect_math_pct_se": 9.91929529715111e-05,
  "pvalue": 1.453199938989467e-166,
  "converged": true
}


## Files produced


- leave_one_out/loo_7_aggregated_by_model_ratio.csv

- leave_one_out/loo_7_agg_regression_plot.png

- leave_one_out/loo_7_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_7_bootstrap_slope_ci.json

- leave_one_out/loo_7_cooks_distance.csv

- leave_one_out/loo_7_mixedlm_summary.txt

- leave_one_out/loo_7_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
