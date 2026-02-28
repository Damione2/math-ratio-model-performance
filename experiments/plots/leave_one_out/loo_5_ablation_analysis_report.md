# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_5_Qwen2.5-Coder-1.5B_math_0.353147877013.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.001511244760456193,
  "intercept": 0.9584465012970358,
  "r_squared": 0.46649617935705534,
  "p_value": 0.020534684171051535,
  "stderr": 0.0005387140158178411,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0015517251008278629,
  "ci_lower": -0.0050722329267815485,
  "ci_upper": 0.0001178181110990453,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 9.621744e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_5_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.0027320773921032286,
  "fixed_effect_math_pct_se": 9.39125133941626e-05,
  "pvalue": 4.567906398928804e-186,
  "converged": true
}


## Files produced


- leave_one_out/loo_5_aggregated_by_model_ratio.csv

- leave_one_out/loo_5_agg_regression_plot.png

- leave_one_out/loo_5_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_5_bootstrap_slope_ci.json

- leave_one_out/loo_5_cooks_distance.csv

- leave_one_out/loo_5_mixedlm_summary.txt

- leave_one_out/loo_5_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
