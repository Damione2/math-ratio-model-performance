# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_3_Qwen2.5-1.5B_math_0.406761470351.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0009780083415028736,
  "intercept": 0.9439306400625891,
  "r_squared": 0.5362848494485501,
  "p_value": 0.010385426100467873,
  "stderr": 0.0003031441438814201,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0009587364668698188,
  "ci_lower": -0.0013844058766508609,
  "ci_upper": 0.0007241906069835508,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.161641541039, math_pct: 16.16%, best_f1_mean: 0.933628, n_seeds: 3, cooks_d: 1.668691e+00


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_3_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.00268377342249601,
  "fixed_effect_math_pct_se": 0.00011078306862432375,
  "pvalue": 1.1990788988296325e-129,
  "converged": true
}


## Files produced


- leave_one_out/loo_3_aggregated_by_model_ratio.csv

- leave_one_out/loo_3_agg_regression_plot.png

- leave_one_out/loo_3_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_3_bootstrap_slope_ci.json

- leave_one_out/loo_3_cooks_distance.csv

- leave_one_out/loo_3_mixedlm_summary.txt

- leave_one_out/loo_3_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
