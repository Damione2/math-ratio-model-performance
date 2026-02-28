# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_13_Qwen2.5-Math-1.5B_math_0.355581807442.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0022901429258598205,
  "intercept": 0.9834760952937099,
  "r_squared": 0.8233856322920889,
  "p_value": 7.433917683270725e-06,
  "stderr": 0.0003061845956669016,
  "n": 14
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0022797181602576527,
  "ci_lower": -0.0027399423331307252,
  "ci_upper": -0.0011675729380033017,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.122098214286, math_pct: 12.21%, best_f1_mean: 0.928230, n_seeds: 3, cooks_d: 3.311279e-01

- model: Qwen2.5-Coder-1.5B, math_ratio: 0.000000000000, math_pct: 0.00%, best_f1_mean: 0.998038, n_seeds: 4, cooks_d: 3.688549e-01

- model: Qwen2.5-Math-1.5B, math_ratio: 0.000000000000, math_pct: 0.00%, best_f1_mean: 0.998038, n_seeds: 4, cooks_d: 3.688549e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_13_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.002521831592092923,
  "fixed_effect_math_pct_se": 6.478494892874942e-05,
  "pvalue": 0.0,
  "converged": true
}


## Files produced


- leave_one_out/loo_13_aggregated_by_model_ratio.csv

- leave_one_out/loo_13_agg_regression_plot.png

- leave_one_out/loo_13_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_13_bootstrap_slope_ci.json

- leave_one_out/loo_13_cooks_distance.csv

- leave_one_out/loo_13_mixedlm_summary.txt

- leave_one_out/loo_13_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
