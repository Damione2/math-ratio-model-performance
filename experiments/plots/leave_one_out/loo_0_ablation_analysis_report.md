# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_0_Qwen2.5-1.5B_math_0.161641541039.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.001544763681154665,
  "intercept": 0.9595637014781356,
  "r_squared": 0.2905179161471694,
  "p_value": 0.08709609913616512,
  "stderr": 0.0008046834176680024,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0015900687965732935,
  "ci_lower": -0.0084421918399705,
  "ci_upper": 0.0006150303876927836,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.217507645260, math_pct: 21.75%, best_f1_mean: 0.918699, n_seeds: 3, cooks_d: 2.011022e+00

- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 1.341020e+00


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_0_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.0027678393210720928,
  "fixed_effect_math_pct_se": 9.926742649372721e-05,
  "pvalue": 4.3308829936801915e-171,
  "converged": true
}


## Files produced


- leave_one_out/loo_0_aggregated_by_model_ratio.csv

- leave_one_out/loo_0_agg_regression_plot.png

- leave_one_out/loo_0_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_0_bootstrap_slope_ci.json

- leave_one_out/loo_0_cooks_distance.csv

- leave_one_out/loo_0_mixedlm_summary.txt

- leave_one_out/loo_0_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
