# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_1_Qwen2.5-1.5B_math_0.217507645260.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0017180051379977086,
  "intercept": 0.9659571547091511,
  "r_squared": 0.47730955275058584,
  "p_value": 0.018573002640664107,
  "stderr": 0.0005992739807574185,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.001752320312548043,
  "ci_lower": -0.0084421918399705,
  "ci_upper": 0.0006150303876927836,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.161641541039, math_pct: 16.16%, best_f1_mean: 0.933628, n_seeds: 3, cooks_d: 2.794404e+00

- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 8.997922e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_1_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.0027412010350538474,
  "fixed_effect_math_pct_se": 9.68615941422399e-05,
  "pvalue": 3.437814725060445e-176,
  "converged": true
}


## Files produced


- leave_one_out/loo_1_aggregated_by_model_ratio.csv

- leave_one_out/loo_1_agg_regression_plot.png

- leave_one_out/loo_1_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_1_bootstrap_slope_ci.json

- leave_one_out/loo_1_cooks_distance.csv

- leave_one_out/loo_1_mixedlm_summary.txt

- leave_one_out/loo_1_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
