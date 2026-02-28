# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\plots\leave_one_out\ablation_summary_loo_drop_2_Qwen2.5-1.5B_math_0.292976588629.csv**

- Output folder: **D:\guardian_project\experiments\plots\leave_one_out**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0015698491591561396,
  "intercept": 0.9610339068887773,
  "r_squared": 0.4965227134272604,
  "p_value": 0.01546944580692618,
  "stderr": 0.0005269350063667211,
  "n": 11
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0015721449314388429,
  "ci_lower": -0.0084421918399705,
  "ci_upper": -0.0005894529179585799,
  "n_boot": 500
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 9.165800e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **leave_one_out/loo_2_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.0027443424526292963,
  "fixed_effect_math_pct_se": 9.719875543159736e-05,
  "pvalue": 2.216450892441468e-175,
  "converged": true
}


## Files produced


- leave_one_out/loo_2_aggregated_by_model_ratio.csv

- leave_one_out/loo_2_agg_regression_plot.png

- leave_one_out/loo_2_regression_bestf1_vs_mathratio.json

- leave_one_out/loo_2_bootstrap_slope_ci.json

- leave_one_out/loo_2_cooks_distance.csv

- leave_one_out/loo_2_mixedlm_summary.txt

- leave_one_out/loo_2_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
