# Ablation Analysis Report


## Inputs and locations


- Seed-level summary CSV: **D:\guardian_project\experiments\ablation_summary_filtered_excl_high_influence.csv**

- Output folder: **D:\guardian_project\experiments\plots\sensitivity_excluding_high_influence**


## Aggregated regression (groups with min_seeds >= 2)


Regression results (OLS on aggregated group means):


{
  "slope": -0.0015147058185276674,
  "intercept": 0.9585029894136574,
  "r_squared": 0.47134448471027446,
  "p_value": 0.013667902686227388,
  "stderr": 0.0005072773396279183,
  "n": 12
}


Bootstrap 95% CI for aggregated slope (percent scale):


{
  "median": -0.0015155995246111803,
  "ci_lower": -0.0075537952561878985,
  "ci_upper": 0.0002456122938551395,
  "n_boot": 5000
}


High-influence aggregated groups (Cook's D > 4/n):


- model: Qwen2.5-1.5B, math_ratio: 0.406761470351, math_pct: 40.68%, best_f1_mean: 0.868984, n_seeds: 3, cooks_d: 9.386261e-01


## Mixed-effects model (seed-level)


MixedLM summary saved to: **sensitivity_excluding_high_influence/sensitivity_mixedlm_summary.txt**


MixedLM result (JSON):


{
  "fixed_effect_math_pct_coef": -0.002729826925630949,
  "fixed_effect_math_pct_se": 9.498322137076124e-05,
  "pvalue": 1.2044747801927582e-181,
  "converged": true
}


## Files produced


- sensitivity_excluding_high_influence/sensitivity_aggregated_by_model_ratio.csv

- sensitivity_excluding_high_influence/sensitivity_agg_regression_plot.png

- sensitivity_excluding_high_influence/sensitivity_regression_bestf1_vs_mathratio.json

- sensitivity_excluding_high_influence/sensitivity_bootstrap_slope_ci.json

- sensitivity_excluding_high_influence/sensitivity_cooks_distance.csv

- sensitivity_excluding_high_influence/sensitivity_mixedlm_summary.txt

- sensitivity_excluding_high_influence/sensitivity_mixedlm_result.json


## Notes and interpretation guidance


- Aggregated regression uses group means; groups with n_seeds < min_seeds were excluded from the regression but remain in the seed-level CSV.

- MixedLM estimates the fixed effect of math_pct while accounting for model-level random intercepts; check mixedlm_summary.txt for convergence and diagnostics.

- Bootstrap CI gives a nonparametric interval for the aggregated slope; if the CI excludes zero, the aggregated slope is robust to sampling variability across groups.

- High Cook's D points should be inspected: they can disproportionately affect the aggregated regression slope.
