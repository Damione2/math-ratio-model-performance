# Ablation One‑Page Summary

## Purpose
Summarize the main finding, robustness checks, and three key plots for a single slide or one‑page figure panel.

## Headline
**Higher observed math fraction predicts lower validation F1 across models.**  
Primary, defensible estimate (seed‑level mixed model): **math_pct fixed effect = −0.00253** (SE = 0.000062). Interpret as **≈ −0.025 F1 per 10%** increase in observed math content.

## Key numbers
| **Estimate** | **Value** | **Notes / file** |
|---|---:|---|
| **MixedLM (seed‑level)** | **−0.0025267** | SE = 6.21e‑05; `experiments/plots/mixedlm_result.json` |
| **Aggregated OLS (base)** | **−0.0022713** | SE = 0.0002910; R² = 0.8241; `experiments/plots/regression_bestf1_vs_mathratio.json` |
| **WLS (weights = n_seeds)** | **−0.0023036** | SE = 0.0002731; R² = 0.8455; `experiments/plots/extra_analysis/wls_regression.json` |
| **Sensitivity OLS (exclude flagged)** | **−0.0015147** | SE = 0.0005073; bootstrap CI includes 0; `experiments/plots/sensitivity_excluding_high_influence/` |
| **Leave‑one‑out range** | approx **−0.00245 → −0.00204** | `experiments/plots/leave_one_out/leave_one_out_summary.csv` |

## Diagnostics and robustness
- **Influential groups (Cook’s D):** `Qwen2.5-Coder-1.5B @ 0%`, `Qwen2.5-Math-1.5B @ 0%`, `Qwen2.5-1.5B @ 12.21%`. See `experiments/plots/cooks_distance.csv`.  
- **Interaction test:** MixedLM with interaction did not converge; OLS fallback shows **different intercepts by model** but **no significant math_pct × model interactions**. Files: `experiments/plots/extra_analysis/mixedlm_interaction_summary.txt` and `.../mixedlm_interaction_result.json`.  
- **Takeaway:** Direction is robust across methods; aggregated means are partly influenced by a few high‑leverage groups, but seed‑level MixedLM and WLS agree on a negative effect.

## Slide layout (one slide)
**Left column (text)**  
- Headline (bold) and one‑line interpretation.  
- Bullet list: primary estimate (MixedLM), WLS confirmation, sensitivity note, Cook’s D flagged groups.

**Right column (three stacked plots, top→bottom)**  
1. Aggregated OLS plot — `D:\guardian_project\experiments\plots\agg_regression_plot.png`  
2. WLS plot — `D:\guardian_project\experiments\plots\extra_analysis\wls_regression_plot.png`  
3. Slopes comparison — `D:\guardian_project\experiments\plots\extra_analysis\slopes_comparison.png`

## Files to attach with slide
- `D:\guardian_project\experiments\plots\agg_regression_plot.png`  
- `D:\guardian_project\experiments\plots\extra_analysis\wls_regression_plot.png`  
- `D:\guardian_project\experiments\plots\extra_analysis\slopes_comparison.png`  
- `D:\guardian_project\experiments\plots\mixedlm_summary.txt`  
- `D:\guardian_project\experiments\plots\cooks_distance.csv`  
- `D:\guardian_project\experiments\plots\leave_one_out\leave_one_out_summary.csv`

## Short caption text for the slide
**Primary:** MixedLM fixed effect for math_pct = **−0.00253 ± 0.000062** (seed‑level, random intercepts).  
**Robustness:** WLS ≈ −0.00230; aggregated bootstrap CI excludes 0 in base analysis but includes 0 after removing flagged groups. Report both results and diagnostics.
