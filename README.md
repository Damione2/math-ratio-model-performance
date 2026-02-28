# Guardian Project: Effect of Math Example Ratio on Model Performance

This repository contains the code, experiments, and analysis for studying how the **fraction of math examples** (`math_ratio`) in training data affects a model’s **best F1 score** (`best_f1`).  
We run **multi-seed ablations**, perform **robust statistical analysis** (permutation tests, bootstrap, WLS, mixed-effects models), and provide all scripts needed to reproduce the figures and tables.

---

## Key result (short summary)

Across **52 runs**, we find a **robust negative association** between `math_ratio` and `best_f1`:

- **WLS (HC3) slope:** ≈ **−0.236** per unit `math_ratio`  
  - ≈ **−0.00236** per percentage point  
  - 95% CI ≈ **[−0.271, −0.202]**
- **Bootstrap (5,000 resamples):** median slope ≈ **−0.2367**  
  - 95% CI ≈ **[−0.2639, −0.1963]**
- **Permutation test (5,000 permutations):**  
  - observed slope = **−0.2363**  
  - \(p_{\text{perm}} \approx 0.0002\)
- **Mixed-effects model (random intercept):** slope ≈ **−0.273**

Influence diagnostics (Cook’s D and leave-one-out) show that the effect is **not driven by a single high-influence group**.

---

## Repository structure

```text
guardian_project/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.txt
│
├── paper/                  # LaTeX source for arXiv
│   ├── main.tex
│   ├── refs.bib
│   └── figs/
│       ├── figure1_scatter.png
│       ├── figure2_bootstrap.png
│       ├── figure3_residuals.png
│       ├── cooks_distance.csv
│       └── leave_one_out_summary.csv
│
├── experiments/
│   ├── ablation_summary.csv
│   ├── ablation_summary_filtered_excl_high_influence.csv
│   ├── bootstrap_slopes_direct.txt
│   ├── bootstrap_slopes_direct_summary.txt
│   ├── permutation_test_result.txt
│   ├── wls_regression_result.txt
│   ├── mixedlm_random_intercept_summary.txt
│   └── plots/
│       ├── math_vs_bestf1_labeled.png
│       ├── bootstrap_slope_distribution.png
│       ├── residuals_vs_fitted.png
│       ├── cooks_distance.csv
│       └── leave_one_out_summary.csv
│
├── scripts/
│   ├── bootstrap_slope_direct.py
│   ├── permutation_test_math_effect.py
│   ├── wls_regression.py
│   ├── plot_math_vs_bestf1_labeled.py
│   ├── plot_bootstrap_ci.py
│   ├── plot_residuals.py
│   └── reproduce_all.sh   # optional convenience script
│
├── runs/                   # training runs (not all tracked in git)
├── runs_archive/           # archived runs (not tracked in git)
└── final_report/           # collected figures/tables for the paper
