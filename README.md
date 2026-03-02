# Guardian Project: Effect of Math Example Ratio on Model Performance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18816903.svg)](https://doi.org/10.5281/zenodo.18816903)

This repository contains the code, experiments, and analysis for studying how the **fraction of math examples** (`math_ratio`) in training data affects a model’s **best F1 score** (`best_f1`).  
We run **multi-seed ablations**, perform **robust statistical analysis** (permutation tests, bootstrap, WLS, mixed-effects models), and provide all scripts needed to reproduce the figures and tables.

---

## Paper (preprint)

The preprint PDF and LaTeX source are included in the repository:

- **paper/paper.pdf** — downloadable preprint (v1.0.0)  
- **paper/latex_src/** — LaTeX source files used to generate the PDF (for arXiv or local compilation)

Direct links: `paper/paper.pdf` and `paper/latex_src/`

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

## Repository structure (relevant parts)

math-ratio-model-performance/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.txt
├── paper/
│   ├── paper.pdf
│   └── latex_src/
│       ├── main.tex
│       ├── refs.bib
│       └── figs/ (figure files used by the LaTeX source)
├── experiments/
│   └── plots/ (figures used in the manuscript)
├── scripts/
└── docs/

---

## How to reproduce (quick)

1. Install requirements: `pip install -r requirements.txt`  
2. Generate the main scatter plot: `python scripts/plot_math_vs_bestf1_labeled.py`  
3. Run bootstrap analysis: `python scripts/bootstrap_slope_direct.py`  
4. Reproduce mixed-effects analysis: `python scripts/wls_regression.py` and `python scripts/mixedlm_analysis.py`

For full reproduction, see `docs/REPRODUCIBILITY.md` (detailed environment, exact commands, and expected outputs).

---

## Paper build (local)

To compile the LaTeX source locally:

1. Ensure TeX Live or MikTeX is installed.  
2. From repository root:
```bash
cd paper/latex_src
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
