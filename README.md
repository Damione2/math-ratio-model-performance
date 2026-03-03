# Guardian Project: Effect of Math Example Ratio on Model Performance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18816903.svg)](https://doi.org/10.5281/zenodo.18816903)

**Short summary:** this repository contains code, experiments, and analysis that study how the **fraction of math examples** (`math_ratio`) in training data affects a model’s **best F1** (`best_f1`) and its tendency toward confident, ungrounded outputs. All figures, scripts, and reproducibility artifacts are included.

---

## Key result

Across **52 runs**, we observe a robust negative association between `math_ratio` and `best_f1`:

- **WLS (HC3) slope:** ≈ **−0.236** per unit `math_ratio`  
  - ≈ **−0.00236** per percentage point  
  - 95% CI ≈ **[−0.271, −0.202]**  
- **Bootstrap (5,000 resamples):** median slope ≈ **−0.2367**; 95% CI ≈ **[−0.2639, −0.1963]**  
- **Permutation test (5,000 permutations):** observed slope = **−0.2363**, \(p_{\text{perm}} \approx 0.0002\)  
- **Mixed‑effects model (random intercept):** slope ≈ **−0.273**

Influence diagnostics (Cook’s D, leave‑one‑out) indicate the effect is not driven by a single high‑influence run.

---

## Definitions

- **CfC‑LNN** — **Closed‑form Continuous‑time network** (the liquid time‑constant architecture used in the experts). This is an **architecture**, not a behavioral metric.  
- **Response consistency** — observed consistency of model outputs across internal trajectories (sometimes called chain‑of‑thought consistency in other work). This is a **measured behavior**, distinct from CfC‑LNN.  
- **Calibration / ECE** — expected calibration error; measures how predicted confidence aligns with actual accuracy.  
- **Swagger** — mean predicted confidence on incorrect answers; a simple indicator of confident hallucinations.  
- **math_ratio** — fraction of training examples that are mathematical in nature (expressed as a proportion between 0 and 1).

---

## Philosophical framing

**Math is infinite; the world is bounded.** Over‑feeding models with mathematical examples sculpts a narrow, high‑confidence attractor: models become excellent at equations but may apply that lens where it does not belong, producing confident, ungrounded outputs on general prompts. The problem is not math per se, but **overrepresentation without grounding**.

---

## Repository structure

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
│       └── figs/
├── experiments/
│   ├── plots/
│   │   ├── math_vs_bestf1_labeled.png
│   │   └── other_figs.png
│   └── diagnostics/
│       └── per_seed_diagnostics.csv
├── scripts/
│   ├── plot_math_vs_bestf1_labeled.py
│   ├── bootstrap_slope_direct.py
│   ├── wls_regression.py
│   ├── mixedlm_analysis.py
│   └── compute_diagnostics.py
├── examples/
│   └── sample_predictions.csv
├── docs/
│   └── REPRODUCIBILITY.md
└── paper_assets/
└── high_res_figs/

---

## Quick reproduction (one‑click)

1. Install dependencies:
```bash
pip install -r requirements.txt
Generate the main scatter plot (one command):

bash
python scripts/plot_math_vs_bestf1_labeled.py --predictions examples/sample_predictions.csv
Run bootstrap analysis:

bash
python scripts/bootstrap_slope_direct.py --predictions examples/sample_predictions.csv
Reproduce regression and mixed‑effects analyses:

bash
python scripts/wls_regression.py --predictions examples/sample_predictions.csv
python scripts/mixedlm_analysis.py --predictions examples/sample_predictions.csv
Full environment and exact commands are in docs/REPRODUCIBILITY.md.

## How to interpret the numbers
The negative slope indicates that increasing math_ratio produces a small but consistent decrease in best_f1. The effect is statistically robust across multiple analyses.

Concurrent diagnostics show increased response consistency (internal temporal coherence produced by CfC‑LNN experts) and worse calibration (higher ECE) on out‑of‑domain prompts. In practice this looks like more confident hallucinations: the model is more consistent internally but less grounded externally.

## Practical recommendations
Avoid naive percentage mixing. Prefer context‑aware mixing: increase math examples only when the input is math‑like.

Add contrastive grounding tasks and negative examples that force the model to distinguish mathematical structure from general descriptive or physical language.

Use a dual detection strategy at inference: response consistency + calibration metrics (ECE and swagger). Flag outputs that are both highly consistent and poorly calibrated.

Apply adaptive temperature or panic gating (vibration/entropy signals) to suppress overconfident outputs in ambiguous contexts.

Include human verification for high‑risk, high‑confidence outputs until automated detectors are validated.

## Files to inspect first
experiments/plots/math_vs_bestf1_labeled.png — main figure used in the manuscript.

scripts/compute_diagnostics.py — how routing probabilities, entropy, and liquid state norms are extracted.

docs/REPRODUCIBILITY.md — exact package versions, expected outputs, and verification steps.

examples/sample_predictions.csv — small dataset for one‑click plotting.

## Communication guidance
Use CfC‑LNN when referring to the architecture and response consistency when referring to observed behavior.

Keep the philosophical hook (“Math is infinite; the world is bounded”) at the top of public posts, then immediately provide the one‑click reproducibility command and the PNG link so readers can verify quickly.

## Citation
If you use this work, please cite the Zenodo DOI badge above and the preprint in paper/paper.pdf. -show this in markdown format