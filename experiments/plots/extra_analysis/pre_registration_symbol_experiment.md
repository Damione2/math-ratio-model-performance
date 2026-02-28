# Pre‑registration for Symbol Manipulation Experiment

## Title
Does symbol content causally reduce validation F1

## Hypothesis
Introducing or increasing the presence of a defined set of mathcode symbols (factor P) in training examples causes a decrease in best validation F1 compared with matched control runs.

## Primary outcome
Best validation F1 per training run.

## Experimental arms
- Control training data unchanged.
- Treatment identical training data except a pre‑specified symbol manipulation applied to a matched subset (see Manipulation).

## Manipulation (explicit)
- Symbol set S digits `0 1 2 3 4 5 6 7 8 9`, operators `+ -   =`, parentheses `() [] {}`, colon ``, backtick `` ` ``, backslash ``, LaTeX markers `(` `)` `frac`.
- Treatment operation (choose one)  
  - Append augmentation append the neutral token sequence ` [ ( )  ]` to selected examples.  
  - Mask removal replace every token in S with the placeholder token `SYM` in selected examples.
- The manipulation must be applied deterministically and identically across runs.

## Data selection and matching
- Select a pool of training examples matched on label, length, and source.
- Randomly split the pool into two matched halves A and B.
- Apply the manipulation only to half A for the Treatment arm; half B remains unchanged for Control.
- All other training data remain identical across arms.

## Randomization and seeds
- Randomly assign seeds to arms.
- Planned sample size see power calculation script; initial recommendation 10 seeds per arm (20 runs total). If resources allow, use 20 seeds per arm.

## Analysis plan
- Primary test linear mixed model on best F1 with fixed effect `treatment` and random intercept for `seed` or `model` if multiple models are tested.
- Model formula example `best_f1 ~ treatment + (1seed)` or `best_f1 ~ treatment + (1model)`.
- Report estimate for `treatment`, 95% CI, and p‑value.
- Secondary analyses stratified F1 on high‑symbol vs low‑symbol validation examples; learning curves; calibration metrics.
- Robustness checks bootstrap CI, leave‑one‑seed‑out, Cook’s D on aggregated analyses.

## Exclusion rules
- Predefine criteria for failed runs (OOM, early termination, corrupted logs). Exclude only if failure is unrelated to treatment and report exclusions.

## Significance threshold
- Two‑sided α = 0.05.

## Pre‑registration metadata
- Date YYYY‑MM‑DD
- Responsible researcher Damyan
- Repository link (add repo URL)
