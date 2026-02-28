Analysis report
================
Generated: 2026-02-20T15:18:55.296196 UTC

Files in this report:
 - bootstrap_diffs.npy
 - comparison_summary.csv
 - excerpts\Control_run_1771585724_1_excerpt.txt
 - excerpts\error_excerpts_index.csv
 - excerpts\Treatment_run_1771585724_0_excerpt.txt
 - holdout_note.txt
 - merged_runs.csv
 - README.txt
 - summary_stats.csv

Reproducibility
----------------
To reproduce this analysis run:

python experiments/plots/extra_analysis/analyze_experiment_results.py \
  --runs-dir runs \
  --assignments experiments\plots\extra_analysis\seed_assignments.csv \
  --outdir analysis \
  --n-boot 10000 --n-perm 10000 --seed 42

Notes:
 - This script extracts best_f1 from metrics.json or stdout/stderr.
 - If some runs are missing best_f1 they are excluded from statistical tests.
 - For holdout evaluation, provide --holdout and --artifacts-dir if available.
