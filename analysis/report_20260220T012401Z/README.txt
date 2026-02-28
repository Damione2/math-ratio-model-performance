Analysis report
================
Generated: 2026-02-20T01:24:03.082619 UTC

Files in this report:
 - bootstrap_diffs.npy
 - comparison_summary.csv
 - excerpts\Control_run_1771521950_0_excerpt.txt
 - excerpts\Control_run_1771521950_11_excerpt.txt
 - excerpts\Control_run_1771521950_13_excerpt.txt
 - excerpts\Control_run_1771521950_14_excerpt.txt
 - excerpts\Control_run_1771521950_17_excerpt.txt
 - excerpts\Control_run_1771521950_2_excerpt.txt
 - excerpts\Control_run_1771521950_4_excerpt.txt
 - excerpts\Control_run_1771521950_6_excerpt.txt
 - excerpts\Control_run_1771521950_7_excerpt.txt
 - excerpts\Control_run_1771521950_8_excerpt.txt
 - excerpts\error_excerpts_index.csv
 - excerpts\Treatment_run_1771521950_10_excerpt.txt
 - excerpts\Treatment_run_1771521950_12_excerpt.txt
 - excerpts\Treatment_run_1771521950_15_excerpt.txt
 - excerpts\Treatment_run_1771521950_16_excerpt.txt
 - excerpts\Treatment_run_1771521950_18_excerpt.txt
 - excerpts\Treatment_run_1771521950_19_excerpt.txt
 - excerpts\Treatment_run_1771521950_1_excerpt.txt
 - excerpts\Treatment_run_1771521950_3_excerpt.txt
 - excerpts\Treatment_run_1771521950_5_excerpt.txt
 - excerpts\Treatment_run_1771521950_9_excerpt.txt
 - holdout_note.txt
 - merged_runs.csv
 - README.txt
 - robustness_sensitivity.txt
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
