Analysis report
================
Generated: 2026-02-21T10:43:48.945234 UTC

Files in this report:
 - bootstrap_diffs.npy
 - comparison_summary.csv
 - excerpts\Control_run_1771601706_0_excerpt.txt
 - excerpts\Control_run_1771601706_10_excerpt.txt
 - excerpts\Control_run_1771601706_11_excerpt.txt
 - excerpts\Control_run_1771601706_14_excerpt.txt
 - excerpts\Control_run_1771601706_18_excerpt.txt
 - excerpts\Control_run_1771601706_19_excerpt.txt
 - excerpts\Control_run_1771601706_3_excerpt.txt
 - excerpts\Control_run_1771601706_5_excerpt.txt
 - excerpts\Control_run_1771601706_6_excerpt.txt
 - excerpts\Control_run_1771601706_9_excerpt.txt
 - excerpts\error_excerpts_index.csv
 - excerpts\Treatment_run_1771601706_12_excerpt.txt
 - excerpts\Treatment_run_1771601706_13_excerpt.txt
 - excerpts\Treatment_run_1771601706_15_excerpt.txt
 - excerpts\Treatment_run_1771601706_16_excerpt.txt
 - excerpts\Treatment_run_1771601706_17_excerpt.txt
 - excerpts\Treatment_run_1771601706_1_excerpt.txt
 - excerpts\Treatment_run_1771601706_2_excerpt.txt
 - excerpts\Treatment_run_1771601706_4_excerpt.txt
 - excerpts\Treatment_run_1771601706_7_excerpt.txt
 - excerpts\Treatment_run_1771601706_8_excerpt.txt
 - excerpts\Unknown_run_1771601706_0_excerpt.txt
 - excerpts\Unknown_run_1771601706_11_excerpt.txt
 - excerpts\Unknown_run_1771601706_13_excerpt.txt
 - excerpts\Unknown_run_1771601706_16_excerpt.txt
 - excerpts\Unknown_run_1771601706_19_excerpt.txt
 - excerpts\Unknown_run_1771601706_1_excerpt.txt
 - excerpts\Unknown_run_1771601706_5_excerpt.txt
 - excerpts\Unknown_run_1771601706_6_excerpt.txt
 - excerpts\Unknown_run_1771601706_7_excerpt.txt
 - excerpts\Unknown_run_1771601706_8_excerpt.txt
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
