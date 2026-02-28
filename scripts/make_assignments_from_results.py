# save as scripts/make_assignments_from_results.py and run in venv
import csv, pathlib, json
res = pathlib.Path("experiments/plots/extra_analysis/experiment_results.csv")
out = pathlib.Path("experiments/plots/extra_analysis/seed_assignments.csv")
rows = []
with res.open() as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        # r: run_id, arm, seed, run_dir, return_code, timestamp
        rows.append({"run_id": r["run_id"], "arm": r["arm"], "seed": r["seed"], "subset": "", "outdir": r["run_dir"], "best_f1": ""})
# write header matching analyzer expectations
with out.open("w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=["run_id","arm","seed","subset","outdir","best_f1"])
    writer.writeheader()
    writer.writerows(rows)
print("Wrote", out)
