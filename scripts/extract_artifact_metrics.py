#scripts/extract_artifact_metrics.py
import json, pathlib
p = pathlib.Path(r"C:\guardian_artifacts") / "training_summary.json"
if p.exists():
    j = json.loads(p.read_text())
    print(json.dumps(j, indent=2))
else:
    print("No training_summary.json found at", p)
