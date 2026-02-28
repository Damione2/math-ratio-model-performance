#!/usr/bin/env python3
"""
scripts/auto_run_and_verify.py

Run orchestrator, restore archived snapshots (move back instead of copying), verify metrics,
and run experiment_postprocess.py.

Usage (from project root):
  python scripts/auto_run_and_verify.py \
    --orchestrator-cmd "python -m experiments.plots.extra_analysis.orchestrate_experiment_pipeline --n-per-arm 1 --batch-size 16 --num-workers 0" \
    --runs-dir runs \
    --snapshot-root "C:\guardian_runs" \
    --archive-root "C:\guardian_runs_archive" \
    --artifacts-dir "C:\guardian_artifacts" \
    --postprocess-args "--runs-dir runs --snapshot-root C:\guardian_runs --artifacts-dir C:\guardian_artifacts --assignments experiments/plots/extra_analysis/seed_assignments.csv"

Notes
- This version MOVES archived snapshots from archive_root -> snapshot_root (restores by moving),
  which avoids duplicating large files on the C: drive.
- Use with care: moving removes the archived copy. If you prefer a copy, revert to copytree.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import List, Optional

# Try to import helper from experiment_postprocess to reuse ensure_metrics_for_run if available
# If import fails, we'll fallback to a local minimal implementation.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.experiment_postprocess import ensure_metrics_for_run  # type: ignore
    HAVE_POSTPROCESS_IMPORT = True
except Exception:
    HAVE_POSTPROCESS_IMPORT = False

def restore_snapshot_from_archive(snapshot_root: Path, archive_root: Path, snapshot_name: str) -> Optional[Path]:
    """
    If snapshot_root/<snapshot_name>/artifacts_snapshot exists return it.
    Otherwise, if archive_root/<snapshot_name> exists, MOVE it back to snapshot_root and return the restored artifacts_snapshot path.
    Returns Path to artifacts_snapshot or None if not found/restored.
    """
    live = snapshot_root / snapshot_name / "artifacts_snapshot"
    if live.exists():
        return live
    archived = archive_root / snapshot_name
    if not archived.exists():
        return None
    dst_parent = snapshot_root / snapshot_name
    dst_parent.parent.mkdir(parents=True, exist_ok=True)
    try:
        # If destination exists, remove it first to avoid partial state
        if dst_parent.exists():
            try:
                shutil.rmtree(dst_parent)
            except Exception:
                pass
        # Move the archived snapshot folder back to live snapshots (this removes it from archive)
        shutil.move(str(archived), str(dst_parent))
        restored = dst_parent / "artifacts_snapshot"
        if restored.exists():
            return restored
        # If the archive layout was different (no artifacts_snapshot subfolder), accept dst_parent
        if dst_parent.exists():
            return dst_parent
        return None
    except Exception as e:
        print(f"⚠️ Failed to move archived snapshot {snapshot_name} from {archive_root} to {snapshot_root}: {e}")
        return None

def find_run_folders(runs_dir: Path) -> List[Path]:
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)

def ensure_metrics_local(run_id: str, runs_dir: Path, snapshot_root: Path, artifacts_dir: Path) -> Optional[float]:
    """
    Local fallback implementation that mirrors the logic in experiment_postprocess.ensure_metrics_for_run.
    It will try to create runs/<run_id>/metrics.json from training_summary.json or training_log.csv.
    Returns best_f1 if found/written, else None.
    """
    import json, re
    run_path = runs_dir / run_id
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        try:
            j = json.loads(metrics_path.read_text(encoding="utf-8"))
            if j and ("best_f1" in j or "best_f1_score" in j):
                return float(j.get("best_f1") or j.get("best_f1_score"))
        except Exception:
            pass

    candidates = [
        run_path / "training_summary.json",
        snapshot_root / run_id / "artifacts_snapshot" / "training_summary.json",
        artifacts_dir / "training_summary.json",
    ]
    best = None
    for c in candidates:
        if c.exists():
            try:
                j = json.loads(c.read_text(encoding="utf-8"))
                for key in ("best_f1", "best_f1_score", "best_score"):
                    if key in j and j[key] not in (None, ""):
                        best = float(j[key])
                        break
                if best is not None:
                    break
            except Exception:
                continue

    if best is not None:
        run_path.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"best_f1": float(best)}, indent=2), encoding="utf-8")
        print(f"✅ Wrote metrics.json for {run_id} -> best_f1={best}")
        return best

    # fallback: parse training_log.csv
    log_candidates = [
        run_path / "training_log.csv",
        snapshot_root / run_id / "artifacts_snapshot" / "training_log.csv",
        artifacts_dir / "training_log.csv",
    ]
    for lc in log_candidates:
        if lc.exists():
            try:
                txt = lc.read_text(errors="ignore")
                m = re.search(r"Best F1[:=]\s*([0-9]*\.?[0-9]+)", txt)
                if m:
                    best = float(m.group(1))
                    run_path.mkdir(parents=True, exist_ok=True)
                    metrics_path.write_text(json.dumps({"best_f1": float(best)}, indent=2), encoding="utf-8")
                    print(f"✅ Wrote metrics.json for {run_id} from log -> best_f1={best}")
                    return best
            except Exception:
                continue
    return None

def main():
    p = argparse.ArgumentParser(description="Run orchestrator, restore archived snapshots (move), verify metrics, and run postprocess.")
    p.add_argument("--orchestrator-cmd", type=str, required=True, help="Command to run the orchestrator (quoted).")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--snapshot-root", type=str, default=r"C:\guardian_runs")
    p.add_argument("--archive-root", type=str, default=r"C:\guardian_runs_archive")
    p.add_argument("--artifacts-dir", type=str, default=r"C:\guardian_artifacts")
    p.add_argument("--postprocess-args", type=str, default="", help="Arguments to pass to scripts/experiment_postprocess.py (shell string).")
    p.add_argument("--restore-only", action="store_true", help="Only restore archived snapshots and verify metrics, skip running postprocess.")
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    snapshot_root = Path(args.snapshot_root)
    archive_root = Path(args.archive_root)
    artifacts_dir = Path(args.artifacts_dir)

    print("=== Starting orchestrator ===")
    start_time = time.time()
    rc = subprocess.run(args.orchestrator_cmd, shell=True).returncode
    if rc != 0:
        print(f"⚠️ Orchestrator exited with code {rc}. Continuing to verification (you may inspect runs/*/stderr.txt).")

    print("=== Orchestrator finished. Scanning runs folder ===")
    run_folders = find_run_folders(runs_dir)
    if not run_folders:
        print("No run folders found under", runs_dir)
    summary = []
    for run_path in run_folders:
        run_name = run_path.name  # e.g., run_1771697042_1 or similar
        # snapshot name expected to be run_name without leading 'run_' if present
        snap_name = run_name.replace("run_", "") if run_name.startswith("run_") else run_name
        snap_exists = (snapshot_root / snap_name / "artifacts_snapshot").exists()
        restored = False
        if not snap_exists:
            restored_path = restore_snapshot_from_archive(snapshot_root, archive_root, snap_name)
            if restored_path:
                print(f"🔁 Moved archived snapshot for {snap_name} -> {restored_path}")
                restored = True
                snap_exists = True
            else:
                print(f"ℹ️ No live snapshot for {snap_name} and no archived snapshot found.")
        # Ensure metrics.json exists (use imported helper if available)
        best = None
        if HAVE_POSTPROCESS_IMPORT:
            try:
                best = ensure_metrics_for_run(run_name, Path(args.runs_dir), Path(args.snapshot_root), Path(args.artifacts_dir), dry_run=False)
            except Exception as e:
                print(f"⚠️ ensure_metrics_for_run import failed for {run_name}: {e}")
                best = ensure_metrics_local(run_name, Path(args.runs_dir), snapshot_root, artifacts_dir)
        else:
            best = ensure_metrics_local(run_name, Path(args.runs_dir), snapshot_root, artifacts_dir)

        summary.append({
            "run": run_name,
            "snapshot_exists": snap_exists,
            "restored": restored,
            "best_f1": best
        })

    # Print concise summary
    print("\n=== Verification summary ===")
    for s in summary:
        print(f"- {s['run']}: snapshot_exists={s['snapshot_exists']}, restored={s['restored']}, best_f1={s['best_f1']}")

    if args.restore_only:
        print("Restore-only mode: skipping postprocess.")
        return

    # Run experiment_postprocess.py with provided args
    post_cmd = f'python scripts/experiment_postprocess.py {args.postprocess_args}'
    print("\n=== Running postprocess ===")
    print("Executing:", post_cmd)
    rc2 = subprocess.run(post_cmd, shell=True).returncode
    if rc2 != 0:
        print(f"⚠️ Postprocess exited with code {rc2}. Check output for details.")
    else:
        print("✅ Postprocess completed successfully.")

    print("=== Automation complete ===")

if __name__ == "__main__":
    main()
