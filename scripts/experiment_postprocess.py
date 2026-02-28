#!/usr/bin/env python3
"""
scripts/experiment_postprocess.py

Single automated script to:
- discover runs in runs/
- ensure runs/<run_id>/metrics.json exists (create from training_summary.json if missing)
- build/update assignments CSV (experiments/plots/extra_analysis/seed_assignments.csv)
- fill best_f1 values into assignments
- optionally run the analyzer
- copy top-K winning models to a safe folder
- prune snapshots (move non-winners to archive)
- archive non-winner run folders
- supports --dry-run to preview actions

Usage examples:
  # Dry run to preview actions
  python scripts/experiment_postprocess.py --dry-run

  # Full run with analyzer and keep top 3 winners
  python scripts/experiment_postprocess.py \
    --runs-dir runs \
    --snapshot-root "C:\\guardian_runs" \
    --artifacts-dir "C:\\guardian_artifacts" \
    --assignments "experiments/plots/extra_analysis/seed_assignments.csv" \
    --analyzer-cmd "python experiments/plots/extra_analysis/analyze_experiment_results.py --runs-dir runs --assignments experiments/plots/extra_analysis/seed_assignments.csv" \
    --keep 3 \
    --archive-root "C:\\guardian_runs_archive" \
    --best-models "D:\\guardian_project\\best_models"
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional

def log(msg: str):
    print(msg)

def find_run_ids(runs_dir: Path) -> List[str]:
    if not runs_dir.exists():
        return []
    return sorted([p.name for p in runs_dir.iterdir() if p.is_dir()])

def read_json_safe(p: Path) -> Optional[dict]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def ensure_metrics_for_run(run_id: str, runs_dir: Path, snapshot_root: Path, artifacts_dir: Path, dry_run: bool=False) -> Optional[float]:
    """
    Ensure runs/<run_id>/metrics.json exists. If missing, try to create it from:
      - runs/<run_id>/training_summary.json
      - snapshot_root/<run_id>/artifacts_snapshot/training_summary.json
      - artifacts_dir/training_summary.json (global fallback)
      - training_log.csv pattern fallback
    Returns best_f1 (float) if found/written, else None.
    """
    run_path = runs_dir / run_id
    metrics_path = run_path / "metrics.json"
    if metrics_path.exists():
        j = read_json_safe(metrics_path)
        if j and ("best_f1" in j or "best_f1_score" in j):
            try:
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
            j = read_json_safe(c)
            if not j:
                continue
            for key in ("best_f1", "best_f1_score", "best_score"):
                if key in j and j[key] not in (None, ""):
                    try:
                        best = float(j[key])
                        break
                    except Exception:
                        continue
            if best is not None:
                break

    if best is not None:
        if dry_run:
            log(f"[dry-run] Would write metrics.json for {run_id} with best_f1={best}")
        else:
            run_path.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps({"best_f1": float(best)}, indent=2), encoding="utf-8")
            log(f"Wrote metrics.json for {run_id} -> best_f1={best}")
        return best

    # fallback: try to parse training_log.csv in snapshot or runs for "Best F1" pattern
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
                    if dry_run:
                        log(f"[dry-run] Would write metrics.json for {run_id} from log with best_f1={best}")
                    else:
                        run_path.mkdir(parents=True, exist_ok=True)
                        metrics_path.write_text(json.dumps({"best_f1": float(best)}, indent=2), encoding="utf-8")
                        log(f"Wrote metrics.json for {run_id} from log -> best_f1={best}")
                    return best
            except Exception:
                continue
    return None

def build_assignments_csv(run_ids: List[str], runs_dir: Path, assignments_csv: Path, default_arm: str="Unknown", dry_run: bool=False):
    """
    Create or update assignments CSV with columns:
    run_id,arm,seed,subset,outdir,best_f1
    If file exists, preserve existing rows and append missing runs.
    """
    rows: Dict[str, dict] = {}
    if assignments_csv.exists():
        with assignments_csv.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                rows[r.get("run_id","")] = r

    changed = False
    for rid in run_ids:
        if rid not in rows:
            outdir = str((runs_dir / rid).as_posix())
            rows[rid] = {"run_id": rid, "arm": default_arm, "seed": "", "subset": "", "outdir": outdir, "best_f1": ""}
            changed = True

    if dry_run:
        log(f"[dry-run] Would write/update assignments CSV at {assignments_csv} with {len(rows)} rows")
        return

    assignments_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["run_id","arm","seed","subset","outdir","best_f1"]
    with assignments_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rid in sorted(rows.keys()):
            writer.writerow({k: rows[rid].get(k,"") for k in fieldnames})
    log(f"Wrote assignments CSV: {assignments_csv} ({len(rows)} rows)")

def fill_best_f1_in_assignments(assignments_csv: Path, runs_dir: Path, snapshot_root: Path, artifacts_dir: Path, dry_run: bool=False):
    if not assignments_csv.exists():
        log(f"Assignments file not found: {assignments_csv}")
        return
    rows = []
    with assignments_csv.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        for r in reader:
            run_id = r.get("run_id","").strip()
            outdir = r.get("outdir","").strip() or str((runs_dir / run_id).as_posix())
            best = None
            # try runs/<run_id>/metrics.json
            metrics_path = Path(outdir) / "metrics.json"
            if metrics_path.exists():
                j = read_json_safe(metrics_path)
                if j:
                    best = j.get("best_f1") or j.get("best_f1_score")
            if best is None:
                # try snapshot training_summary
                ts = snapshot_root / run_id / "artifacts_snapshot" / "training_summary.json"
                if ts.exists():
                    j = read_json_safe(ts)
                    if j:
                        best = j.get("best_f1") or j.get("best_f1_score")
            if best is None:
                # try artifacts_dir training_summary.json
                ts2 = artifacts_dir / "training_summary.json"
                if ts2.exists():
                    j = read_json_safe(ts2)
                    if j:
                        best = j.get("best_f1") or j.get("best_f1_score")
            if best is not None:
                r["best_f1"] = str(best)
            rows.append(r)
    if dry_run:
        log(f"[dry-run] Would update assignments CSV {assignments_csv} with best_f1 values for {len(rows)} rows")
        return
    with assignments_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"Updated assignments CSV with best_f1 where available: {assignments_csv}")

def run_analyzer_cmd(cmd: str, dry_run: bool=False) -> int:
    if not cmd:
        log("No analyzer command provided; skipping analyzer run.")
        return 0
    log(f"Running analyzer command: {cmd}")
    if dry_run:
        log("[dry-run] Analyzer command not executed.")
        return 0
    try:
        rc = subprocess.run(cmd, shell=True).returncode
        log(f"Analyzer exited with code {rc}")
        return rc
    except Exception as e:
        log(f"Analyzer run failed: {e}")
        return 1

def collect_best_f1s_from_runs(run_ids: List[str], runs_dir: Path, snapshot_root: Path) -> Dict[str, float]:
    results = {}
    for rid in run_ids:
        # try runs/<rid>/metrics.json
        m = runs_dir / rid / "metrics.json"
        if m.exists():
            j = read_json_safe(m)
            if j and ("best_f1" in j or "best_f1_score" in j):
                try:
                    results[rid] = float(j.get("best_f1") or j.get("best_f1_score"))
                    continue
                except Exception:
                    pass
        # try snapshot training_summary
        ts = snapshot_root / rid / "artifacts_snapshot" / "training_summary.json"
        if ts.exists():
            j = read_json_safe(ts)
            if j and ("best_f1" in j or "best_f1_score" in j):
                try:
                    results[rid] = float(j.get("best_f1") or j.get("best_f1_score"))
                    continue
                except Exception:
                    pass
    return results

def copy_winners(winners: List[str], snapshot_root: Path, dest_dir: Path, dry_run: bool=False):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for rid in winners:
        src_model = snapshot_root / rid / "artifacts_snapshot" / "guardian_spider_native.pth"
        src_full = snapshot_root / rid / "artifacts_snapshot" / "guardian_spider_native_full.pth"
        if src_model.exists():
            dst = dest_dir / f"{rid}.guardian_spider_native.pth"
            if dry_run:
                log(f"[dry-run] Would copy {src_model} -> {dst}")
            else:
                shutil.copy2(src_model, dst)
                log(f"Copied model for {rid} -> {dst}")
        if src_full.exists():
            dstf = dest_dir / f"{rid}.guardian_spider_native_full.pth"
            if dry_run:
                log(f"[dry-run] Would copy {src_full} -> {dstf}")
            else:
                shutil.copy2(src_full, dstf)
                log(f"Copied full model for {rid} -> {dstf}")

def prune_snapshots_keep_topk(snapshot_root: Path, best_map: Dict[str, float], keep_k: int, archive_root: Path, dry_run: bool=False):
    """
    Keep top-k snapshots by best_f1. Move others to archive_root (fast on same drive).
    """
    if keep_k <= 0:
        log("keep_k <= 0, skipping prune.")
        return
    sorted_runs = sorted(best_map.items(), key=lambda kv: kv[1], reverse=True)
    keep = [rid for rid, _ in sorted_runs[:keep_k]]
    log(f"Top {keep_k} runs to keep: {keep}")
    archive_root.mkdir(parents=True, exist_ok=True)
    for d in snapshot_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if name in keep:
            log(f"Keeping snapshot: {name}")
            continue
        dest = archive_root / name
        if dry_run:
            log(f"[dry-run] Would move snapshot {d} -> {dest}")
        else:
            shutil.move(str(d), str(dest))
            log(f"Moved snapshot {name} -> archive")

def archive_run_folders(runs_dir: Path, keep_run_ids: List[str], archive_root: Path, dry_run: bool=False):
    archive_root.mkdir(parents=True, exist_ok=True)
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name in keep_run_ids:
            log(f"Keeping run folder: {d.name}")
            continue
        dest = archive_root / d.name
        if dry_run:
            log(f"[dry-run] Would move run folder {d} -> {dest}")
        else:
            shutil.move(str(d), str(dest))
            log(f"Moved run folder {d.name} -> archive")

def parse_args():
    p = argparse.ArgumentParser(description="Post-process experiment runs: ensure metrics, build assignments, run analyzer, copy winners, prune snapshots.")
    p.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run folders")
    p.add_argument("--snapshot-root", type=str, default=r"C:\guardian_runs", help="Root where snapshots are stored")
    p.add_argument("--artifacts-dir", type=str, default=r"C:\guardian_artifacts", help="Canonical artifacts dir")
    p.add_argument("--assignments", type=str, default="experiments/plots/extra_analysis/seed_assignments.csv", help="Path to assignments CSV to create/update")
    p.add_argument("--analyzer-cmd", type=str, default="", help="Command to run analyzer (shell string)")
    p.add_argument("--keep", type=int, default=3, help="Number of top snapshots to keep")
    p.add_argument("--archive-root", type=str, default=r"C:\guardian_runs_archive", help="Where to move archived snapshots")
    p.add_argument("--best-models", type=str, default=r"D:\guardian_project\best_models", help="Where to copy winning models")
    p.add_argument("--dry-run", action="store_true", help="Show actions without making changes")
    return p.parse_args()

def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    snapshot_root = Path(args.snapshot_root)
    artifacts_dir = Path(args.artifacts_dir)
    assignments_csv = Path(args.assignments)
    archive_root = Path(args.archive_root)
    best_models_dir = Path(args.best_models)
    dry_run = args.dry_run

    log("=== Experiment postprocess started ===")
    log(f"runs_dir: {runs_dir}, snapshot_root: {snapshot_root}, artifacts_dir: {artifacts_dir}")
    run_ids = find_run_ids(runs_dir)
    log(f"Found {len(run_ids)} run folders in {runs_dir}")

    # 1) Ensure metrics.json exists for each run (try to create from training_summary)
    best_map = {}
    for rid in run_ids:
        best = ensure_metrics_for_run(rid, runs_dir, snapshot_root, artifacts_dir, dry_run=dry_run)
        if best is not None:
            best_map[rid] = best

    # 2) Build or update assignments CSV
    build_assignments_csv(run_ids, runs_dir, assignments_csv, default_arm="Unknown", dry_run=dry_run)

    # 3) Fill best_f1 in assignments CSV
    fill_best_f1_in_assignments(assignments_csv, runs_dir, snapshot_root, artifacts_dir, dry_run=dry_run)

    # 4) Re-collect best_f1s (in case fill wrote new metrics)
    best_map = collect_best_f1s_from_runs(run_ids, runs_dir, snapshot_root)
    log(f"Collected best_f1 for {len(best_map)} runs")

    # 5) Run analyzer if requested
    if args.analyzer_cmd:
        rc = run_analyzer_cmd(args.analyzer_cmd, dry_run=dry_run)
        if rc != 0 and not dry_run:
            log("Analyzer returned non-zero exit code; continuing with postprocess.")

    # 6) Copy winners
    winners: List[str] = []
    if best_map:
        sorted_runs = sorted(best_map.items(), key=lambda kv: kv[1], reverse=True)
        winners = [rid for rid, _ in sorted_runs[:args.keep]]
        log(f"Winners selected: {winners}")
        copy_winners(winners, snapshot_root, best_models_dir, dry_run=dry_run)
    else:
        log("No best_f1 values found; skipping winner copy.")

    # 7) Prune snapshots (move non-winners to archive)
    if best_map:
        prune_snapshots_keep_topk(snapshot_root, best_map, args.keep, archive_root, dry_run=dry_run)
    else:
        log("No best_f1 values found; skipping snapshot prune.")

    # 8) Archive run folders (keep winners' run folders)
    keep_run_ids = winners if best_map else []
    archive_run_folders(runs_dir, keep_run_ids, archive_root, dry_run=dry_run)

    log("=== Postprocess complete ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())
