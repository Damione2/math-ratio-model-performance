# pipeline/snapshot_utils.py
"""
Lightweight snapshot utilities for Guardian pipeline.

Purpose
-------
Create per-run snapshots on the same drive as the canonical artifacts directory
(e.g., C:\guardian_artifacts -> snapshots under C:\guardian_runs\<run_id>\artifacts_snapshot).
Large files are hard-linked when possible (no extra disk usage). Small files are copied.
If linking fails, the function falls back to copying.

Usage
-----
from pipeline.snapshot_utils import snapshot_artifacts_on_C
snapshot_path = snapshot_artifacts_on_C(
    run_id="run_1771521950_0",
    artifacts_dir=r"C:\guardian_artifacts",
    snapshot_root=r"C:\guardian_runs"
)
"""

from __future__ import annotations
import os
import shutil
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


def same_drive(p1: Path, p2: Path) -> bool:
    """Return True if p1 and p2 are on the same drive/volume (Windows-friendly)."""
    try:
        return str(p1.resolve()).split(":", 1)[0].upper() == str(p2.resolve()).split(":", 1)[0].upper()
    except Exception:
        return False


def sha256_of_file(path: Path, chunk: int = 8192) -> Optional[str]:
    """Compute SHA256 checksum of a file. Returns None on error."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None


def link_or_copy(src: Path, dst: Path) -> str:
    """
    Try to create a hard link dst -> src. If that fails, fall back to copy2.
    Returns 'hardlink' or 'copied'.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    try:
        os.link(str(src), str(dst))
        return "hardlink"
    except Exception:
        shutil.copy2(str(src), str(dst))
        return "copied"


def snapshot_artifacts_on_C(
    run_id: str,
    artifacts_dir: str = r"C:\guardian_artifacts",
    snapshot_root: str = r"C:\guardian_runs",
    small_files: Optional[List[str]] = None,
    large_files: Optional[List[str]] = None,
    extra_files: Optional[List[str]] = None,
    write_checksums: bool = True,
) -> Path:
    """
    Create a per-run snapshot on C: that hard-links large files and copies small files.
    Accepts extra_files: list of absolute or relative paths to include in the snapshot
    (useful for per-run files like runs/<run>/metrics.json).
    """
    artifacts_dir = Path(artifacts_dir)
    snapshot_root = Path(snapshot_root)
    run_snapshot = snapshot_root / run_id / "artifacts_snapshot"
    run_snapshot.mkdir(parents=True, exist_ok=True)

    small_files = small_files or [
        "metrics.json",
        "training_summary.json",
        "training_log.csv",
        "train_meta.json",
        "01_manifest.json",
    ]
    large_files = large_files or [
        "guardian_spider_native.pth",
        "guardian_spider_native_full.pth",
        "train_features.bin",
        "val_features.bin",
        "val_metadata.pkl",
        "scaler.pkl",
    ]

    pointers: Dict[str, Any] = {}

    # Normalize extra_files: resolve relative paths to absolute where possible
    _extra_resolved: List[Path] = []
    if extra_files:
        for ef in extra_files:
            try:
                p = Path(ef)
                if not p.is_absolute():
                    # try relative to cwd first, then to artifacts_dir, then to runs/
                    cand = Path.cwd() / p
                    if cand.exists():
                        p = cand
                    else:
                        cand2 = artifacts_dir / p
                        if cand2.exists():
                            p = cand2
                        else:
                            # leave as-is (may be created later or missing)
                            p = Path(ef)
                _extra_resolved.append(p)
            except Exception:
                continue

    # Copy small files from artifacts_dir (by name)
    for fname in small_files:
        src = artifacts_dir / fname
        key = fname
        if src.exists():
            try:
                dst = run_snapshot / src.name
                shutil.copy2(src, dst)
                pointers[key] = {"type": "copied", "path": str(dst)}
            except Exception as e:
                pointers[key] = {"type": "error", "error": str(e)}
        else:
            pointers[key] = {"type": "missing"}

    # Handle explicit extra files (absolute or resolved paths)
    for p in _extra_resolved:
        key = p.name
        try:
            if not p.exists():
                pointers[key] = {"type": "missing", "requested_path": str(p)}
                continue
            dst = run_snapshot / p.name
            # If the extra file is on the same drive as snapshot_root and is large, attempt hardlink
            try:
                # prefer hardlink when possible to avoid extra disk usage
                if same_drive(p, run_snapshot):
                    os.link(str(p), str(dst))
                    method = "hardlink"
                else:
                    shutil.copy2(str(p), str(dst))
                    method = "copied"
            except Exception:
                shutil.copy2(str(p), str(dst))
                method = "copied"
            entry: Dict[str, Any] = {"type": method, "path": str(dst)}
            if write_checksums and p.is_file():
                entry["sha256"] = sha256_of_file(p)
            pointers[key] = entry
        except Exception as e:
            pointers[key] = {"type": "error", "error": str(e)}

    # Hard-link or copy large files (from artifacts_dir)
    for fname in large_files:
        src = artifacts_dir / fname
        key = fname
        if not src.exists():
            pointers[key] = {"type": "missing"}
            continue
        dst = run_snapshot / src.name
        try:
            method = link_or_copy(src, dst)
            entry: Dict[str, Any] = {"type": method, "path": str(dst)}
            if write_checksums:
                entry["sha256"] = sha256_of_file(src)
            pointers[key] = entry
        except Exception as e:
            pointers[key] = {"type": "error", "error": str(e)}

    # Write pointer/manifest file
    manifest = {
        "snapshot_time": time.time(),
        "artifacts_dir": str(artifacts_dir),
        "snapshot_root": str(run_snapshot),
        "pointers": pointers,
    }
    try:
        (run_snapshot / "artifact_pointers.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        # best-effort: ignore write errors but return snapshot path
        pass

    return run_snapshot


if __name__ == "__main__":
    # Quick local test (only runs when executed directly)
    import argparse

    parser = argparse.ArgumentParser(description="Create a snapshot of C:\\guardian_artifacts on C:\\guardian_runs")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--artifacts-dir", type=str, default=r"C:\guardian_artifacts")
    parser.add_argument("--snapshot-root", type=str, default=r"C:\guardian_runs")
    args = parser.parse_args()

    snap = snapshot_artifacts_on_C(run_id=args.run_id, artifacts_dir=args.artifacts_dir, snapshot_root=args.snapshot_root)
    print("Snapshot created at:", snap)
