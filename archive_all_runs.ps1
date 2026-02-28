# archive_all_runs.ps1
# Archive ALL runs and snapshots (dry-run then confirm)
# Usage: run from project root: .\archive_all_runs.ps1

# Configuration (edit if you want different archive locations)
$archiveRunsDir = Join-Path (Get-Location) "runs_archive"
$archiveSnapshotsDir = "C:\guardian_runs_archive"
$runsDir = Join-Path (Get-Location) "runs"
$snapshotsRoot = "C:\guardian_runs"

# Create archive dirs if missing
New-Item -ItemType Directory -Path $archiveRunsDir -Force | Out-Null
New-Item -ItemType Directory -Path $archiveSnapshotsDir -Force | Out-Null

# Gather run folders and snapshot folders
$runs = @()
if (Test-Path $runsDir) { $runs = Get-ChildItem $runsDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending }
$snapshots = @()
if (Test-Path $snapshotsRoot) { $snapshots = Get-ChildItem $snapshotsRoot -Directory -ErrorAction SilentlyContinue | Sort-Object Name }

# Dry-run summary
Write-Host "=== DRY RUN: Archive All Runs ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Runs found under $runsDir : $($runs.Count)" -ForegroundColor Yellow
if ($runs.Count -gt 0) { $runs | Select-Object Name,LastWriteTime | Format-Table -AutoSize } else { Write-Host "  (no run folders found)" }

Write-Host ""
Write-Host "Snapshots found under $snapshotsRoot : $($snapshots.Count)" -ForegroundColor Yellow
if ($snapshots.Count -gt 0) { $snapshots | Select-Object Name,LastWriteTime | Format-Table -AutoSize } else { Write-Host "  (no snapshot folders found)" }

Write-Host ""
Write-Host "Planned moves (dry-run):" -ForegroundColor Cyan
if ($runs.Count -gt 0) {
  Write-Host "  Run folders -> $archiveRunsDir" -ForegroundColor Green
  foreach ($r in $runs) { Write-Host "    Move: $($r.FullName) -> $($archiveRunsDir)\$($r.Name)" }
} else {
  Write-Host "  No run folders to move."
}
if ($snapshots.Count -gt 0) {
  Write-Host "  Snapshot folders -> $archiveSnapshotsDir" -ForegroundColor Green
  foreach ($s in $snapshots) { Write-Host "    Move: $($s.FullName) -> $($archiveSnapshotsDir)\$($s.Name)" }
} else {
  Write-Host "  No snapshot folders to move."
}

Write-Host ""
Write-Host "WARNING: This will MOVE (not copy) the folders above. This action is reversible only by moving them back from the archive." -ForegroundColor Red
Write-Host ""

# Confirm
$confirm = Read-Host "Type YES to proceed with archiving ALL runs and snapshots"
if ($confirm -ne "YES") {
  Write-Host "Aborted by user. No changes made." -ForegroundColor Yellow
  exit 0
}

# Perform moves
Write-Host "`n=== Archiving now ===" -ForegroundColor Cyan

# Move runs
if ($runs.Count -gt 0) {
  foreach ($r in $runs) {
    $dstRun = Join-Path $archiveRunsDir $r.Name
    try {
      Move-Item -Path $r.FullName -Destination $dstRun -Force
      Write-Host "Moved run: $($r.Name) -> $dstRun" -ForegroundColor Green
    } catch {
      Write-Host "Failed to move run $($r.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
  }
} else {
  Write-Host "No run folders to move." -ForegroundColor Yellow
}

# Move snapshots
if ($snapshots.Count -gt 0) {
  foreach ($s in $snapshots) {
    $dstSnap = Join-Path $archiveSnapshotsDir $s.Name
    try {
      Move-Item -Path $s.FullName -Destination $dstSnap -Force
      Write-Host "Moved snapshot: $($s.Name) -> $dstSnap" -ForegroundColor Green
    } catch {
      Write-Host "Failed to move snapshot $($s.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
  }
} else {
  Write-Host "No snapshot folders to move." -ForegroundColor Yellow
}

Write-Host "`nArchive complete." -ForegroundColor Cyan
Write-Host "Runs archived to: $archiveRunsDir"
Write-Host "Snapshots archived to: $archiveSnapshotsDir"
Write-Host "You can restore a snapshot with: Move-Item 'C:\guardian_runs_archive\<snap>' 'C:\guardian_runs\<snap>' -Force"
