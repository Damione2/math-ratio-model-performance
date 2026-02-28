# D:\guardian_project\config.py
from pathlib import Path
import os

# Get the actual drive letter where this project is located
PROJECT_ROOT = Path(__file__).parent.resolve()
PROJECT_DRIVE = PROJECT_ROOT.drive  # 'D:' or whatever drive the project is on

# Option 1: Use C: drive (absolute paths - your current setup)
# Use this if you want to keep artifacts on C: drive regardless of project location
USE_ABSOLUTE_PATHS = True

if USE_ABSOLUTE_PATHS:
    # Absolute paths on C: drive (your current setup)
    ARTIFACTS_DIR = Path("C:/guardian_artifacts")
    DATA_DIR = Path("C:/guardian_data")
else:
    # Option 2: Use project drive (D: in your case)
    # Uncomment below to put artifacts on same drive as project
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    DATA_DIR = PROJECT_ROOT / "data"

# Create directories
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Debug info (remove after confirming paths)
if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print(f"Data: {DATA_DIR}")
    print(f"Artifacts exists: {ARTIFACTS_DIR.exists()}")
    print(f"Data exists: {DATA_DIR.exists()}")