import json
import glob
import random
from pathlib import Path

# Import centralized DATA_DIR from root config
try:
    from config import DATA_DIR
except ImportError:
    import sys
    from pathlib import Path
    # Ensure root directory is in path to find config.py
    sys.path.append(str(Path(__file__).parent.parent))
    from config import DATA_DIR

# The script now looks for individual .jsonl files in the data directory
ADV_DIR = DATA_DIR

# The final output file is also saved to the centralized data directory
OUTPUT = DATA_DIR / "llm_adv_merged.jsonl"

def load_jsonl(path):
    """Safely loads a JSONL file and yields each dictionary."""
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON line in {path.name}: {line[:80]}...")
                continue

def main():
    # Clear previous merge to ensure fresh data
    if OUTPUT.exists():
        OUTPUT.unlink()

    all_items = []
    seen = set()

    # List of all potential adversarial source files
    # This now explicitly includes your new hard negatives
    jsonl_files = [
        f for f in ADV_DIR.glob("*.jsonl")
        if f.name != OUTPUT.name
    ]

    print("🔍 Found JSONL files for merging:")
    for f in jsonl_files:
        print(f"   • {f.name}")

    for file in jsonl_files:
        count_before = len(all_items)
        for item in load_jsonl(file):
            # Deduplicate based on the unique Question-Answer pair
            key = (item.get("question"), item.get("answer"))
            if key not in seen:
                seen.add(key)
                all_items.append(item)
        
        count_after = len(all_items)
        print(f"📄 {file.name}: added {count_after - count_before} unique items")

    # Shuffle to ensure hard negatives are distributed throughout the training set
    random.shuffle(all_items)

    # Save the merged dataset
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item) + "\n")

    print(f"\n✅ MERGE COMPLETE: {len(all_items)} total unique samples saved to {OUTPUT.name}")

if __name__ == "__main__":
    main()