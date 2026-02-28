# utils/convert2memmap.py (updated)
import numpy as np
import pickle
import gc
import json
from pathlib import Path
from typing import List, Dict

# USE CENTRALIZED CONFIG - not hardcoded path
try:
    from config import ARTIFACTS_DIR
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR

print(f"[INFO] Using ARTIFACTS_DIR: {ARTIFACTS_DIR}")

# Canonical domain mapping used across the pipeline
DOMAIN_MAP: Dict[str, int] = {
    "math": 0,
    "code": 1,
    "real_world": 2,
    "real": 2,  # accept legacy alias
}

def _safe_collect_samples(data_list: List[Dict]) -> Dict[str, List]:
    """
    Validate and collect features, labels and domains from data_list.
    Skips samples missing required keys but logs a warning.
    Returns dict with keys: features, labels, domains (raw domain strings).
    Raises ValueError if no valid samples are found.
    """
    feat_list = []
    lbl_list = []
    dom_list = []
    skipped = 0

    for i, d in enumerate(data_list):
        if not isinstance(d, dict):
            print(f"[WARN] Skipping non-dict sample at index {i}")
            skipped += 1
            continue

        # Required keys
        if "features" not in d or "label" not in d or "domain" not in d:
            print(f"[WARN] Skipping sample {i} due to missing keys: {list(d.keys())}")
            skipped += 1
            continue

        feat = d["features"]
        lbl = d["label"]
        dom = d["domain"]

        # Basic shape/type checks for features (expect array-like)
        try:
            arr = np.asarray(feat)
        except Exception as e:
            print(f"[WARN] Could not convert features to array for sample {i}: {e}")
            skipped += 1
            continue

        # Accept features that are 2D (28, 1536) or can be coerced to that shape
        if arr.ndim != 2:
            print(f"[WARN] Skipping sample {i} with unexpected feature ndim={arr.ndim}")
            skipped += 1
            continue

        feat_list.append(arr.astype(np.float32))
        lbl_list.append(int(lbl))
        dom_list.append(str(dom))

    if len(feat_list) == 0:
        raise ValueError("No valid samples found in provided data_list after validation.")

    if skipped:
        print(f"[INFO] Skipped {skipped} invalid samples during collection.")

    return {"features": feat_list, "labels": lbl_list, "domains": dom_list}


def convert_set(pkl_name: str, prefix: str):
    """
    Convert a pickled list of dict samples into three .npy memmap files:
      - {prefix}_features.npy  shape: (N, 28, 1536) dtype: float32
      - {prefix}_labels.npy    shape: (N,) dtype: int64
      - {prefix}_domains.npy   shape: (N,) dtype: int64 (mapped via DOMAIN_MAP)

    The input pickle is expected to contain a list of dicts with keys:
      - 'features' : array-like of shape (28, 1536)
      - 'label'    : int (0 or 1)
      - 'domain'   : str (e.g., 'math', 'code', 'real' or 'real_world')

    The function is defensive: it validates keys and shapes, skips invalid samples,
    writes memmaps safely and saves a small metadata JSON for reproducibility.
    """
    pkl_path = ARTIFACTS_DIR / pkl_name
    if not pkl_path.exists():
        print(f"[SKIP] {pkl_name} not found under {ARTIFACTS_DIR}.")
        return

    print(f"[LOAD] Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data_list = pickle.load(f)

    if not isinstance(data_list, list):
        raise ValueError(f"Expected a list in {pkl_path}, got {type(data_list)}")

    print(f"[INFO] Validating and collecting samples from {pkl_name} ...")
    collected = _safe_collect_samples(data_list)

    # Stack features into a single contiguous array
    features = np.stack(collected["features"], axis=0).astype(np.float32)  # (N, 28, 1536)
    labels = np.asarray(collected["labels"], dtype=np.int64)
    # Map domain strings to ints using DOMAIN_MAP (default to 'real_world' index if unknown)
    domains = np.asarray([DOMAIN_MAP.get(d.lower(), DOMAIN_MAP["real_world"]) for d in collected["domains"]], dtype=np.int64)

    # Validate shapes
    if features.ndim != 3:
        raise ValueError(f"Expected features to be 3D (N, 28, 1536), got shape {features.shape}")

    if features.shape[1] != 28 or features.shape[2] != 1536:
        raise ValueError(f"Unexpected feature dimensions: expected (N, 28, 1536), got {features.shape}")

    if labels.shape[0] != features.shape[0] or domains.shape[0] != features.shape[0]:
        raise ValueError("Labels/domains length mismatch with features count")

    print(f"[SAVE] Creating memmaps for prefix '{prefix}' (N={features.shape[0]}) ...")

    feat_path = ARTIFACTS_DIR / f"{prefix}_features.npy"
    lbl_path = ARTIFACTS_DIR / f"{prefix}_labels.npy"
    dom_path = ARTIFACTS_DIR / f"{prefix}_domains.npy"

    # Create writable memmap files and write data, then flush explicitly
    fp = np.lib.format.open_memmap(feat_path, mode="w+", dtype=features.dtype, shape=features.shape)
    try:
        fp[:] = features[:]
        fp.flush()
    finally:
        del fp

    lp = np.lib.format.open_memmap(lbl_path, mode="w+", dtype=labels.dtype, shape=labels.shape)
    try:
        lp[:] = labels[:]
        lp.flush()
    finally:
        del lp

    dp = np.lib.format.open_memmap(dom_path, mode="w+", dtype=domains.dtype, shape=domains.shape)
    try:
        dp[:] = domains[:]
        dp.flush()
    finally:
        del dp

    # Save metadata for reproducibility
    meta = {
        "num_samples": int(features.shape[0]),
        "feature_shape": [int(features.shape[1]), int(features.shape[2])],
        "feature_dtype": str(features.dtype),
        "label_dtype": str(labels.dtype),
        "domain_dtype": str(domains.dtype),
        "domain_map": DOMAIN_MAP,
        "source_pickle": str(pkl_path.name),
    }
    meta_path = ARTIFACTS_DIR / f"{prefix}_meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(f"[OK] {prefix} memmaps created: {feat_path}, {lbl_path}, {dom_path}")
    print(f"[META] Saved metadata to {meta_path}")

    # Clean up large objects and run GC
    del data_list, collected, features, labels, domains
    gc.collect()


if __name__ == "__main__":
    # Convert standard splits if present
    convert_set("03_train_features.pkl", "train")
    convert_set("03_val_features.pkl", "val")
    convert_set("03_test_features.pkl", "test")
    print("[DONE] All requested memmap conversions completed.")
