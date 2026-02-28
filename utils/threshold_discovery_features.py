#!/usr/bin/env python3
"""
threshold_discovery_features.py (Updated for Step 3 Memmap Compatibility)
----------------------------------------------------------------------
Feature-based threshold discovery for GuardianVisionNet.

This module:
  - Loads validation memmaps from Step 3 (.bin + .pkl format)
  - Loads Guardian model + scaler (same as tester)
  - Runs inference directly on feature tensors
  - Computes optimal per-domain thresholds (F1-maximizing)
  - Computes global threshold
  - Saves results to ARTIFACTS_DIR/thresholds_features.json

UPDATED: Fixed PyTorch 2.0+ AMP API (torch.amp.autocast instead of torch.cuda.amp.autocast).
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

import torch

# Disable Unsloth patches before importing Guardian
os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_DTYPE"] = "float32"

try:
    from config import ARTIFACTS_DIR
    from core.guardian_vision_core import GuardianVisionNet
    from core.guardian_utils import GuardianScaler
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR
    from core.guardian_vision_core import GuardianVisionNet
    from core.guardian_utils import GuardianScaler


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def find_optimal_threshold(probs, labels):
    """
    Returns threshold that maximizes F1.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return float(best_t), float(best_f1)


# ------------------------------------------------------------
# Load validation memmaps (UPDATED for Step 3 format)
# ------------------------------------------------------------
def load_validation_memmaps():
    """
    Load validation data from Step 3 memmap format.
    Step 3 produces: {split}_features.bin + {split}_metadata.pkl
    """
    bin_path = ARTIFACTS_DIR / "val_features.bin"
    meta_path = ARTIFACTS_DIR / "val_metadata.pkl"
    
    if not bin_path.exists():
        raise FileNotFoundError(f"Validation features not found: {bin_path}. Run Step 3 first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Validation metadata not found: {meta_path}. Run Step 3 first.")
    
    print(f"Loading validation memmaps from Step 3...")
    print(f"  Features: {bin_path}")
    print(f"  Metadata: {meta_path}")
    
    # Load metadata (list of dicts)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)
    
    n_samples = len(metadata)
    
    # Load memmap
    X = np.memmap(bin_path, dtype='float16', mode='r', shape=(n_samples, 29, 8, 1536))
    
    # Extract labels and domains from metadata list
    y = np.array([m['label'] for m in metadata], dtype=np.int64)
    
    # Handle domain mapping with fallback
    domain_list = []
    for m in metadata:
        domain = m.get('domain', 'real_world')
        # Normalize domain names
        if domain in ['real', 'realworld', 'real_world']:
            domain = 'real_world'
        elif domain in ['math', 'mathematics']:
            domain = 'math'
        elif domain in ['code', 'coding', 'programming']:
            domain = 'code'
        domain_list.append(domain)
    
    d = np.array(domain_list)
    
    # Map string domains to indices
    domain_to_idx = {'math': 0, 'code': 1, 'real_world': 2}
    d = np.array([domain_to_idx.get(str(x).lower(), 2) for x in d], dtype=np.int64)
    
    print(f"Loaded: {X.shape[0]} samples, shape={X.shape}")
    print(f"Domain distribution: math={(d==0).sum()}, code={(d==1).sum()}, real_world={(d==2).sum()}")
    
    return X, y, d


# ------------------------------------------------------------
# Load model + scaler (FIXED: No arguments to GuardianVisionNet)
# ------------------------------------------------------------
def load_model_and_scaler(device):
    # Load scaler
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}. Run Step 3 (training split) first.")
    
    scaler = GuardianScaler()
    scaler.load(str(scaler_path))
    print(f"✅ Scaler loaded from {scaler_path}")

    # Load model - try multiple possible filenames
    model_files = [
        "guardian_spider_native.pth",
        "guardian_moe_final.pth", 
        "guardian_best.pth",
        "guardian_final.pth"
    ]
    
    model_path = None
    for fname in model_files:
        candidate = ARTIFACTS_DIR / fname
        if candidate.exists():
            model_path = candidate
            break
    
    if model_path is None:
        raise FileNotFoundError(f"Model not found. Tried: {model_files}")
    
    print(f"Loading model from {model_path}")
    
    # Force fp32 for dtype safety
    torch.set_default_dtype(torch.float32)
    
    # FIXED: No arguments to GuardianVisionNet - dimensions are hardcoded
    model = GuardianVisionNet().to(device)
    
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    
    # Force fp32 and set inference mode
    model = model.float()
    model.eval()
    model.set_inference_mode(True)  # Prevents CfC state leakage
    
    print(f"✅ Model loaded (fp32 mode, inference_mode=True)")

    return model, scaler


# ------------------------------------------------------------
# Run inference on memmap features (FIXED: PyTorch 2.0+ AMP API)
# ------------------------------------------------------------
def run_inference_on_features(model, scaler, X, device, batch_size=64):
    """
    X: (N, 29, 8, 1536) - Spider-Triangulation features from memmap
    Scaler expects: (N, 29*8*1536) = (N, 356352) - fully flattened
    """
    N = X.shape[0]
    probs = np.zeros(N, dtype=np.float32)
    domains_pred = np.zeros(N, dtype=np.int64)
    domain_conf = np.zeros(N, dtype=np.float32)

    print(f"Running inference on {N} samples (batch_size={batch_size})...")
    print(f"  Feature shape: {X.shape}")
    print(f"  Scaler expects: (N, {29*8*1536}) fully flattened")

    # Process in batches
    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        bsz = batch_end - i
        
        # Load batch from memmap and convert to numpy
        batch_np = np.array(X[i:batch_end])  # (bsz, 29, 8, 1536)
        
        # Flatten ENTIRE batch for scaler: (bsz, 29*8*1536) = (bsz, 356352)
        batch_flat = batch_np.reshape(bsz, -1)
        
        # Convert to torch tensor
        batch_tensor = torch.from_numpy(batch_flat.astype(np.float32))
        
        # Apply scaler (expects flattened input)
        scaled_flat = scaler.transform(batch_tensor)
        
        # Reshape back to (bsz, 29, 8, 1536) for model
        scaled = scaled_flat.reshape(bsz, 29, 8, 1536)
        
        # Move to device and run inference
        scaled_device = scaled.to(device)
        
        with torch.no_grad():
            # ✅ FIXED: Use torch.amp.autocast for PyTorch 2.0+ compatibility
            autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
            with torch.amp.autocast(device_type=autocast_device, enabled=False):
                logits, routing_info = model(scaled_device)
                
                # Hallucination probability
                p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

                # Domain routing
                if isinstance(routing_info, dict):
                    if 'train_routing_probs' in routing_info:
                        gate_probs = routing_info['train_routing_probs']
                    else:
                        gate_probs = torch.ones(bsz, 3, device=device) / 3
                    conf, pred = torch.max(gate_probs, dim=-1)
                else:
                    gate_probs = torch.softmax(routing_info, dim=-1)
                    conf, pred = torch.max(gate_probs, dim=-1)

                probs[i:batch_end] = p
                domains_pred[i:batch_end] = pred.cpu().numpy()
                domain_conf[i:batch_end] = conf.cpu().numpy()
        
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {batch_end}/{N} samples...")
        
        # Cleanup
        del batch_np, batch_flat, batch_tensor, scaled_flat, scaled, scaled_device
        torch.cuda.empty_cache()

    return probs, domains_pred, domain_conf


# ------------------------------------------------------------
# Main threshold discovery
# ------------------------------------------------------------
def discover_thresholds():
    print("\n" + "="*60)
    print("🔍 FEATURE-BASED THRESHOLD DISCOVERY")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    X, y, d = load_validation_memmaps()

    # Load model + scaler
    model, scaler = load_model_and_scaler(device)

    # Run inference
    probs, domains_pred, domain_conf = run_inference_on_features(
        model, scaler, X, device
    )

    # Compute thresholds per domain (using ground truth domains from metadata)
    domain_names = {0: "math", 1: "code", 2: "real_world"}
    thresholds = {}
    diagnostics = {}

    print("\n📊 Per-domain threshold optimization:")
    
    for dom_idx, dom_name in domain_names.items():
        mask = (d == dom_idx)
        n_samples = mask.sum()
        
        if n_samples < 10:
            print(f"  ⚠️ {dom_name}: too few samples ({n_samples}), using default 0.45")
            thresholds[dom_name] = 0.45
            diagnostics[dom_name] = {"f1": 0.0, "n": int(n_samples)}
            continue

        dom_probs = probs[mask]
        dom_labels = y[mask]

        t, f1 = find_optimal_threshold(dom_probs, dom_labels)
        thresholds[dom_name] = t
        diagnostics[dom_name] = {"f1": f1, "n": int(n_samples)}

        print(f"  ✓ {dom_name:12s}: threshold={t:.3f}, F1={f1:.3f}, n={n_samples}")

    # Global threshold
    global_t, global_f1 = find_optimal_threshold(probs, y)
    print(f"\n🌍 Global: threshold={global_t:.3f}, F1={global_f1:.3f}")

    # Save results
    out = {
        "thresholds": thresholds,
        "global_threshold": global_t,
        "diagnostics": diagnostics,
        "timestamp": str(np.datetime64('now')),
        "n_total": len(y)
    }

    out_path = ARTIFACTS_DIR / "thresholds_features.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n💾 Saved thresholds to {out_path}")
    print("="*60)
    
    return out


# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
if __name__ == "__main__":
    discover_thresholds()