# core/guardian_tester_moe.py (v15 - Biomimetic Diagnostics & State Management + Dtype Safety + PyTorch 2.0+ Fix)
import os
import torch
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import sys

# ✅ CRITICAL: Disable Unsloth patches BEFORE any other imports
os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
os.environ["UNSLOTH_DISABLE_FLASH_ATTENTION"] = "1"
os.environ["TORCH_DTYPE"] = "float32"

# Import from core module
from .guardian_vision_core import GuardianVisionNet
from .guardian_utils import FeatureExtractor

# Update ARTIFACTS_DIR to use centralized config
try:
    from config import ARTIFACTS_DIR
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR

# Internal cache for resources
_RESOURCES = {"model": None, "extractor": None, "device": None, "temps": None, "global_temp": None}

def _get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the appropriate compute device"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_all_resources(device: torch.device = None,
                      model_path: Path = None,
                      extractor_kwargs: Dict[str, Any] = None) -> None:
    """
    Initialize and cache model and extractor for inference.
    ✅ ENABLES inference mode by default for stability.
    ✅ LOADS both domain and global temperatures.
    ✅ RESETS CfC state buffer to prevent cross-contamination.
    ✅ FORCES fp32 dtype to prevent Half/Float mismatch.
    """
    if device is None:
        device = _get_device()
    _RESOURCES["device"] = device

    # Set default dtype to fp32 for this device
    torch.set_default_dtype(torch.float32)

    # Extractor
    if _RESOURCES["extractor"] is None:
        try:
            extractor_kwargs = extractor_kwargs or {}
            _RESOURCES["extractor"] = FeatureExtractor(**extractor_kwargs)
            print("✅ FeatureExtractor ready")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FeatureExtractor: {e}")

    # Model
    if _RESOURCES["model"] is None:
        try:
            mdl = GuardianVisionNet()
        except Exception as e:
            raise RuntimeError(f"Failed to construct GuardianVisionNet: {e}")

        model_path = Path(model_path) if model_path is not None else ARTIFACTS_DIR / "guardian_spider_native.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}. Please train first.")
        
        try:
            # ✅ Load with map_location to control device placement
            state = torch.load(str(model_path), map_location="cpu", weights_only=False)
            mdl.load_state_dict(state)
        except Exception as e:
            raise RuntimeError(f"Failed to load model state from {model_path}: {e}")

        # ✅ CRITICAL: Force fp32 before moving to device
        mdl = mdl.float()
        
        mdl.to(device)
        mdl.eval()
        
        # ✅ CRITICAL: Enable inference mode to bypass unstable adaptation
        mdl.set_inference_mode(True)
        
        # ✅ CRITICAL: Reset CfC state buffer with correct dtype
        mdl.ganglia.last_liquid_state = mdl.ganglia.last_liquid_state.float().zero_()
        
        _RESOURCES["model"] = mdl
        print("✅ Model ready for inference (inference_mode=ON, fp32=ON, CfC state reset)")

    # Temperatures
    if _RESOURCES["temps"] is None:
        calib_path = ARTIFACTS_DIR / "calib_temp.json"
        if calib_path.exists():
            try:
                with open(calib_path, 'r', encoding='utf-8') as f:
                    calib_data = json.load(f)
                    _RESOURCES["temps"] = calib_data["temperatures"]
                    _RESOURCES["global_temp"] = calib_data.get("global_temperature", 1.0)
                print(f"✅ Calibration loaded: {_RESOURCES['temps']}")
            except Exception as e:
                print(f"⚠️  Failed to load calibration: {e}, using defaults")
                _RESOURCES["temps"] = {"math": 1.0, "code": 1.0, "real_world": 1.0}
                _RESOURCES["global_temp"] = 1.0
        else:
            print("⚠️  No calibration file found, using T=1.0")
            _RESOURCES["temps"] = {"math": 1.0, "code": 1.0, "real_world": 1.0}
            _RESOURCES["global_temp"] = 1.0

def diagnose_sample(question: str, answer: str) -> Tuple[str, float, str, Dict[str, Any]]:
    """
    Takes a Q&A pair and returns (status, confidence, domain_name, diagnostics).
    Uses confidence-aware temperature scaling and resets CfC state per sample.
    ✅ RETURNS full diagnostics dictionary for biomarker analysis.
    ✅ FORCES fp32 dtype throughout to prevent Half/Float mismatch.
    ✅ FIXED: PyTorch 2.0+ AMP API compatibility.
    """
    # Ensure resources are loaded
    device = _get_device()
    load_all_resources(device=device)

    extractor: FeatureExtractor = _RESOURCES["extractor"]
    model: torch.nn.Module = _RESOURCES["model"]
    temps: Dict[str, float] = _RESOURCES["temps"]
    global_temp: float = _RESOURCES["global_temp"]

    # ✅ Reset CfC state before each sample to prevent contamination (ensure fp32)
    model.ganglia.last_liquid_state = model.ganglia.last_liquid_state.float().zero_()

    # 1. Feature Extraction
    try:
        feat_dict = extractor.extract([question], [answer])
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {e}")
    
    # ✅ Ensure all hidden states are fp32
    hidden_states = [feat_dict[i].float() for i in range(len(feat_dict))]

    # 2. Model Inference with confidence-aware temperature
    # ✅ FIXED: Use torch.amp.autocast for PyTorch 2.0+ compatibility
    autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device, enabled=False):
        try:
            logits, routing_info = model(hidden_states)
        except Exception as e:
            raise RuntimeError(f"Model forward pass failed: {e}")
        
        # ✅ EXTRACT DOMAIN AND BIOMARKERS
        if isinstance(routing_info, dict):
            domain = routing_info.get('primary_domain', 'math')
            domain_confidence = routing_info.get('domain_confidence', 0.0)
            liquid_norm = routing_info.get('ganglia_liquid_norm', 0.1)
            routing_entropy = routing_info.get('routing_entropy', 0.0)
            panic_factor = routing_info.get('panic_factor', 0.0)
            vibration = routing_info.get('vibration', 0.0)
        else:
            domain = 'math'
            domain_confidence = 0.0
            liquid_norm = 0.1
            routing_entropy = 0.0
            panic_factor = 0.0
            vibration = 0.0
        
        # ✅ TEMPERATURE SELECTION: Domain if confident, else global
        CONFIDENCE_THRESHOLD = 0.65
        if domain_confidence >= CONFIDENCE_THRESHOLD:
            temp_to_use = temps.get(domain, global_temp)
        else:
            temp_to_use = global_temp  # Fallback
        
        # ✅ APPLY SCALING (ensure fp32)
        scaled_logits = logits.float() / temp_to_use
        
        # Compute probabilities
        probs = torch.softmax(scaled_logits, dim=-1).cpu().numpy().flatten()
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])
        
        # ✅ COMPREHENSIVE DIAGNOSTICS
        diagnostics = {
            "domain": domain,
            "domain_confidence": float(domain_confidence),
            "liquid_state_norm": float(liquid_norm),
            "routing_entropy": float(routing_entropy),
            "panic_factor": float(panic_factor),
            "vibration": float(vibration.mean() if torch.is_tensor(vibration) else vibration),
            "temperature_used": float(temp_to_use),
            "global_temperature": float(global_temp),
            "raw_logits": logits.cpu().float().numpy().flatten().tolist(),
            "scaled_logits": scaled_logits.cpu().float().numpy().flatten().tolist(),
            "all_probs": probs.tolist()
        }
        
    # 3. Status mapping
    is_hallu = (prediction == 1)
    status = "HALLUCINATION" if is_hallu else "VALID"

    return status, confidence, domain, diagnostics

def batch_diagnose(questions: List[str], answers: List[str]) -> List[Tuple[str, float, str, Dict[str, Any]]]:
    """Process multiple Q&A pairs efficiently with per-sample CfC state reset"""
    results = []
    for q, a in zip(questions, answers):
        try:
            status, conf, domain, diags = diagnose_sample(q, a)
            results.append((status, conf, domain, diags))
        except Exception as e:
            print(f"[ERROR] Failed on Q='{q[:50]}...': {e}")
            results.append(("ERROR", 0.0, "unknown", {"error": str(e)}))
    return results

def run_batch_inference(questions: List[str], answers: List[str], batch_size: int = 8) -> List[Dict[str, Any]]:
    """
    Optimized batch inference with per-sample routing and biomarker collection.
    ✅ CORRECTLY HANDLES per-sample diagnostics in batched mode.
    ✅ RESETS CfC state between batches.
    ✅ FORCES fp32 dtype throughout.
    ✅ FIXED: PyTorch 2.0+ AMP API compatibility.
    """
    if len(questions) != len(answers):
        raise ValueError("Mismatched Q/A lengths")
    
    device = _get_device()
    load_all_resources(device=device)
    
    extractor: FeatureExtractor = _RESOURCES["extractor"]
    model: torch.nn.Module = _RESOURCES["model"]
    temps: Dict[str, float] = _RESOURCES["temps"]
    global_temp: float = _RESOURCES["global_temp"]
    
    all_results = []
    
    from tqdm import tqdm  # Import here to avoid dependency issues
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Batch inference", unit="batch"):
        batch_q = questions[i:i+batch_size]
        batch_a = answers[i:i+batch_size]
        
        # ✅ Reset CfC state buffer before each batch (ensure fp32)
        model.ganglia.last_liquid_state = model.ganglia.last_liquid_state.float().zero_()
        
        # Extract features
        feat_dict = extractor.extract(batch_q, batch_a)
        # ✅ Ensure fp32
        hidden_states = [feat_dict[j].float() for j in range(len(feat_dict))]
        
        # ✅ FIXED: Use torch.amp.autocast for PyTorch 2.0+ compatibility
        autocast_device = 'cuda' if device.type == 'cuda' else 'cpu'
        with torch.no_grad(), torch.amp.autocast(device_type=autocast_device, enabled=False):
            logits, routing_infos = model(hidden_states)
            
            # Process each sample
            for j in range(len(batch_q)):
                # ✅ EXTRACT per-sample routing info
                if isinstance(routing_infos, dict):
                    # Single routing info for batch—use same for all
                    domain = routing_infos.get('primary_domain', 'math')
                    domain_confidence = routing_infos.get('domain_confidence', 0.0)
                    routing_entropy = routing_infos.get('routing_entropy', 0.0)
                    panic_factor = routing_infos.get('panic_factor', 0.0)
                    vibration = routing_infos.get('vibration', 0.0)
                    router_probs = routing_infos.get('train_routing_probs')[j]  # Per-sample
                elif isinstance(routing_infos, list) and j < len(routing_infos):
                    rinfo = routing_infos[j]
                    domain = rinfo.get('primary_domain', 'math')
                    domain_confidence = rinfo.get('domain_confidence', 0.0)
                    routing_entropy = rinfo.get('routing_entropy', 0.0)
                    panic_factor = rinfo.get('panic_factor', 0.0)
                    vibration = rinfo.get('vibration', 0.0)
                    router_probs = rinfo.get('train_routing_probs')
                else:
                    domain = 'math'
                    domain_confidence = 0.0
                    routing_entropy = 0.0
                    panic_factor = 0.0
                    vibration = 0.0
                    router_probs = torch.ones(3, device=device) / 3
                
                # Temperature selection
                CONFIDENCE_THRESHOLD = 0.65
                if domain_confidence >= CONFIDENCE_THRESHOLD:
                    temp_to_use = temps.get(domain, global_temp)
                else:
                    temp_to_use = global_temp
                
                # Apply temperature (ensure fp32)
                sample_logits = logits[j:j+1].float() / temp_to_use
                probs = torch.softmax(sample_logits, dim=-1).cpu().numpy().flatten()
                prediction = int(np.argmax(probs))
                confidence = float(probs[prediction])
                
                diagnostics = {
                    "domain": domain,
                    "domain_confidence": float(domain_confidence),
                    "routing_entropy": float(routing_entropy),
                    "panic_factor": float(panic_factor),
                    "vibration": float(vibration.mean() if torch.is_tensor(vibration) else vibration),
                    "temperature_used": float(temp_to_use),
                    "router_probs": router_probs.cpu().float().numpy().tolist() if torch.is_tensor(router_probs) else router_probs,
                    "all_probs": probs.tolist()
                }
                
                status = "HALLUCINATION" if prediction == 1 else "VALID"
                
                all_results.append({
                    "question": batch_q[j],
                    "answer": batch_a[j],
                    "status": status,
                    "confidence": confidence,
                    "domain": domain,
                    "diagnostics": diagnostics
                })
        
        # Cleanup
        del hidden_states, logits, routing_infos
        torch.cuda.empty_cache()
    
    return all_results

def print_biomarkers(diagnostics: Dict[str, Any]):
    """Pretty-print biomarker values for manual inspection"""
    print(f"  └─ Biomarkers:")
    print(f"     • Domain: {diagnostics['domain']} (conf: {diagnostics['domain_confidence']:.2f})")
    print(f"     • Entropy: {diagnostics['routing_entropy']:.3f} (low=specialization)")
    print(f"     • Panic: {diagnostics['panic_factor']:.3f} (high=suppression)")
    print(f"     • Vibration: {diagnostics['vibration']:.3f} (high=uncertainty)")

# ==================== MAIN CLI ====================
if __name__ == "__main__":
    # Demo with biomarker visualization
    try:
        load_all_resources()
    except Exception as e:
        print(f"[ERROR] Resource init failed: {e}")
        raise SystemExit(1)

    test_cases = [
        ("What is 12 * 7?", "84"),
        ("What is 12 * 7?", "74"),
        ("Write a Python function to add two numbers", "def add(a, b): return a + b"),
        ("Write a Python function to add two numbers", "def add(a, b): return a * b"),
        ("Is the moon made of cheese?", "Yes, it's a dairy product"),
        ("Calculate eigenvalues of [[1,2],[3,4]]", "The eigenvalues are -0.372 and 5.372"),
    ]
    
    print(f"\n🕷️ Spider-Triangulation Inference (Biomimetic v15 - Dtype Safe)")
    print(f"{'='*80}")
    print(f"Note: Routing entropy <0.5 = healthy specialization | Panic >0.7 = stable")
    print(f"      Dtype: Forced FP32 (Unsloth patches disabled)")
    print(f"{'='*80}\n")
    
    for q, a in test_cases:
        print(f"\nQ: {q}")
        print(f"A: {a}")
        try:
            status, conf, domain, diags = diagnose_sample(q, a)
            print(f"→ Status: {status:12} | Confidence: {conf:.2%} | Domain: {domain}")
            print_biomarkers(diags)
        except Exception as e:
            print(f"→ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Batch inference demo
    print(f"\n{'='*80}")
    print(f"Batch Inference Demo (4 samples)")
    print(f"{'='*80}\n")
    
    batch_results = run_batch_inference(
        [q for q, _ in test_cases[:4]],
        [a for _, a in test_cases[:4]],
        batch_size=2
    )
    
    for res in batch_results:
        print(f"Domain: {res['domain']:8} | Status: {res['status']:12} | Conf: {res['confidence']:.2%}")
        print_biomarkers(res['diagnostics'])
        print("-" * 60)