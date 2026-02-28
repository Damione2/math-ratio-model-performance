# core/guardian_calib.py (v9.3 - Production-Ready Calibration with Compile Fix)
"""
Spider-Native Temperature Calibration:
- Calibrates per-domain temperatures using domain confidence weighting
- Integrates with SpiderAdaptation's variance tracking
- Handles 8-leg triangulation features: (N, 29, 8, 1536)
- Full torch.compile() wrapper compatibility
- Confidence-weighted optimization focuses on high-certainty samples
"""

import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pickle
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import time
from tqdm import tqdm

from .guardian_vision_core import GuardianVisionNet
# Use centralized config
try:
    from config import ARTIFACTS_DIR
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import ARTIFACTS_DIR

# ==================== DATASET FOR CALIBRATION ====================
class CalibrationDataset(torch.utils.data.Dataset):
    """
    Load memmap and metadata for calibration.
    Features shape: (N, 29, 8, 1536) - Spider-Triangulation format
    Metadata: list of dicts with 'label' and 'domain'
    """
    def __init__(self, features_file: Path, metadata_file: Path):
        """
        Args:
            features_file: Path to memmap file (.bin) with shape (N, 29, 8, 1536)
            metadata_file: Path to pickle file with metadata list
        """
        if not features_file.exists():
            raise FileNotFoundError(f"Features file not found: {features_file}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load metadata first to get number of samples
        with open(metadata_file, "rb") as f:
            self.metadata = pickle.load(f)
        
        # Open memmap: (N, 29, 8, 1536)
        self.features = np.memmap(
            features_file, 
            dtype='float16', 
            mode='r', 
            shape=(len(self.metadata), 29, 8, 1536)
        )
        
        # FIX: Create domains array for final stats
        self.domains = np.array([self._domain_to_int(m.get('domain', 'real_world')) 
                                 for m in self.metadata], dtype=np.int64)
        
        print(f"📊 Calibration dataset loaded: {len(self.metadata)} samples | Shape: {self.features.shape}")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. Load features: (29, 8, 1536)
        features = torch.from_numpy(np.array(self.features[idx], copy=True)).float()
        
        # 2. Convert to list format for GuardianVisionNet: 29 tensors × (8, 1536)
        hidden_states = [features[i, :, :] for i in range(features.shape[0])]
        
        # 3. Extract metadata
        meta = self.metadata[idx]
        
        return {
            'hidden_states': hidden_states,  # List format for model compatibility
            'label': torch.tensor(meta['label'], dtype=torch.long),
            'domain': torch.tensor(self._domain_to_int(meta['domain']), dtype=torch.long)
        }
    
    def _domain_to_int(self, domain_str: str) -> int:
        """Convert domain string to integer index"""
        domain_map = {"math": 0, "code": 1, "real_world": 2, "real": 2}
        return domain_map.get(domain_str, 2)

# ==================== TEMPERATURE CALIBRATION ====================
class TemperatureCalibrator:
    def __init__(self, model: nn.Module, device: str = "cuda", lr: float = 0.01, max_iter: int = 50):
        self.model = model
        self.device = device
        self.lr = lr
        self.max_iter = max_iter
        
    def collect_validation_outputs(self, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Run model on validation set and collect logits per domain WITH routing info.
        
        FIXED: Collects routing_info alongside logits for confidence weighting
        FIXED: Memory leak prevention with aggressive cleanup
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        all_domains = []
        all_routing_infos = []  # ✅ NEW: Store routing diagnostics
        
        print("⏳ Collecting validation logits and routing info for calibration...")
        
        with tqdm(total=len(val_loader), desc="Collecting outputs", unit="batch") as pbar:
            for batch in val_loader:
                hidden_states = batch['hidden_states']  # List format
                labels = batch['label']
                domains = batch['domain']
                
                # ✅ FIX: Move data to device (prevents RuntimeError)
                hidden_states = [hs.to(self.device, non_blocking=True) for hs in hidden_states]
                labels = labels.to(self.device, non_blocking=True)
                domains = domains.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.no_grad():
                    logits, routing_info = self.model(hidden_states)
                
                # ✅ FIXED: Expand batch-level routing to sample-level
                batch_size = logits.shape[0]
                for i in range(batch_size):
                    all_routing_infos.append(routing_info)
                
                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_domains.append(domains.detach().cpu())
                
                # ✅ AGGRESSIVE cleanup to prevent memory leaks
                del hidden_states, labels, domains, logits, routing_info, batch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
                
                pbar.update(1)
        
        # Concatenate on CPU (prevents GPU memory bloat)
        return {
            'logits': torch.cat(all_logits, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'domains': torch.cat(all_domains, dim=0),
            'routing_infos': all_routing_infos  # ✅ Now sample-level
        }
    
    def optimize_temperature(self, domain_logits: torch.Tensor, 
                           domain_labels: torch.Tensor,
                           domain_name: str,
                           routing_infos: list) -> float:
        """
        Optimize temperature for a single domain using LBFGS with CONFIDENCE WEIGHTING.
        
        ✅ NEW: Uses routing_info to weight high-confidence samples more heavily
        ✅ NEW: Implements per-sample loss weighting
        """
        # ✅ EXTRACT CONFIDENCE WEIGHTS
        confidence_weights = []
        for routing_info in routing_infos:
            if isinstance(routing_info, dict) and routing_info.get('primary_domain') == domain_name:
                conf = routing_info.get('domain_confidence', 0.5)
                confidence_weights.append(max(0.1, conf))  # Min weight 0.1
            else:
                confidence_weights.append(0.5)  # Default weight
        
        weights = torch.tensor(confidence_weights, dtype=torch.float32, device=self.device)
        
        log_temp = nn.Parameter(torch.log(torch.tensor(1.5, device=self.device)))
        optimizer = optim.LBFGS([log_temp], lr=self.lr, max_iter=self.max_iter)
        criterion = nn.CrossEntropyLoss(reduction='none')  # ✅ Per-sample loss
        
        logits = domain_logits.to(self.device)
        labels = domain_labels.to(self.device)
        
        def closure():
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            scaled_logits = logits / temp.clamp(min=0.1, max=10.0)
            losses = criterion(scaled_logits, labels)
            
            # ✅ WEIGHTED LOSS: High confidence = more influence
            loss = (losses * weights).mean()
            loss.backward()
            return loss
        
        try:
            optimizer.step(closure)
        except Exception as e:
            print(f"⚠️  Optimization failed for {domain_name}: {e}")
            return 1.0
        
        final_temp = float(torch.exp(log_temp).item())
        
        # Clamp to reasonable range
        if not (0.1 <= final_temp <= 10.0):
            print(f"⚠️  Temperature {final_temp} outside range for {domain_name}, using 1.0")
            return 1.0
        
        return final_temp
    
    def calibrate_all_domains(self, val_loader: DataLoader) -> tuple[Dict[str, float], float]:
        """
        Calibrate temperatures for all three domains with confidence weighting.
        Returns dict mapping domain name → temperature.
        """
        print(f"\n🔧 Calibrating per-domain temperatures with confidence weighting...")
        print(f"   This will weight high-confidence samples more heavily in optimization.")
        
        outputs = self.collect_validation_outputs(val_loader)
        
        domain_map = {0: "math", 1: "code", 2: "real_world"}
        temperatures = {}
        
        # ✅ FIXED: routing_infos is now sample-level, can directly index
        domain_routing_infos = {name: [] for name in domain_map.values()}
        for idx, domain_idx in enumerate(outputs['domains'].cpu().numpy()):
            domain_name = domain_map[int(domain_idx)]
            domain_routing_infos[domain_name].append(outputs['routing_infos'][idx])
        
        for domain_idx, domain_name in domain_map.items():
            mask = (outputs['domains'] == domain_idx)
            n_samples = int(mask.sum())
            
            if n_samples < 10:
                print(f"  ⚠️  {domain_name}: only {n_samples} samples, using default 1.0")
                temperatures[domain_name] = 1.0
                continue
            
            domain_logits = outputs['logits'][mask]
            domain_labels = outputs['labels'][mask]
            domain_routing = domain_routing_infos[domain_name]
            
            print(f"  ⏳ {domain_name}: {n_samples} samples (confidence-weighted)")
            
            temp = self.optimize_temperature(
                domain_logits, domain_labels, domain_name, domain_routing
            )
            temperatures[domain_name] = temp
            
            print(f"  ✅ {domain_name}: temperature = {temp:.4f}")
        
        # Global temperature (unweighted for baseline)
        global_temp = self.optimize_temperature(
            outputs['logits'], outputs['labels'], "global", 
            [None] * len(outputs['labels'])  # No weights for global
        )
        print(f"  🌍 global: temperature = {global_temp:.4f}")
        
        return temperatures, global_temp

# ==================== MAIN CALIBRATION FUNCTION ====================
def run_calibration(model_path: Optional[Path] = None, 
                    val_files: Optional[Dict[str, Path]] = None,
                    resume: bool = True,
                    checkpoint_path: Optional[Path] = None):
    """
    Main calibration entry point.
    Loads model and validation data, runs temperature optimization.
    
    Args:
        model_path: Path to trained model weights (.pth)
        val_files: Dict with keys 'features' and 'metadata' pointing to validation files
        resume: Whether to resume from existing checkpoint (default: True)
        checkpoint_path: Path to full checkpoint (for torch.compile wrapper)
    """
    # Default paths matching Spider-Triangulation pipeline
    if model_path is None:
        model_path = ARTIFACTS_DIR / "guardian_spider_native.pth"
    
    if val_files is None:
        val_files = {
            'features': ARTIFACTS_DIR / "val_features.bin",
            'metadata': ARTIFACTS_DIR / "val_metadata.pkl"
        }
    
    # Validate files exist
    for name, path in val_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Validation {name} not found: {path}")
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GuardianVisionNet().to(device)
    
    # FIX: Use the passed resume parameter
    # FIX: Handle torch.compile wrapper
    state_dict_to_load = None
    
    # Try checkpoint first if provided
    if checkpoint_path and checkpoint_path.exists():
        print(f"📂 Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict_to_load = checkpoint['model_state_dict']
    
    # Fall back to regular model path
    elif resume and model_path.exists():
        print(f"📂 Loading model from {model_path}")
        state_dict_to_load = torch.load(model_path, map_location=device, weights_only=False)
    else:
        print(f"⚠️  No checkpoint found, using fresh model")
    
    # Apply state dict if we loaded one
    if state_dict_to_load is not None:
        # FIX: Strip _orig_mod prefix from compiled model
        if any(key.startswith('_orig_mod.') for key in state_dict_to_load.keys()):
            print("🔧 Stripping _orig_mod prefix from compiled model state dict...")
            new_state_dict = {}
            for key, value in state_dict_to_load.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            state_dict_to_load = new_state_dict
        
        model.load_state_dict(state_dict_to_load)
        print("✅ Model loaded successfully")
    
    # ✅ SET INFERENCE MODE for stable calibration
    model.set_inference_mode(True)
    model.eval()
    
    # Load validation dataset
    val_dataset = CalibrationDataset(
        val_files['features'],
        val_files['metadata']
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=0
    )
    
    print(f"📊 Loaded validation set: {len(val_dataset)} samples")
    
    # Run calibration
    calibrator = TemperatureCalibrator(model, device=device)
    temperatures, global_temp = calibrator.calibrate_all_domains(val_loader)
    
    # Save results
    calibration_data = {
        "temperatures": temperatures,
        "global_temperature": global_temp,
        "calibration_timestamp": time.time(),
        "model_path": str(model_path),
        "n_samples": len(val_dataset),
        "domain_distribution": {
            "math": int((val_dataset.domains == 0).sum()),
            "code": int((val_dataset.domains == 1).sum()),
            "real_world": int((val_dataset.domains == 2).sum())
        },
        "confidence_weighted": True,  # ✅ Flag for new method
        "notes": "High-confidence samples weighted more heavily in optimization"
    }
    
    output_path = ARTIFACTS_DIR / "calib_temp.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(calibration_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Calibration complete!")
    print(f"   Per-domain temperatures: {temperatures}")
    print(f"   Global temperature: {global_temp:.4f}")
    print(f"   Saved to: {output_path}")
    
    return calibration_data

# ==================== COMMAND-LINE INTERFACE ====================
if __name__ == "__main__":
    """
    Run calibration from command line:
    python guardian_calib.py --model path/to/model.pth --val_features path/to/val_features.bin --val_metadata path/to/val_metadata.pkl
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate GuardianVisionNet temperatures for Spider-Triangulation")
    parser.add_argument("--model", type=Path, help="Path to model weights (.pth)")
    parser.add_argument("--val_features", type=Path, help="Path to validation features (.bin)")
    parser.add_argument("--val_metadata", type=Path, help="Path to validation metadata (.pkl)")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from checkpoint if exists")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Start calibration fresh")
    
    args = parser.parse_args()
    
    # Build file paths dict
    val_files = None
    if args.val_features and args.val_metadata:
        val_files = {
            'features': args.val_features,
            'metadata': args.val_metadata
        }
    
    # Run calibration
    run_calibration(
        model_path=args.model,
        val_files=val_files,
        resume=args.resume,
        checkpoint_path=ARTIFACTS_DIR / "guardian_spider_native_full.pth"
    )