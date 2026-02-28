# core/guardian_utils.py (v8.0 - Spider-Triangulation Utilities)
"""
Spider-Native Utility Classes:
- FeatureExtractor updated to return 8-leg triangulation format
- GuardianScaler handles variable-dimensional flattened features
- FocalLoss preserved for class imbalance
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from typing import List, Dict, Any

# ==================== WINDOWS TRITON PATCH (MUST RUN FIRST) ====================
# CRITICAL: This patch MUST be applied before importing Unsloth
if os.name == 'nt':  # Windows only
    # Fix the NUL path error
    triton_cache = os.path.join(os.path.expanduser("~"), ".triton_cache")
    os.makedirs(triton_cache, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache
    
    # Disable problematic Unsloth kernels that crash on Windows
    os.environ["UNSLOTH_DISABLE_RMS_LN"] = "1"
    os.environ["UNSLOTH_DISABLE_KERNELS"] = "1"

# Monkey patch Unsloth's RMS LayerNorm kernel (source of FileNotFoundError)
def _patch_unsloth_kernels():
    try:
        import unsloth.kernels.rms_layernorm
        
        def safe_rms_layernorm(X, W, eps, gemma=None):
            """Pure PyTorch RMS LayerNorm replacement for Windows"""
            variance = X.float().pow(2).mean(-1, keepdim=True)
            X = X * torch.rsqrt(variance + eps)
            if W is not None:
                X = X * W
            return X.type_as(X)
        
        unsloth.kernels.rms_layernorm.fast_rms_layernorm = safe_rms_layernorm
        print("✅ SUCCESS: Unsloth RMS kernel patched for Windows compatibility")
        return True
    except Exception as e:
        print(f"⚠️  WARNING: Could not patch Unsloth kernels: {e}")
        print("   This may cause the Triton cache error to persist.")
        return False

# Apply patch immediately
_patch_unsloth_kernels()

# Now it's safe to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("❌ ERROR: Unsloth not installed. Run: pip install unsloth[cu118-torch21]")
    UNSLOTH_AVAILABLE = False


# ==================== GUARDIAN SCALER (Float64-Safe) ====================
class GuardianScaler:
    """
    Float64-safe scaler to prevent overflow during mean/std computation.
    Robust with small-variance guard.
    Now handles variable-dimensional flattened features from any stream.
    """
    def __init__(self):
        self.mean: torch.Tensor | None = None
        self.scale: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> None:
        """
        Fit on X shaped (N, D) or any flattened feature array.
        Uses float64 for stability.
        """
        X64 = X.double()
        if X64.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X64.shape}")

        self.mean = X64.mean(dim=0)
        self.scale = X64.std(dim=0)

        # Guard against near-zero variance (biological: neurons have baseline noise)
        # Scale minimum = 1e-8 to prevent division by near-zero
        min_scale = 1e-8
        self.scale = torch.clamp(self.scale, min=min_scale)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.scale is None:
            raise RuntimeError("Scaler must be fitted before transform.")

        X64 = X.double()
        normalized = (X64 - self.mean) / self.scale
        return normalized.float()  # Return float32 for model compatibility

    def save(self, path: str | os.PathLike):
        """Save mean and scale to pickle file"""
        data = {
            'mean': self.mean.cpu().numpy() if self.mean is not None else None,
            'scale': self.scale.cpu().numpy() if self.scale is not None else None
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str | os.PathLike):
        """Load mean and scale from pickle file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.mean = torch.from_numpy(data['mean'])
            self.scale = torch.from_numpy(data['scale'])


# ==================== FEATURE EXTRACTOR (Spider-Triangulation) ====================
class FeatureExtractor:
    """
    Extracts 8-leg triangulation features from Qwen2.5-1.5B.
    Splits sequence into 8 legs of 16 tokens each and averages within each leg.
    """
    def __init__(self,
                 model_name: str = "unsloth/Qwen2.5-1.5B",
                 max_seq_length: int = 128,
                 load_in_4bit: bool = False):
        
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is not available.")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seq_length = max_seq_length
        self.num_legs = 8
        self.chunk_size = self.seq_length // self.num_legs 
        self.n_layers = 29  # 28 layers + 1 embedding layer
        
        print(f"🔄 Loading {model_name} in FP16 mode on {self.device}...")

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                load_in_4bit=False,
                torch_dtype=torch.float16,
                device_map=self.device
            )
            FastLanguageModel.for_inference(self.model)
            print("✅ SUCCESS: FP16 model loaded (3GB VRAM, higher precision)")
            
        except Exception as e:
            print(f"❌ FAILED to load FP16 model: {e}")
            print("   Attempting fallback to standard Transformers...")
            
            from transformers import AutoTokenizer as FallbackTokenizer
            from transformers import AutoModelForCausalLM as FallbackModel
            
            self.tokenizer = FallbackTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = FallbackModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            self.device = "cpu"
            print("⚠️  FALLBACK: Using CPU FP16 model")
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            for param in self.model.parameters():
                param.requires_grad = False

    def extract(self, questions: List[str], answers: List[str]) -> Dict[int, torch.Tensor]:
        if len(questions) != len(answers):
            raise ValueError("Questions and answers must have the same length")
            
        texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.seq_length,
            return_attention_mask=True
        ).to(self.device)
        
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        hidden_states = {}
        for i, layer_tensor in enumerate(outputs.hidden_states):
            batch_size = layer_tensor.shape[0]
            # Reshape to (B, 8, 16, 1536)
            reshaped = layer_tensor.view(batch_size, self.num_legs, self.chunk_size, -1)
            # Average pool to (B, 8, 1536)
            leg_pooled = reshaped.mean(dim=2)
            hidden_states[i] = leg_pooled

        return hidden_states

# ==================== FOCAL LOSS (Hard Example Mining) ====================
class FocalLoss(nn.Module):
    """
    Focal loss for class imbalance and hard example mining.
    Gamma=5.0 puts strong focus on difficult samples (hallucinations).
    Alpha=0.75 weights hallucination class (label=1) more heavily.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 5.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices (0=valid, 1=hallucination)
        """
        # Standard cross-entropy (no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        
        # Probability of correct class
        pt = torch.exp(-ce_loss)
        
        # Focal modulation: down-weight easy examples
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal