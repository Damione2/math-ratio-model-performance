# core/guardian_vision_core.py (v13.0 - Biomimetic Fusion with Panic Gate)
"""
Spider-Triangulation Hallucination Detection System
✅ FIXED: Liquid CfC experts replace GRUs for 8-leg temporal processing
✅ FIXED: Gated fusion eliminates domain interference (math ≠ code pollution)
✅ NEW: Panic gate suppresses uncertain samples via vibration modulation
✅ NEW: Router pretraining support via exposed train_routing_probs
✅ FIXED: State preservation buffer prevents CfC state leakage between batches
✅ FIXED: Gradient detachment stops "backward through graph twice" error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# ==================== SPIDER EYES (Multi-Stream Sensory) ====================
class SpiderEyes(nn.Module):
    """Multi-stream sensory extraction from 29 layers"""
    def __init__(self):
        super().__init__()
        self.principal_layers = [20, 21, 22, 23, 24]
        self.motion_layers = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        self.texture_layers = [16, 17, 18, 19]
        self.nocturnal_layers = [0, 1, 2, 3, 4]

    def extract_stream(self, hidden_states, layer_indices):
        """Extract specific layers while preserving leg dimension (B, L, 8, H)"""
        if isinstance(hidden_states, torch.Tensor):
            selected = hidden_states[:, layer_indices, :, :]
        else:
            selected = torch.stack([hidden_states[i] for i in layer_indices], dim=1)
        return selected

    def forward(self, hidden_states):
        return {
            'principal': self.extract_stream(hidden_states, self.principal_layers),
            'motion': self.extract_stream(hidden_states, self.motion_layers),
            'texture': self.extract_stream(hidden_states, self.texture_layers),
            'nocturnal': self.extract_stream(hidden_states, self.nocturnal_layers)
        }

# ==================== OPTIC NERVE (Parallel Compression) ====================
class SpiderOpticNerve(nn.Module):
    """Compresses each sensory stream with dedicated bandwidth"""
    def __init__(self):
        super().__init__()
        self.compressors = nn.ModuleDict({
            'principal': nn.Linear(1536, 64),
            'motion': nn.Linear(1536, 32),
            'texture': nn.Linear(1536, 32),
            'nocturnal': nn.Linear(1536, 16)
        })
        self.layer_norms = nn.ModuleDict({
            name: nn.LayerNorm(dim) for name, dim in zip(
                ['principal', 'motion', 'texture', 'nocturnal'],
                [64, 32, 32, 16]
            )
        })

    def forward(self, streams: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        compressed = {}
        for name, features in streams.items():
            B, L, Legs, H = features.shape
            flat = features.reshape(B * L * Legs, H)
            projected = F.gelu(self.compressors[name](flat))
            compressed[name] = self.layer_norms[name](projected).view(B, L, Legs, -1)
        return compressed

# ==================== LATERAL INHIBITION (DoG Filtering) ====================
class LateralInhibition(nn.Module):
    """Biological center-surround (Difference-of-Gaussians)"""
    def __init__(self, kernel_size: int = 5, sigma_center: float = 0.8, 
                 sigma_surround: float = 1.6, noise_scale: float = 0.01):
        super().__init__()
        self.kernel_size = kernel_size
        self.noise_scale = noise_scale
        
        half = kernel_size // 2
        xs = torch.arange(-half, half + 1, dtype=torch.float32)
        center = torch.exp(-0.5 * (xs / sigma_center) ** 2)
        surround = torch.exp(-0.5 * (xs / sigma_surround) ** 2)
        dog = center - surround
        self.register_buffer("dog_kernel", dog / dog.abs().sum())

    def forward(self, compressed_streams: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inhibited = {}
        for name, features in compressed_streams.items():
            B, L, Legs, H = features.shape
            x_t = features.permute(0, 2, 3, 1).contiguous().view(B * Legs * H, 1, L)
            
            out = F.conv1d(x_t, self.dog_kernel.view(1, 1, -1), padding=self.kernel_size // 2)
            cs = out.view(B, Legs, H, L).permute(0, 3, 1, 2).contiguous()
            cs_normalized = cs - 0.3 * cs.mean(dim=1, keepdim=True)
            
            activated = F.relu(cs_normalized)
            inhibited[name] = activated
            if self.noise_scale > 0:
                inhibited[name] = activated + torch.randn_like(activated) * self.noise_scale
        
        return inhibited

# ==================== CFC GANGLIA (Per-Sample Liquid Routing) ====================
class SpiderGangliaCfC(nn.Module):
    """
    ✅ FIXED: Per-sample CfC-based ganglia with state preservation.
    ✅ ENHANCED: 256-neuron capacity for multi-domain separation.
    Each sample gets its own routing decision with stabilized temporal dynamics.
    """
    def __init__(self, input_dim=288):
        super().__init__()
        # ✅ 256-neuron capacity for domain separation
        self.cfc = CfCNet(input_size=input_dim, hidden_size=256, num_layers=1, mode="no_gate")
        self.policy_head = nn.Sequential(
            nn.Linear(256, 48), nn.Tanh(), nn.Dropout(0.1), nn.Linear(48, 3)
        )
        self.register_buffer('liquid_state', torch.zeros(1, 1, 256))
        self.register_buffer('routing_temp', torch.tensor(1.5, dtype=torch.float32))
        
        # ✅ State preservation buffer for batch-to-batch stability
        self.register_buffer('last_liquid_state', torch.zeros(1, 1, 256))

    def forward(self, inhibited_streams: Dict[str, torch.Tensor], 
                vibration: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        
        # Per-sample feature extraction
        features = []
        energy_stats = {}
        
        for name, stream in inhibited_streams.items():
            B, L, Legs, H = stream.shape
            
            pooled = torch.mean(stream, dim=(1, 2))
            energy_std = torch.std(stream, dim=(1, 2))
            
            stream_features = torch.cat([pooled, energy_std], dim=-1)
            features.append(stream_features)
            
            energy_stats[f'{name}_energy'] = pooled.mean().item()
            energy_stats[f'{name}_std'] = energy_std.mean().item()
        
        energy_input = torch.cat(features, dim=-1).unsqueeze(1)
        
        batch_size = energy_input.shape[0]
        
        # ✅ Initialize from preserved state with momentum
        if self.last_liquid_state is not None:
            liquid_state = self.last_liquid_state.expand(-1, batch_size, -1)
        else:
            liquid_state = self.liquid_state.expand(-1, batch_size, -1)
        
        routing_features, new_state = self.cfc(energy_input, liquid_state)
        
        # ✅ Capture logits BEFORE softmax
        routing_logits = self.policy_head(routing_features)
        routing_logits = routing_logits.squeeze(1)
        
        # ✅ FIX: Aggressive clipping to prevent overflow
        routing_logits = torch.clamp(routing_logits, -5.0, 5.0)
        
        routing_probs = F.softmax(routing_logits / self.routing_temp, dim=-1)
        
        math_w = routing_probs[:, 0].view(batch_size, 1, 1, 1)
        code_w = routing_probs[:, 1].view(batch_size, 1, 1, 1)
        real_w = routing_probs[:, 2].view(batch_size, 1, 1, 1)
        
        routing = {}
        routing['math'] = inhibited_streams['principal'] * math_w
        routing['code'] = inhibited_streams['motion'] * code_w
        routing['real_world'] = inhibited_streams['nocturnal'] * real_w
        
        # ✅ CRITICAL FIX: Detach state update to prevent graph errors
        momentum = 0.95
        updated_state = momentum * self.last_liquid_state + (1 - momentum) * new_state.mean(dim=1, keepdim=True)
        self.last_liquid_state = torch.clamp(updated_state.detach(), -5, 5)
        
        # ✅ Routing entropy for collapse detection
        entropy = -torch.mean(torch.sum(routing_probs * torch.log(routing_probs + 1e-8), dim=-1))
        
        diagnostics = {
            **energy_stats,
            'routing_weights': {
                'math': routing_probs[:, 0].mean().item(),
                'code': routing_probs[:, 1].mean().item(),
                'real_world': routing_probs[:, 2].mean().item()
            },
            'liquid_state_norm': torch.norm(new_state).item(),
            'routing_entropy': float(entropy.item()),
            'train_routing_logits': routing_logits,
            'train_routing_probs': routing_probs
        }
        
        return routing, diagnostics

# ==================== SPIDER LIQUID EXPERT (Temporal Processing) ====================
class SpiderLiquidExpert(nn.Module):
    """
    ✅ NEW: CfC-based liquid expert processing 8 legs as temporal sequence.
    Each leg is a time-step, capturing coherence patterns that GRUs miss.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.cfc = CfCNet(input_size=input_dim, hidden_size=hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
    def forward(self, routed_input: torch.Tensor) -> torch.Tensor:
        """
        routed_input: (Batch, Layers, Legs, Dim) -> e.g., (B, 5, 8, 64) for math
        Returns: (Batch, 2) logits
        """
        # Average over layers: (B, Legs, Dim)
        per_leg = torch.mean(routed_input, dim=1)
        
        # Process legs as temporal sequence: (B, Seq=Legs, Dim)
        cfc_out, _ = self.cfc(per_leg)  # (B, 8, hidden_dim)
        
        # Use final leg's representation for decision
        final_state = cfc_out[:, -1, :]  # (B, hidden_dim)
        
        return self.classifier(final_state)

# ==================== TRICHOBOthria (Spectral Vibration) ====================
class RealVibrationHead(nn.Module):
    """Models spider's vibration-sensitive hairs (trichobothria)"""
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(96)
        self.spectral_classifier = nn.Sequential(
            nn.Linear(96, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.register_buffer('routing_temp', torch.tensor(0.5, dtype=torch.float32))

    def forward(self, streams: Dict[str, torch.Tensor]) -> torch.Tensor:
        principal = streams['principal'].mean(dim=1)
        motion = streams['motion'].mean(dim=1)
        
        principal_fft = torch.fft.rfft(principal, dim=1, norm='ortho')
        motion_fft = torch.fft.rfft(motion, dim=1, norm='ortho')
        
        principal_energy = principal_fft.abs().mean(dim=2)
        motion_energy = motion_fft.abs().mean(dim=2)
        
        def _pad_to(x, n_bins=48):
            B, F = x.shape
            if F >= n_bins:
                return x[:, :n_bins]
            pad = torch.zeros(B, n_bins - F, device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=1)
        
        spectral_energy = torch.cat([_pad_to(principal_energy), _pad_to(motion_energy)], dim=-1)
        spectral_energy = self.norm(spectral_energy)
        
        modulation = self.spectral_classifier(spectral_energy)
        spectral_diversity = torch.std(spectral_energy, dim=-1, keepdim=True)
        vibration = torch.sigmoid(modulation * spectral_diversity).squeeze(-1)
        
        return vibration

# ==================== CENTRAL BRAIN (Decision Fusion) ====================
class SpiderCentralBrain(nn.Module):
    """Subesophageal ganglion integrator with homeostatic scaling"""
    def forward(self, expert_outputs: Dict[str, torch.Tensor], 
                vibration_signal: torch.Tensor) -> torch.Tensor:
        # Stack expert logits: (3, Batch, 2) -> (Batch, 3, 2)
        logits_stack = torch.stack([
            expert_outputs['math'], expert_outputs['code'], expert_outputs['real_world']
        ], dim=1)
        
        # Clamp for stability
        logits_clamped = torch.clamp(logits_stack, min=-20.0, max=20.0)
        
        # Confidence weighting
        confidences = torch.softmax(logits_clamped, dim=-1)
        
        # Vibration modulation
        modulation = (1.0 - vibration_signal).view(-1, 1, 1)  # (Batch, 1, 1)
        weighted = confidences * modulation
        
        # Average across domains
        final_probs = torch.mean(weighted, dim=1)
        final_logits = torch.log(final_probs.clamp(min=1e-12))
        
        return final_logits

# ==================== HOMEOSTATIC ADAPTATION (BCM Theory) ====================
class SpiderAdaptation(nn.Module):
    """Bienenstock-Cooper-Munro homeostatic plasticity"""
    def __init__(self, alpha=0.1, bcm_boost_factor=1.1):
        super().__init__()
        self.base_temps = nn.Parameter(torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
        self.register_buffer('bcm_boost', torch.ones(3))
        self.register_buffer('running_mean', torch.zeros(3))
        self.register_buffer('running_var', torch.ones(3))
        self.register_buffer('recent_confidences', torch.zeros(100, 3))
        self.register_buffer('pointer', torch.tensor(0, dtype=torch.long))
        self.bcm_boost_factor = bcm_boost_factor
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, domain_idx: int) -> torch.Tensor:
        confidences = torch.softmax(logits, dim=-1).max(dim=-1).values.detach()
        confidences = torch.clamp(confidences, min=1e-6, max=1-1e-6)
        
        current_conf_mean = confidences.mean()
        self.recent_confidences[self.pointer.item(), domain_idx] = current_conf_mean
        self.pointer = (self.pointer + 1) % 100
        
        variance = 1e-8
        recent = self.recent_confidences[:min(self.pointer.item() + 1, 100), domain_idx]
        if len(recent) > 1:
            variance = max(float(torch.var(recent, unbiased=False)), 1e-8)
        
        if variance < 0.01:
            self.bcm_boost[domain_idx] = min(
                self.bcm_boost[domain_idx].item() * self.bcm_boost_factor,
                10.0
            )
        
        self.running_mean[domain_idx] = (1 - self.alpha) * self.running_mean[domain_idx] + self.alpha * current_conf_mean
        self.running_var[domain_idx] = (1 - self.alpha) * self.running_var[domain_idx] + self.alpha * variance
        
        adaptive_temp = self.base_temps[domain_idx] * self.bcm_boost[domain_idx] * (1.0 + self.running_var[domain_idx])
        return logits / adaptive_temp.clamp(min=0.1, max=10.0)

# ==================== MAIN GUARDIAN MODEL ====================
class GuardianVisionNet(nn.Module):
    """Complete Spider-Native hallucination detection model with biomimetic fusion"""
    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.spider_eyes = SpiderEyes()
        self.optic_nerve = SpiderOpticNerve()
        self.lateral_inhibition = LateralInhibition()
        self.ganglia = SpiderGangliaCfC()
        
        # ✅ NEW: Liquid CfC experts
        self.experts = nn.ModuleDict({
            'math': SpiderLiquidExpert(input_dim=64, hidden_dim=256),
            'code': SpiderLiquidExpert(input_dim=32, hidden_dim=128),
            'real_world': SpiderLiquidExpert(input_dim=16, hidden_dim=128)
        })
        
        self.vibration_head = RealVibrationHead()
        self.central_brain = SpiderCentralBrain()
        self.adaptation = SpiderAdaptation()
        
        self._inference_mode = False
        self.debug_mode = debug_mode

    def set_inference_mode(self, enabled: bool = True):
        self._inference_mode = enabled

    def set_debug_mode(self, enabled: bool = True):
        self.debug_mode = enabled

    def forward(self, hidden_states):
        # Convert to list format if tensor input
        if isinstance(hidden_states, (list, tuple)):
            hs_list = hidden_states
        elif torch.is_tensor(hidden_states):
            hs_list = [hidden_states[:, i, :, :] for i in range(hidden_states.shape[1])]
        else:
            raise TypeError("hidden_states must be list of tensors or a single tensor")

        batch_size = hs_list[0].size(0)
        
        # Multi-stream sensory extraction: each stream is (B, L, 8, H)
        streams = self.spider_eyes(hs_list)
        
        # Parallel compression: each stream becomes (B, L, 8, compressed_dim)
        compressed = self.optic_nerve(streams)
        
        # Lateral inhibition: (B, L, 8, compressed_dim)
        inhibited = self.lateral_inhibition(compressed)
        
        # Spectral vibration: (B, 1) or (B,)
        vibration = self.vibration_head(streams)
        
        # Per-sample CfC ganglia routing
        routing, ganglia_diag = self.ganglia(inhibited, vibration)
        
        # Process each domain through liquid experts
        expert_outputs = {}
        for domain_name in ['math', 'code', 'real_world']:
            routed_input = routing[domain_name]
            expert_logits = self.experts[domain_name](routed_input)
            expert_outputs[domain_name] = expert_logits  # (B, 2)

        # ────────────────────────────────────────────────────────────────
        # BIOMIMETIC PANIC GATE - FIXED BROADCASTING
        # ────────────────────────────────────────────────────────────────
        # Ensure panic_factor is (B, 1) to broadcast with (B, 2)
        panic_factor = (1.0 - vibration.clamp(0, 1)).unsqueeze(-1)   # shape: (B, 1)
        
        # GATED FUSION: prevents domain interference
        all_logits = torch.stack([
            expert_outputs['math'],
            expert_outputs['code'],
            expert_outputs['real_world']
        ], dim=1)                                       # (B, 3, 2)

        router_probs = ganglia_diag['train_routing_probs']  # (B, 3)

        # Apply router-based gating
        gated_logits = all_logits * router_probs.unsqueeze(-1)  # (B, 3, 2)

        # Sum across domains
        fused_logits = gated_logits.sum(dim=1)          # (B, 2)

        # Apply panic suppression (now broadcasts correctly)
        final_logits = fused_logits * panic_factor      # (B, 2) * (B, 1) → (B, 2)
        # ────────────────────────────────────────────────────────────────

        # Domain diagnostics
        primary_domain_idx = torch.argmax(router_probs, dim=1)
        domain_confidence = router_probs.max(dim=1).values
        
        domain_map = {0: "math", 1: "code", 2: "real_world"}
        
        entropy = -torch.mean(torch.sum(router_probs * torch.log(router_probs + 1e-8), dim=-1))
        
        routing_info = {
            "primary_domain": domain_map.get(primary_domain_idx[0].item(), "math"),
            "domain_confidence": float(domain_confidence.mean().item()),
            "routing_entropy": float(entropy.item()),
            "panic_factor": float(panic_factor.mean().item()),
            "vibration": vibration.detach(),
            "train_routing_probs": router_probs,
            "cfc_hidden_size": 256
        }
        
        return final_logits, routing_info

# ==================== CfC Network (Closed-form Continuous-time) ====================
class CfCNet(nn.Module):
    """Minimal CfC implementation for liquid time-constant dynamics"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, mode: str = "no_gate"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.time_const = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_in.weight)
        nn.init.orthogonal_(self.w_hh.weight)
        nn.init.constant_(self.time_const, 1.0)

    def forward(self, x: torch.Tensor, hx: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, input_size)
            hx: (batch, hidden_size) initial liquid state
        Returns:
            output: (batch, seq_len, hidden_size)
            hx: (batch, hidden_size) final liquid state
        """
        batch_size, seq_len, _ = x.shape
        
        if hx is None:
            hx = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        else:
            hx = hx.squeeze(0) if hx.dim() == 3 else hx
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)
            
            # ✅ FIX: Clamp time constant to prevent divergence
            tau = self.time_const.clamp(min=0.5, max=5.0)
            decay = torch.exp(-1.0 / tau)
            
            f_t = torch.tanh(self.w_in(x_t) + self.w_hh(hx))
            hx = decay * hx + (1 - decay) * f_t
            
            outputs.append(hx.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        return output, hx.unsqueeze(0)