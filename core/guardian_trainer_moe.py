#!/usr/bin/env python3
# core/guardian_trainer_moe.py (v17.1 - Logging + Summary Save)
"""
Spider-Native Co-Evolutionary Trainer
- ✅ Router pretraining phase (evolutionary hardwiring)
- ✅ Liquid CfC experts for 8-leg temporal processing
- ✅ Temperature annealing schedule (neural maturation)
- ✅ Biomarker monitoring (entropy, panic, vibration)
- ✅ Math-specific loss scaling (3.5× weight for domain_idx=0)
- ✅ Logging: per-epoch CSV and final JSON summary
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm
import csv

# Root setup
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.guardian_vision_core import GuardianVisionNet
from core.guardian_utils import FocalLoss
from core.guardian_dataset_live import GuardianLiveDataLoader
from config import ARTIFACTS_DIR

# Defaults (will be overridden by CLI)
ARTIFACTS_DIR = Path(ARTIFACTS_DIR)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)

SEED = 42
BATCH_SIZE = 2048
LEARNING_RATE_FAST = 2e-3
LEARNING_RATE_SLOW = 1.5e-4
LEARNING_RATE_MID = 8e-4
EPOCHS = 60
FAST_STEPS = 1
VIBRATION_LOSS_WEIGHT = 0.8
BCM_VARIANCE_THRESHOLD = 0.005
LONG_PATIENCE = 30

# Router-specific gradient controls
ROUTER_GRAD_CLIP = 0.5
ROUTER_GRAD_SCALE = 0.5

# Temperature annealing
TEMP_START = 1.5
TEMP_FLOOR = 1.0
TEMP_ANNEAL_EPOCHS = 30

# Gradient clipping
MAX_GRAD_NORM = 1.0

# Math-specific loss scaling (applied to domain loss for math domain)
MATH_DOMAIN_IDX = 0
MATH_LOSS_MULTIPLIER = 3.5

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

def calculate_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error"""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if mask.any():
            bin_acc = labels[mask].mean()
            bin_conf = probs[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(probs))
    return float(ece)

def save_json(obj, path):
    """Save JSON with proper encoding"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class Colors:
    """Terminal color codes"""
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

class CoEvolutionaryTrainer:
    """Main trainer class with biomimetic training strategies"""
    
    def __init__(self, artifacts_dir=ARTIFACTS_DIR, device="cuda", domain_loss_weight=0.1):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Instance variables for weights (default values)
        self.domain_loss_weight = domain_loss_weight
        self.balance_loss_weight = 5.0
        self.entropy_loss_weight = 0.001
        self.router_grad_clip = ROUTER_GRAD_CLIP
        
        self.optimizer_fast = None
        self.optimizer_slow = None
        self.scheduler_fast = None
        self.scheduler_slow = None
        
        self.scaler = GradScaler(
            init_scale=2**14,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=500,
            enabled=True
        )
        self.criterion_hallu = FocalLoss()
        self.criterion_domain = nn.CrossEntropyLoss()
        self.metrics_history = []
        self.current_temp = TEMP_START
        
        # BCM tracking
        self.recent_variance = {0: [], 1: [], 2: []}
        
        # Domain collapse detection
        self.domain_collapse_counter = 0
        
        # NaN detection counter
        self.nan_counter = 0
        
        # Pretraining flag
        self.pretrained = False
        
        # Resume flag (set by pipeline)
        self.resume = False

    def load_data(self, train_prefix="train", val_prefix="val"):
        """Load raw data paths for live extraction"""
        print("🕸️ Loading raw data for live feature extraction...")
        
        train_path = self.artifacts_dir / f"{train_prefix}.pkl"
        val_path = self.artifacts_dir / f"{val_prefix}.pkl"
        
        if not train_path.exists() or not val_path.exists():
            raise FileNotFoundError(f"Raw data not found at {train_path} or {val_path}")
        
        self.train_path = train_path
        self.val_path = val_path
        
        print(f"✅ Data paths ready: train={train_path.name}, val={val_path.name}")

    def build_model(self):
        """Build model and separate optimizers for molecular clocks"""
        print("🕷️ Building Spider-Native GuardianVisionNet with Molecular Clocks...")
        
        self.model = GuardianVisionNet().to(self.device)
        
        # Disable compile for stability on Windows
        if hasattr(torch, 'compile'):
            print("⚠️  Skipping torch.compile (Windows stability)")
        
        # SEPARATE PARAMETER GROUPS
        fast_params = [
            {'params': self.model.ganglia.parameters(), 'lr': LEARNING_RATE_FAST},
            {'params': self.model.vibration_head.parameters(), 'lr': LEARNING_RATE_FAST * 0.2},
            {'params': self.model.spider_eyes.parameters(), 'lr': LEARNING_RATE_FAST * 0.5},
        ]
        
        slow_params = [
            {'params': self.model.experts.parameters(), 'lr': LEARNING_RATE_SLOW},
            {'params': self.model.central_brain.parameters(), 'lr': LEARNING_RATE_SLOW},
            {'params': self.model.optic_nerve.parameters(), 'lr': LEARNING_RATE_MID},
            {'params': self.model.lateral_inhibition.parameters(), 'lr': LEARNING_RATE_MID * 0.5},
        ]
        
        self.optimizer_fast = optim.AdamW(fast_params, weight_decay=1e-5)
        self.optimizer_slow = optim.AdamW(slow_params, weight_decay=1e-4)
        
        # SEPARATE SCHEDULERS
        self.scheduler_fast = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fast, mode='min', factor=0.5, patience=3
        )
        self.scheduler_slow = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_slow, mode='min', factor=0.5, patience=8
        )
        
        print(f"✅ Model built with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"   Fast components: {len(fast_params)} groups, LR={LEARNING_RATE_FAST}")
        print(f"   Slow components: {len(slow_params)} groups, LR={LEARNING_RATE_SLOW}")

    def pretrain_router(self, epochs: int = 2, batch_size: int = 128):
        """Evolutionary pretraining: Router develops innate domain sense"""
        print(f"\n{Colors.YELLOW}{'='*80}{Colors.RESET}")
        print(f"{Colors.YELLOW}🧬 PHASE 1: ROUTER EVOLUTION (Pretraining){Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*80}{Colors.RESET}")
        
        # Freeze everything except ganglia and optic nerve
        for name, param in self.model.named_parameters():
            if "ganglia" not in name and "optic_nerve" not in name:
                param.requires_grad = False
        
        # Dedicated optimizer for routing pathway
        router_params = list(self.model.ganglia.parameters()) + \
                       list(self.model.optic_nerve.parameters())
        optimizer = torch.optim.AdamW(router_params, lr=5e-4, weight_decay=1e-5)
        
        # Set high temperature for exploration
        try:
            self.model.ganglia.routing_temp.fill_(2.5)
        except Exception:
            pass
        
        for epoch in range(epochs):
            self.model.train()
            total_dom_loss = 0.0
            
            train_loader = GuardianLiveDataLoader(
                data_path=self.train_path,
                batch_size=batch_size,
                split="train"
            ).get_dataloader(shuffle=True)
            
            with tqdm(total=len(train_loader), desc=f"Router Epoch {epoch}") as pbar:
                for batch in train_loader:
                    hidden_states = batch['hidden_states'].to(self.device, non_blocking=True)
                    domains = batch['domain'].to(self.device, non_blocking=True)
                    
                    # Forward through frozen pipeline
                    with torch.no_grad():
                        streams = self.model.spider_eyes(hidden_states)
                        compressed = self.model.optic_nerve(streams)
                        inhibited = self.model.lateral_inhibition(compressed)
                        vibration = self.model.vibration_head(streams)
                    
                    # Only train ganglia
                    _, ganglia_diag = self.model.ganglia(inhibited, vibration)
                    routing_logits = ganglia_diag['train_routing_logits']
                    
                    loss = self.criterion_domain(routing_logits, domains)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(router_params, 0.5)
                    
                    optimizer.step()
                    
                    total_dom_loss += loss.item()
                    pbar.update(1)
            
            # Log entropy to monitor specialization
            with torch.no_grad():
                probs = torch.softmax(routing_logits, dim=-1)
                entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-8), dim=-1))
            
            print(f"  → Domain Loss: {total_dom_loss / len(train_loader):.4f} | Entropy: {entropy:.3f}")
        
        # Unfreeze for Phase 2
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Reset temperature for joint training
        try:
            self.model.ganglia.routing_temp.fill_(2.0)
        except Exception:
            pass
        
        self.pretrained = True
        
        print(f"{Colors.GREEN}✅ Router evolved. Starting adaptive phase...{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*80}{Colors.RESET}\n")

    def calculate_vibration_loss(self, vibration, labels):
        """Calculate spectral diversity loss"""
        if not torch.is_tensor(vibration) or vibration.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        spectral_diversity = torch.std(vibration)
        return -spectral_diversity * VIBRATION_LOSS_WEIGHT

    def calculate_balance_loss(self, routing_info):
        """Calculate domain balance loss"""
        if 'train_routing_logits' not in routing_info:
            return torch.tensor(0.0, device=self.device)
        
        routing_logits = routing_info['train_routing_logits']
        if routing_logits is None or not routing_logits.requires_grad:
            return torch.tensor(0.0, device=self.device)
        
        routing_probs = torch.softmax(routing_logits / self.current_temp, dim=-1)
        batch_probs = routing_probs.mean(dim=0)
        target_dist = torch.tensor([1/3, 1/3, 1/3], device=self.device)
        
        return F.mse_loss(batch_probs, target_dist) * self.balance_loss_weight

    def calculate_entropy_loss(self, routing_info):
        """Calculate entropy regularization loss"""
        if 'train_routing_logits' not in routing_info:
            return torch.tensor(0.0, device=self.device)
        
        routing_logits = routing_info['train_routing_logits']
        if routing_logits is None or not routing_logits.requires_grad:
            return torch.tensor(0.0, device=self.device)
        
        routing_probs = torch.softmax(routing_logits / self.current_temp, dim=-1)
        probs_clipped = routing_probs.clamp(min=1e-8, max=1-1e-8)
        entropy = -torch.mean(torch.sum(probs_clipped * torch.log(probs_clipped + 1e-8), dim=-1))
        
        return -entropy * self.entropy_loss_weight

    def track_bcm_variance(self, routing_info, domains):
        """Track variance for BCM theory"""
        if 'train_routing_logits' not in routing_info:
            return
        
        routing_logits = routing_info['train_routing_logits']
        if routing_logits is None:
            return
        
        routing_probs = torch.softmax(routing_logits, dim=-1)
        confidences = routing_probs.max(dim=-1).values
        
        for i, domain_idx in enumerate(domains):
            domain_int = domain_idx.item()
            conf = confidences[i].item()
            
            self.recent_variance[domain_int].append(conf)
            if len(self.recent_variance[domain_int]) > 100:
                self.recent_variance[domain_int].pop(0)
            
            if len(self.recent_variance[domain_int]) >= 10:
                variance = np.var(self.recent_variance[domain_int])
                if variance < BCM_VARIANCE_THRESHOLD:
                    try:
                        self.model.adaptation.bcm_boost[domain_int] = min(
                            self.model.adaptation.bcm_boost[domain_int].item() * 1.1,
                            10.0
                        )
                    except Exception:
                        pass
                    if getattr(self.model, "debug_mode", False):
                        domain_map = {0: "math", 1: "code", 2: "real_world"}
                        print(f"[BCM] Domain {domain_map.get(domain_int, 'unknown')} variance {variance:.4f} < {BCM_VARIANCE_THRESHOLD} → Temp boosted")

    def detect_domain_collapse(self, routing_info):
        """Detect and recover from domain collapse"""
        if 'routing_weights' not in routing_info:
            return
        
        weights = routing_info['routing_weights']
        max_domain = max(weights, key=weights.get)
        max_weight = weights[max_domain]
        
        if max_weight > 0.95:
            self.domain_collapse_counter += 1
            if self.domain_collapse_counter >= 5:
                print(f"🚨 DOMAIN COLLAPSE DETECTED: {max_domain}={max_weight:.1%}")
                
                try:
                    self.model.ganglia.routing_temp.fill_(2.0)
                    self.model.ganglia.liquid_state.zero_()
                    self.model.ganglia.last_liquid_state.zero_()
                except Exception:
                    pass
                
                for param in self.model.ganglia.parameters():
                    if hasattr(param, 'data'):
                        param.data += torch.randn_like(param) * 0.01
                
                self.domain_collapse_counter = 0
                print("✅ Router state reset - temperature restored to 2.0")
        else:
            self.domain_collapse_counter = 0

    def train_epoch(self, epoch: int, batch_size: int):
        """Co-evolutionary training with molecular clock speeds"""
        self.model.train()
        total_loss = 0.0
        vibration_total = 0.0
        domain_total = 0.0
        balance_total = 0.0
        entropy_total = 0.0
        
        # Use dynamic batch size
        train_loader = GuardianLiveDataLoader(
            data_path=self.train_path,
            batch_size=batch_size,
            split="train"
        ).get_dataloader(shuffle=True)
        
        num_batches = len(train_loader)
        if num_batches == 0:
            return {
                "train_loss": 0.0,
                "vibration_loss": 0.0,
                "domain_loss": 0.0,
                "balance_loss": 0.0,
                "entropy_loss": 0.0
            }
        
        # Reset CfC state buffer at epoch start
        try:
            self.model.ganglia.last_liquid_state.zero_()
        except Exception:
            pass
        
        with tqdm(total=num_batches, desc=f"Epoch {epoch:02d}", unit="batch", ascii=True, dynamic_ncols=True) as pbar:
            for batch_idx, batch in enumerate(train_loader):
                # Move data to device
                hidden_states = batch['hidden_states'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                domains = batch['domain'].to(self.device, non_blocking=True)
                
                # Forward pass (single pass per batch)
                device_type = self.device.type  # 'cuda' or 'cpu'
                with autocast(device_type=device_type, dtype=torch.float16):
                    logits, routing_info = self.model(hidden_states)
                    
                    # Hallucination loss (primary objective)
                    hallu_loss = self.criterion_hallu(logits, labels)
                    
                    # Vibration loss (spectral diversity)
                    vibration = routing_info.get('vibration')
                    vibration_loss = self.calculate_vibration_loss(vibration, labels)
                    
                    # Domain loss (supervise router)
                    domain_loss = torch.tensor(0.0, device=self.device)
                    if self.model.training and 'train_routing_logits' in routing_info:
                        routing_logits = routing_info['train_routing_logits']
                        if torch.is_tensor(routing_logits) and routing_logits.requires_grad:
                            domain_loss = self.criterion_domain(routing_logits, domains)
                    
                    # ✅ MATH BOOST: Increase loss weight for math domain (domain_idx=0)
                    math_mask = (domains == MATH_DOMAIN_IDX)
                    if math_mask.any() and domain_loss.item() > 0:
                        domain_loss = domain_loss * MATH_LOSS_MULTIPLIER  # 3.5× stronger penalty for math routing errors
                    
                    # Balance loss (prevent collapse)
                    balance_loss = self.calculate_balance_loss(routing_info)
                    
                    # Entropy regularization (exploration)
                    entropy_loss = self.calculate_entropy_loss(routing_info)
                
                # Split losses for molecular clocks
                fast_loss_contrib = hallu_loss * 0.3 + vibration_loss + domain_loss * self.domain_loss_weight + \
                                   balance_loss + entropy_loss
                
                slow_loss_contrib = hallu_loss * 0.7
                
                # Accumulate metrics for logging
                total_loss += hallu_loss.item()
                vibration_total += vibration_loss.item()
                domain_total += domain_loss.item()
                balance_total += balance_loss.item()
                entropy_total += entropy_loss.item()
                
                # Clear gradients
                self.optimizer_fast.zero_grad(set_to_none=True)
                self.optimizer_slow.zero_grad(set_to_none=True)
                
                # Perform all backwards BEFORE any optimizer steps
                # Fast path backward (retain graph for slow path)
                self.scaler.scale(fast_loss_contrib).backward(retain_graph=True)
                
                # Slow path backward (on same graph, no retain needed)
                self.scaler.scale(slow_loss_contrib).backward()
                
                # Unscale for clipping
                self.scaler.unscale_(self.optimizer_fast)
                self.scaler.unscale_(self.optimizer_slow)
                
                # Gradient clipping
                try:
                    torch.nn.utils.clip_grad_norm_(self.model.ganglia.parameters(), self.router_grad_clip)
                    torch.nn.utils.clip_grad_norm_(self.model.vibration_head.parameters(), MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.model.spider_eyes.parameters(), MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.model.experts.parameters(), MAX_GRAD_NORM)
                    torch.nn.utils.clip_grad_norm_(self.model.central_brain.parameters(), MAX_GRAD_NORM)
                except Exception:
                    pass
                
                # Optimizer steps
                self.scaler.step(self.optimizer_fast)
                self.scaler.step(self.optimizer_slow)
                
                # Update scaler ONCE per batch
                self.scaler.update()
                
                # Emergency NaN detection and recovery
                if torch.isnan(hallu_loss) or torch.isinf(hallu_loss):
                    self.nan_counter += 1
                    print(f"⚠️ NaN detected in batch {batch_idx}! Counter: {self.nan_counter}")
                    if self.nan_counter >= 3:
                        print("🚨 Emergency recovery: Halving learning rates...")
                        for pg in self.optimizer_fast.param_groups:
                            pg['lr'] *= 0.5
                        for pg in self.optimizer_slow.param_groups:
                            pg['lr'] *= 0.5
                        self.nan_counter = 0
                    continue
                else:
                    self.nan_counter = 0
                
                # BCM variance tracking
                self.track_bcm_variance(routing_info, domains)
                
                # Domain collapse detection
                self.detect_domain_collapse(routing_info)
                
                pbar.update(1)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        metrics = {
            "train_loss": total_loss / num_batches,
            "vibration_loss": vibration_total / num_batches,
            "domain_loss": domain_total / num_batches,
            "balance_loss": balance_total / num_batches,
            "entropy_loss": entropy_total / num_batches,
        }
        
        return metrics

    def validate(self, batch_size: int):
        """Validation with robust per-domain tracking"""
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        domain_stats = defaultdict(lambda: {"correct": 0, "total": 0, "confidences": []})
        
        # Reset CfC state buffer before validation
        try:
            self.model.ganglia.last_liquid_state.zero_()
        except Exception:
            pass
        
        # Use dynamic batch size
        val_loader = GuardianLiveDataLoader(
            data_path=self.val_path,
            batch_size=max(1, batch_size * 2)  # 2x for faster validation
        ).get_dataloader(shuffle=False)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                hidden_states = batch['hidden_states'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                domains = batch['domain'].to(self.device, non_blocking=True)
                
                self.model.set_inference_mode(False)
                logits, routing_info = self.model(hidden_states)
                
                logits = torch.clamp(logits, -20.0, 20.0)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                batch_size_current = hidden_states.shape[0]
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                # assume binary hallucination label at index 1 probability
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                for i in range(batch_size_current):
                    domain_idx = domains[i].item()
                    domain_name = {0: "math", 1: "code", 2: "real_world"}[domain_idx]
                    domain_stats[domain_name]["total"] += 1
                    
                    if preds[i] == labels[i]:
                        domain_stats[domain_name]["correct"] += 1
                    
                    if isinstance(routing_info, dict):
                        domain_stats[domain_name]["confidences"].append(
                            routing_info.get('domain_confidence', 0.0)
                        )
                
                del hidden_states, labels, domains, logits, routing_info, probs, preds
        
        del val_loader
        torch.cuda.empty_cache()
        
        if len(all_labels) == 0:
            return {
                "val_loss": 0.0,
                "val_acc": 0.0,
                "val_f1": 0.0,
                "val_ece": 0.0,
                "domain_accuracies": {}
            }
        
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        accuracy = float(np.mean(y_true == y_pred))
        f1 = f1_score(y_true, y_pred, zero_division=0)
        ece = calculate_ece(np.array(all_probs), y_true)
        
        domain_accuracies = {}
        for domain, stats in domain_stats.items():
            if stats["total"] > 0:
                domain_accuracies[domain] = stats["correct"] / stats["total"]
                avg_conf = np.mean(stats["confidences"]) if stats["confidences"] else 0.0
                print(f"[Val] {domain:12}: acc={domain_accuracies[domain]:.2%}, conf={avg_conf:.2f}, n={stats['total']}")
        
        return {
            "val_loss": 0.0,
            "val_acc": accuracy,
            "val_f1": f1,
            "val_ece": ece,
            "domain_accuracies": domain_accuracies
        }

    def _detect_math_ratio(self):
        """Try to infer math ratio from manifest if available"""
        manifest_path = self.artifacts_dir / "01_manifest.json"
        try:
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = json.load(f)
                domain_counts = m.get("domain_counts", {})
                total = sum(domain_counts.values()) if domain_counts else 0
                math = domain_counts.get("math", 0)
                if total > 0:
                    return float(math) / float(total)
        except Exception:
            pass
        return None

    def _init_logging(self):
        """Ensure CSV header exists"""
        log_path = self.artifacts_dir / "training_log.csv"
        if not log_path.exists():
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "val_acc", "val_f1", "val_ece", "code_acc", "math_acc", "real_acc", "timestamp"])
        return log_path

    def _append_log(self, log_path: Path, epoch: int, val_results: Dict[str, Any]):
        code_acc = val_results["domain_accuracies"].get("code", 0.0)
        math_acc = val_results["domain_accuracies"].get("math", 0.0)
        real_acc = val_results["domain_accuracies"].get("real_world", 0.0)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{val_results['val_acc']:.4f}",
                f"{val_results['val_f1']:.4f}",
                f"{val_results['val_ece']:.4f}",
                f"{code_acc:.4f}",
                f"{math_acc:.4f}",
                f"{real_acc:.4f}",
                int(time.time())
            ])

    def train(self, epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, fast_steps: int = FAST_STEPS,
              pretrain: bool = True):
        """Full co-evolutionary training loop with optional pretraining"""
        print(f"\n{Colors.YELLOW}{'='*80}{Colors.RESET}")
        print(f"{Colors.YELLOW}🕷️ SPIDER-NATIVE CO-EVOLUTIONARY TRAINING v17.0{Colors.RESET}")
        print(f"{Colors.YELLOW}Batch Size: {batch_size} | Device: {self.device}{Colors.RESET}")
        print(f"{Colors.YELLOW}{'='*80}{Colors.RESET}\n")
        print(f"   Domain Weight: {self.domain_loss_weight} | Balance Weight: {self.balance_loss_weight} | Entropy Weight: {self.entropy_loss_weight}")
        print(f"   Math Loss Multiplier: {MATH_LOSS_MULTIPLIER}x")
        
        best_val_f1 = 0.0
        best_epoch = 0
        patience_counter = 0
        start_epoch = 0
        
        checkpoint_path = self.artifacts_dir / "guardian_spider_native_full.pth"
        if hasattr(self, 'resume') and self.resume and checkpoint_path.exists():
            try:
                print(f"📂 Resuming from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                model_state = checkpoint.get('model_state_dict', None)
                if model_state is not None:
                    try:
                        if hasattr(self.model, '_orig_mod'):
                            self.model._orig_mod.load_state_dict(model_state)
                        else:
                            self.model.load_state_dict(model_state)
                    except Exception:
                        # fallback: strict=False
                        self.model.load_state_dict(model_state, strict=False)
                
                if 'optimizer_fast_state_dict' in checkpoint:
                    try:
                        self.optimizer_fast.load_state_dict(checkpoint['optimizer_fast_state_dict'])
                    except Exception:
                        pass
                if 'optimizer_slow_state_dict' in checkpoint:
                    try:
                        self.optimizer_slow.load_state_dict(checkpoint['optimizer_slow_state_dict'])
                    except Exception:
                        pass
                
                if 'scheduler_fast_state_dict' in checkpoint:
                    try:
                        self.scheduler_fast.load_state_dict(checkpoint['scheduler_fast_state_dict'])
                    except Exception:
                        pass
                if 'scheduler_slow_state_dict' in checkpoint:
                    try:
                        self.scheduler_slow.load_state_dict(checkpoint['scheduler_slow_state_dict'])
                    except Exception:
                        pass
                
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_f1 = checkpoint.get('best_val_f1', 0.0)
                best_epoch = checkpoint.get('best_epoch', 0)
                self.recent_variance = checkpoint.get('recent_variance', {0: [], 1: [], 2: []})
                self.domain_collapse_counter = checkpoint.get('domain_collapse_counter', 0)
                self.nan_counter = checkpoint.get('nan_counter', 0)
                self.pretrained = checkpoint.get('pretrained', False)
                
                if 'rng_state' in checkpoint:
                    try:
                        random.setstate(checkpoint['rng_state']['python'])
                        np.random.set_state(checkpoint['rng_state']['numpy'])
                        torch.set_rng_state(checkpoint['rng_state']['torch'].cpu())
                        if torch.cuda.is_available() and 'cuda' in checkpoint['rng_state']:
                            torch.cuda.set_rng_state(checkpoint['rng_state']['cuda'].cpu())
                    except Exception:
                        pass
                
                print(f"✅ Resumed from epoch {start_epoch}, best F1: {best_val_f1:.4f}")
            except Exception as e:
                print(f"⚠️ Failed to resume checkpoint: {e}")
                start_epoch = 0
        else:
            print(f"🚀 Starting fresh training from epoch 0")
        
        # ✅ PHASE 1: Router pretraining (if enabled and not resumed)
        if pretrain and not self.pretrained and start_epoch == 0:
            self.pretrain_router(epochs=2, batch_size=batch_size)
        
        # Training metadata
        meta = {
            "seed": SEED,
            "batch_size": batch_size,
            "fast_lr": LEARNING_RATE_FAST,
            "slow_lr": LEARNING_RATE_SLOW,
            "fast_steps": fast_steps,
            "domain_loss_weight": self.domain_loss_weight,
            "balance_loss_weight": self.balance_loss_weight,
            "entropy_loss_weight": self.entropy_loss_weight,
            "router_grad_clip": self.router_grad_clip,
            "vibration_weight": VIBRATION_LOSS_WEIGHT,
            "bcm_threshold": BCM_VARIANCE_THRESHOLD,
            "epochs": epochs,
            "architecture": "spider_native_v17.0_math_boost",
            "timestamp": time.time(),
            "temp_schedule": "neural_maturation",
            "temp_floor": TEMP_FLOOR,
            "grad_scaler_init": 2**14,
            "nan_counter": self.nan_counter,
            "pretrained": self.pretrained,
            "math_loss_multiplier": MATH_LOSS_MULTIPLIER
        }
        save_json(meta, self.artifacts_dir / "train_meta.json")
        
        # Initialize logging CSV
        log_path = self._init_logging()
        math_ratio = self._detect_math_ratio()
        
        print(f"\n{'Epoch':<6} {'Train':<9} {'Vib':<7} {'Dom':<7} {'Bal':<7} {'Ent':<7} {'Val Acc':<8} {'Val F1':<8} {'Val ECE':<8} {'Status'}")
        print("-" * 90)
        
        for epoch in range(start_epoch, epochs):
            # ✅ ROUTING TEMPERATURE ANNEALING (mimics neural maturation)
            if epoch < TEMP_ANNEAL_EPOCHS:
                # linear anneal from TEMP_START down to TEMP_FLOOR
                frac = epoch / max(1, TEMP_ANNEAL_EPOCHS - 1)
                self.current_temp = TEMP_START - frac * (TEMP_START - TEMP_FLOOR)
            else:
                self.current_temp = TEMP_FLOOR
            
            # Train epoch
            epoch_metrics = self.train_epoch(epoch, batch_size)
            
            # Validate
            val_results = self.validate(batch_size)
            
            # Logging to CSV
            try:
                self._append_log(log_path, epoch, val_results)
            except Exception as e:
                print(f"⚠️ Failed to append log for epoch {epoch}: {e}")
            
            # Track best
            val_f1 = val_results.get("val_f1", 0.0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                # Save best model (state + optimizers)
                try:
                    checkpoint = {
                        "epoch": epoch,
                        "best_val_f1": best_val_f1,
                        "best_epoch": best_epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_fast_state_dict": self.optimizer_fast.state_dict(),
                        "optimizer_slow_state_dict": self.optimizer_slow.state_dict(),
                        "scheduler_fast_state_dict": getattr(self.scheduler_fast, "state_dict", lambda: {})(),
                        "scheduler_slow_state_dict": getattr(self.scheduler_slow, "state_dict", lambda: {})(),
                        "recent_variance": self.recent_variance,
                        "domain_collapse_counter": self.domain_collapse_counter,
                        "nan_counter": self.nan_counter,
                        "pretrained": self.pretrained,
                        "rng_state": {
                            "python": random.getstate(),
                            "numpy": np.random.get_state(),
                            "torch": torch.get_rng_state().cpu().numpy().tolist() if torch.get_rng_state() is not None else None
                        }
                    }
                    torch.save(checkpoint, checkpoint_path)
                    # also save a lightweight best model copy
                    torch.save(self.model.state_dict(), self.artifacts_dir / "guardian_spider_native.pth")
                except Exception as e:
                    print(f"⚠️ Failed to save checkpoint at epoch {epoch}: {e}")
                status = "✓ Best"
                patience_counter = 0
            else:
                status = "⏳"
                patience_counter += 1
            
            # Print epoch summary
            print(f"{epoch:3d} {epoch_metrics['train_loss']:<9.4f} {epoch_metrics['vibration_loss']:<7.3f} "
                  f"{epoch_metrics['domain_loss']:<7.3f} {epoch_metrics['balance_loss']:<7.3f} {epoch_metrics['entropy_loss']:<7.3f} "
                  f"{val_results['val_acc']:<8.4f} {val_results['val_f1']:<8.4f} {val_results['val_ece']:<8.4f} {status}")
            
            # Scheduler step (use domain_loss as proxy)
            try:
                self.scheduler_fast.step(epoch_metrics.get("domain_loss", 0.0))
                self.scheduler_slow.step(epoch_metrics.get("domain_loss", 0.0))
            except Exception:
                pass
            
            # Early stopping
            if patience_counter >= LONG_PATIENCE:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs (patience {LONG_PATIENCE})")
                break
        
        # Final summary save
        try:
            summary = {
                "best_f1": float(best_val_f1),
                "best_epoch": int(best_epoch),
                "final_epoch": int(epoch),
                "math_ratio": float(math_ratio) if math_ratio is not None else None,
                "timestamp": time.time()
            }
            save_json(summary, self.artifacts_dir / "training_summary.json")
        except Exception as e:
            print(f"⚠️ Failed to save training summary: {e}")
        
        print("\n" + "="*80)
        print(f"{Colors.GREEN}✅ Training complete! Best F1: {best_val_f1:.4f}{Colors.RESET}")
        print(f"Model: {self.artifacts_dir / 'guardian_spider_native.pth'}")
        print(f"Checkpoint: {checkpoint_path}")
        print("="*80 + "\n")
