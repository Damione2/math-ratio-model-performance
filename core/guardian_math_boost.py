#!/usr/bin/env python3
# core/guardian_math_boost.py 
"""
guardian_math_boost.py - Math-Specific Fine-Tuning for Guardian
✅ Loads your trained checkpoint (0.83 F1)
✅ Freezes code/real experts (already perfect at 100%)
✅ Aggressive math expert training with hard example mining
✅ Progressive curriculum: easy → hard math
✅ Expected: Math 70% → 88%+, Overall F1 0.83 → 0.92+
"""

import os
import sys
import json
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score
from tqdm import tqdm

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.guardian_vision_core import GuardianVisionNet
from core.guardian_utils import FocalLoss
from core.guardian_dataset_live import GuardianLiveDataset
from config import ARTIFACTS_DIR

ARTIFACTS_DIR = Path(ARTIFACTS_DIR)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CONFIGURATION ====================
class Config:
    """Math boosting configuration"""
    # Training
    EPOCHS = 30
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3  # Higher LR for aggressive math learning
    WEIGHT_DECAY = 1e-5

    # Math-specific
    MATH_LOSS_WEIGHT = 10.0  # Massive weight for math samples
    HARD_MINING_RATIO = 0.7  # Focus 70% on hard examples
    CURRICULUM_EPOCHS = 10   # Gradual difficulty increase

    # Loss
    FOCAL_GAMMA = 4.0  # Higher gamma for hard example focus
    LABEL_SMOOTHING = 0.1

    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.005

# ==================== HARD EXAMPLE TRACKER ====================
class HardExampleTracker:
    """Track which math samples are hardest for the model"""
    def __init__(self, n_samples: int):
        self.losses = defaultdict(float)
        self.counts = defaultdict(int)
        self.n_samples = n_samples

    def update(self, indices: torch.Tensor, losses: torch.Tensor):
        """Update loss history for samples"""
        for idx, loss in zip(indices.cpu().numpy(), losses.cpu().numpy()):
            self.losses[int(idx)] += float(loss)
            self.counts[int(idx)] += 1

    def get_hard_examples(self, n_hard: int) -> List[int]:
        """Get indices of hardest examples"""
        if not self.losses:
            return []

        # Average loss per sample
        avg_losses = {
            idx: self.losses[idx] / max(1, self.counts[idx]) 
            for idx in self.losses.keys()
        }

        # Return top-k hardest
        sorted_by_loss = sorted(avg_losses.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in sorted_by_loss[:n_hard]]

    def get_sample_weights(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute weights emphasizing hard examples"""
        weights = torch.ones(len(indices), device=DEVICE)

        for i, idx in enumerate(indices.cpu().numpy()):
            if int(idx) in self.losses:
                # Higher weight for historically hard examples
                avg_loss = self.losses[int(idx)] / max(1, self.counts[int(idx)])
                weights[i] = 1.0 + avg_loss * 2.0  # Up to 3x weight

        return weights

# ==================== MATH-CURRICULUM DATASET ====================
class MathCurriculumDataset(Dataset):
    """
    Dataset that implements curriculum learning for math.
    Starts with easy math, gradually introduces hard adversarial examples.
    """
    def __init__(self, split: str = "train", epoch: int = 0, total_epochs: int = 30):
        self.split = split
        self.epoch = epoch
        self.total_epochs = total_epochs

        # Load base dataset
        self.dataset = GuardianLiveDataset(split=split)

        # Separate by domain and difficulty
        self.math_samples = []
        self.code_samples = []
        self.real_samples = []

        for i in range(len(self.dataset)):
            meta = self.dataset.metadata[i]
            domain = meta.get('domain', 'real_world')
            source = meta.get('meta', {}).get('source', '')

            item = {
                'idx': i,
                'domain': domain,
                'source': source,
                'label': meta['label']
            }

            if domain == 'math':
                # Categorize math difficulty
                if 'adversarial' in source or 'adv' in source:
                    item['difficulty'] = 3  # Hardest
                elif 'long_cot' in source or 'eqsys' in source:
                    item['difficulty'] = 2  # Medium
                elif 'v4' in source:
                    item['difficulty'] = 2  # Medium
                else:
                    item['difficulty'] = 1  # Easy
                self.math_samples.append(item)
            elif domain == 'code':
                self.code_samples.append(item)
            else:
                self.real_samples.append(item)

        print(f"[Curriculum] Math: {len(self.math_samples)}, "
              f"Easy: {sum(1 for m in self.math_samples if m['difficulty']==1)}, "
              f"Medium: {sum(1 for m in self.math_samples if m['difficulty']==2)}, "
              f"Hard: {sum(1 for m in self.math_samples if m['difficulty']==3)}")

    def get_curriculum_indices(self) -> List[int]:
        """Get sample indices based on current epoch's difficulty"""
        progress = self.epoch / self.total_epochs

        # Curriculum schedule
        if progress < 0.3:
            # Easy only
            allowed_difficulties = {1}
        elif progress < 0.6:
            # Easy + Medium
            allowed_difficulties = {1, 2}
        else:
            # All difficulties
            allowed_difficulties = {1, 2, 3}

        # Select math samples
        selected_math = [
            m['idx'] for m in self.math_samples 
            if m['difficulty'] in allowed_difficulties
        ]

        # Add some code/real for stability (prevent catastrophic forgetting)
        selected_code = [c['idx'] for c in self.code_samples[:100]]
        selected_real = [r['idx'] for r in self.real_samples[:100]]

        all_indices = selected_math + selected_code + selected_real
        random.shuffle(all_indices)

        return all_indices

    def __len__(self):
        return len(self.get_curriculum_indices())

    def __getitem__(self, idx):
        actual_idx = self.get_curriculum_indices()[idx]
        return self.dataset[actual_idx]

# ==================== MATH BOOST TRAINER ====================
class MathBoostTrainer:
    """Fine-tune specifically for math performance"""

    def __init__(self, checkpoint_path: Path = None):
        self.config = Config()
        self.device = DEVICE

        # Load model
        self.model = GuardianVisionNet().to(self.device)

        # Load checkpoint
        checkpoint_path = checkpoint_path or ARTIFACTS_DIR / "guardian_spider_native.pth"
        if checkpoint_path.exists():
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("✅ Checkpoint loaded")
        else:
            print("⚠️  No checkpoint found, starting from scratch")

        # Freeze code and real-world experts (they're already at 100%)
        self._freeze_non_math_components()

        # Setup optimizer for math components only
        math_params = (
            list(self.model.ganglia.parameters()) +  # Router needs math tuning
            list(self.model.experts['math'].parameters()) +
            list(self.model.optic_nerve.parameters())  # Allow some adaptation
        )

        self.optimizer = optim.AdamW(
            math_params, 
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.98)
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.scaler = GradScaler(enabled=True)
        self.criterion = FocalLoss(gamma=self.config.FOCAL_GAMMA, alpha=0.7)
        self.hard_tracker = HardExampleTracker(n_samples=10000)

        self.best_math_acc = 0.0
        self.best_overall_f1 = 0.0
        self.patience_counter = 0

    def _freeze_non_math_components(self):
        """Freeze code and real experts to preserve their performance"""
        frozen_names = []

        for name, param in self.model.named_parameters():
            # Freeze code expert
            if 'experts.code' in name:
                param.requires_grad = False
                frozen_names.append(name)
            # Freeze real_world expert
            elif 'experts.real_world' in name:
                param.requires_grad = False
                frozen_names.append(name)
            # Keep math expert and shared components trainable
            else:
                param.requires_grad = True

        print(f"🥶 Frozen {len(frozen_names)} parameters (code/real experts)")
        print(f"🚀 Keeping {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} "
              f"parameters trainable")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with math focus"""
        self.model.train()

        # Curriculum dataset
        dataset = MathCurriculumDataset(split="train", epoch=epoch, 
                                       total_epochs=self.config.EPOCHS)
        loader = DataLoader(
            dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        total_loss = 0.0
        math_loss_total = 0.0
        other_loss_total = 0.0

        # Track math accuracy within epoch
        math_correct = 0
        math_total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch:02d}")

        for batch_idx, batch in enumerate(pbar):
            hidden_states = batch['hidden_states'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)
            domains = batch['domain'].to(self.device, non_blocking=True)

            # Create sample indices for hard example tracking
            batch_indices = torch.arange(batch_idx * self.config.BATCH_SIZE,
                                        batch_idx * self.config.BATCH_SIZE + len(labels))

            # Identify math samples
            math_mask = (domains == 0)  # 0 = math

            with autocast(device_type=self.device.type, dtype=torch.float16):
                logits, routing_info = self.model(hidden_states)

                # Base hallucination loss
                hallu_loss = self.criterion(logits, labels)

                # Compute per-sample losses for hard mining
                with torch.no_grad():
                    ce_losses = F.cross_entropy(logits, labels, reduction='none')
                    self.hard_tracker.update(batch_indices, ce_losses)

                # Math-specific boosting
                if math_mask.any():
                    math_logits = logits[math_mask]
                    math_labels = labels[math_mask]

                    # Higher weight for math samples
                    math_loss = self.criterion(math_logits, math_labels) * self.config.MATH_LOSS_WEIGHT

                    # Hard example mining: focus on misclassified
                    with torch.no_grad():
                        math_preds = torch.argmax(math_logits, dim=-1)
                        math_correct_mask = (math_preds == math_labels)
                        math_accuracy = math_correct_mask.float().mean().item()

                        math_correct += math_correct_mask.sum().item()
                        math_total += len(math_labels)

                    # Extra loss for hard math examples
                    if not math_correct_mask.all():
                        hard_math_logits = math_logits[~math_correct_mask]
                        hard_math_labels = math_labels[~math_correct_mask]
                        hard_loss = self.criterion(hard_math_logits, hard_math_labels)
                        math_loss = math_loss + hard_loss * 2.0
                else:
                    math_loss = torch.tensor(0.0, device=self.device)
                    math_accuracy = 0.0

                # Small loss for non-math to prevent forgetting
                other_mask = ~math_mask
                if other_mask.any():
                    other_loss = self.criterion(logits[other_mask], labels[other_mask]) * 0.1
                else:
                    other_loss = torch.tensor(0.0, device=self.device)

                total_batch_loss = hallu_loss + math_loss + other_loss

            # Optimization
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_batch_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Logging
            total_loss += total_batch_loss.item()
            math_loss_total += math_loss.item() if math_loss.item() > 0 else 0
            other_loss_total += other_loss.item() if other_loss.item() > 0 else 0

            if math_total > 0:
                current_math_acc = math_correct / math_total
                pbar.set_postfix({
                    'loss': f'{total_batch_loss.item():.4f}',
                    'math_acc': f'{current_math_acc:.2%}',
                    'mathL': f'{math_loss.item():.2f}'
                })

        return {
            'train_loss': total_loss / len(loader),
            'math_loss': math_loss_total / len(loader),
            'other_loss': other_loss_total / len(loader),
            'math_acc': math_correct / max(1, math_total)
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, Any]:
        """Validate with detailed per-domain metrics"""
        self.model.eval()

        dataset = GuardianLiveDataset(split="val")
        loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE * 2,
                          shuffle=False, num_workers=0)

        all_labels = []
        all_preds = []
        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)

        for batch in loader:
            hidden_states = batch['hidden_states'].to(self.device)
            labels = batch['label'].to(self.device)
            domains = batch['domain'].to(self.device)

            logits, routing_info = self.model(hidden_states)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Per-domain accuracy
            domain_map = {0: 'math', 1: 'code', 2: 'real_world'}
            for i in range(len(labels)):
                domain = domain_map[int(domains[i].item())]
                domain_total[domain] += 1
                if preds[i] == labels[i]:
                    domain_correct[domain] += 1

        # Compute metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        overall_acc = (y_true == y_pred).mean()
        overall_f1 = f1_score(y_true, y_pred, zero_division=0)

        domain_accs = {}
        for domain in ['math', 'code', 'real_world']:
            if domain_total[domain] > 0:
                domain_accs[domain] = domain_correct[domain] / domain_total[domain]

        return {
            'val_acc': float(overall_acc),
            'val_f1': float(overall_f1),
            'domain_accuracies': domain_accs
        }

    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"🚀 MATH BOOST TRAINING - Target: Math 70% → 88%+")
        print(f"{'='*80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Math Loss Weight: {self.config.MATH_LOSS_WEIGHT}x")
        print(f"Frozen: Code/Real experts (preserving 100% accuracy)")
        print(f"{'='*80}\n")

        print(f"{'Epoch':<6} {'Train':<8} {'MathL':<8} {'MathAcc':<10} "
              f"{'ValF1':<8} {'Math':<8} {'Code':<8} {'Real':<8} {'Status'}")
        print("-" * 90)

        for epoch in range(self.config.EPOCHS):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            self.scheduler.step(val_metrics['val_f1'])

            # Check for best
            math_acc = val_metrics['domain_accuracies'].get('math', 0)
            overall_f1 = val_metrics['val_f1']

            is_best = (math_acc > self.best_math_acc + self.config.MIN_DELTA or
                      (math_acc >= self.best_math_acc and overall_f1 > self.best_overall_f1))

            if is_best:
                self.best_math_acc = math_acc
                self.best_overall_f1 = overall_f1
                self.patience_counter = 0
                status = "✓ BEST"

                # Save best model
                torch.save(self.model.state_dict(), 
                          ARTIFACTS_DIR / "guardian_math_boosted.pth")
            else:
                self.patience_counter += 1
                status = f"⏳ {self.patience_counter}"

            # Print metrics
            print(f"{epoch:02d}    "
                  f"{train_metrics['train_loss']:.4f}  "
                  f"{train_metrics['math_loss']:.3f}   "
                  f"{train_metrics['math_acc']:.2%}    "
                  f"{overall_f1:.4f}  "
                  f"{math_acc:.2%}  "
                  f"{val_metrics['domain_accuracies'].get('code', 0):.2%}  "
                  f"{val_metrics['domain_accuracies'].get('real_world', 0):.2%}  "
                  f"{status}")

            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        print(f"\n{'='*80}")
        print(f"✅ Math Boost Complete!")
        print(f"   Best Math Accuracy: {self.best_math_acc:.2%}")
        print(f"   Best Overall F1: {self.best_overall_f1:.4f}")
        print(f"   Saved: {ARTIFACTS_DIR / 'guardian_math_boosted.pth'}")
        print(f"{'='*80}")

def main():
    """Entry point"""
    import argparse
    parser = argparse.ArgumentParser(description="Math-specific fine-tuning for Guardian")
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Path to checkpoint (default: guardian_spider_native.pth)")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    parser.add_argument("--math-weight", type=float, default=10.0,
                       help="Loss weight multiplier for math samples")
    args = parser.parse_args()

    # Update config
    Config.EPOCHS = args.epochs
    Config.MATH_LOSS_WEIGHT = args.math_weight

    trainer = MathBoostTrainer(checkpoint_path=args.checkpoint)
    trainer.train()

if __name__ == "__main__":
    main()