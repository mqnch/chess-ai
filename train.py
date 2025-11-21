"""
train.py

Implements the training loop for the AlphaZero-style Chess AI.
Handles sampling from the replay buffer, computing losses, and updating the network.

References:
    - Project Vision: "Phase 5 – Training loop"
    - "Implement the training script... policy cross-entropy plus value mean-squared error"
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Optional, Dict, Tuple
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from model import ChessNet
from checkpoint_utils import save_checkpoint as save_ckpt, load_checkpoint as load_ckpt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    """
    Manages the training process for the ChessNet.
    """
    
    def __init__(self, model: ChessNet, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 0.02, weight_decay: float = 1e-4):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer: SGD with momentum is standard for AlphaZero, but Adam is often easier for smaller scale.
        # Vision.md suggests "start with Adam or SGD".
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, 
                                   momentum=0.9, weight_decay=weight_decay)
        
        # Scheduler: Step decay or Cyclic
        # Using a simple StepLR for now
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.1)
        
        self.writer = SummaryWriter(log_dir='logs/train')

    def _prepare_batch(self, states, policies, values) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ensure inputs are float32 tensors on the configured device.
        """
        if isinstance(states, torch.Tensor):
            states = states.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            states = torch.tensor(states, dtype=torch.float32).to(self.device)

        if isinstance(policies, torch.Tensor):
            policies = policies.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            policies = torch.tensor(policies, dtype=torch.float32).to(self.device)

        if isinstance(values, torch.Tensor):
            values = values.detach().clone().to(dtype=torch.float32, device=self.device)
        else:
            values = torch.tensor(values, dtype=torch.float32).to(self.device)

        return states, policies, values
        
    def compute_loss(self, pred_policy_logits, pred_value, target_policy, target_value):
        """
        Computes the AlphaZero loss function:
        L = (z - v)^2 - pi^T * log(p) + c||theta||^2
        
        where:
        - (z - v)^2 is Mean Squared Error for value (z=target, v=pred)
        - -pi^T * log(p) is Cross Entropy for policy (pi=target, p=pred)
        - L2 regularization is handled by the optimizer's weight_decay
        """
        # Value Loss (MSE)
        # pred_value shape: (batch, 1), target_value shape: (batch,) -> unsqueeze target
        value_loss = nn.MSELoss()(pred_value.view(-1), target_value.view(-1))
        
        # Policy Loss (Cross Entropy)
        # pred_policy_logits: (batch, 8, 8, 73) -> flatten to (batch, 4672)
        # target_policy: (batch, 4672)
        # But wait, model output is (B, 8, 8, 73). We need to match shapes.
        # Let's verify target_policy shape from MCTS/ReplayBuffer.
        # Usually, target policy is probability distribution.
        
        # We use torch.sum(-target * log_pred) for cross entropy with soft targets
        pred_policy_log_probs = pred_policy_logits.view(pred_policy_logits.size(0), -1)
        target_policy_flat = target_policy.view(target_policy.size(0), -1)
        
        # Cross Entropy = - sum(target * log(pred))
        # pred_policy_logits are already log_softmax from the model
        policy_loss = -torch.sum(target_policy_flat * pred_policy_log_probs) / target_policy_flat.size(0)
        
        return value_loss, policy_loss

    def train_step(self, batch) -> Dict[str, float]:
        """
        Performs a single training step on a batch of data.
        """
        self.model.train()
        
        states, policies, values = self._prepare_batch(*batch)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_policies, pred_values = self.model(states)
        
        # Compute loss
        value_loss, policy_loss = self.compute_loss(pred_policies, pred_values, policies, values)
        total_loss = value_loss + policy_loss
        
        # Debug: log value statistics occasionally
        if hasattr(self, '_step_count'):
            self._step_count += 1
        else:
            self._step_count = 0
        
        if self._step_count % 50 == 0:  # Log every 50 steps
            pred_mean = pred_values.mean().item()
            pred_std = pred_values.std().item()
            target_mean = values.mean().item()
            target_std = values.std().item()
            logger.debug(
                f"Step {self._step_count}: Pred values: mean={pred_mean:.3f}, std={pred_std:.3f}, "
                f"Target values: mean={target_mean:.3f}, std={target_std:.3f}"
            )
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }

    def validate(self, replay_buffer: ReplayBuffer, batch_size: int = 64, num_batches: int = 20) -> Optional[Dict[str, float]]:
        """
        evaluate the model on a held-out replay buffer split to monitor overfitting.
        """
        if len(replay_buffer) < batch_size:
            logger.warning("not enough data in validation buffer to evaluate.")
            return None

        self.model.eval()
        totals = {"total_loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}

        with torch.no_grad():
            for _ in range(num_batches):
                batch = replay_buffer.sample(batch_size)
                states, policies, values = self._prepare_batch(*batch)
                pred_policies, pred_values = self.model(states)
                value_loss, policy_loss = self.compute_loss(pred_policies, pred_values, policies, values)
                totals["value_loss"] += value_loss.item()
                totals["policy_loss"] += policy_loss.item()
                totals["total_loss"] += (value_loss + policy_loss).item()

        for key in totals:
            totals[key] /= num_batches

        logger.info(
            f"validation complete. loss: {totals['total_loss']:.4f} "
            f"(pol: {totals['policy_loss']:.4f}, val: {totals['value_loss']:.4f})"
        )
        return totals

    def train_epoch(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 32,
        num_batches: int = 100,
        val_buffer: Optional[ReplayBuffer] = None,
        val_batch_size: Optional[int] = None,
        val_batches: int = 20,
    ) -> Optional[Dict[str, float]]:
        """
        runs training for a specified number of batches and optionally evaluates on a validation buffer.
        """
        if len(replay_buffer) < batch_size:
            logger.warning("Not enough data in replay buffer to train.")
            return None
            
        total_metrics = {"total_loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
        
        for _ in range(num_batches):
            batch = replay_buffer.sample(batch_size)
            metrics = self.train_step(batch)
            
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
                    
        # Average metrics
        for k in total_metrics:
            total_metrics[k] /= num_batches
            
        logger.info(f"Epoch Complete. Loss: {total_metrics['total_loss']:.4f} (Pol: {total_metrics['policy_loss']:.4f}, Val: {total_metrics['value_loss']:.4f})")

        if val_buffer is not None:
            batch_size_val = val_batch_size or batch_size
            val_metrics = self.validate(val_buffer, batch_size=batch_size_val, num_batches=val_batches)
            return {"train": total_metrics, "validation": val_metrics}

        return total_metrics

    def save_checkpoint(
        self,
        path: str = "checkpoint.pth",
        replay_buffer: Optional[ReplayBuffer] = None,
        metadata: Optional[Dict[str, float]] = None,
    ):
        save_ckpt(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            replay_buffer=replay_buffer,
            metadata=metadata,
        )
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(
        self,
        path: str,
        replay_buffer: Optional[ReplayBuffer] = None,
    ) -> Dict[str, float]:
        metadata = load_ckpt(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            replay_buffer=replay_buffer,
            map_location=self.device,
        )
        logger.info(f"Checkpoint loaded from {path}")
        return metadata

