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
from typing import Optional, Dict
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from model import ChessNet

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
        
        states, policies, values = batch
        
        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        policies = torch.tensor(policies, dtype=torch.float32).to(self.device)
        values = torch.tensor(values, dtype=torch.float32).to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward pass
        pred_policies, pred_values = self.model(states)
        
        # Compute loss
        value_loss, policy_loss = self.compute_loss(pred_policies, pred_values, policies, values)
        total_loss = value_loss + policy_loss
        
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

    def train_epoch(self, replay_buffer: ReplayBuffer, batch_size: int = 32, num_batches: int = 100):
        """
        Runs training for a specified number of batches.
        """
        if len(replay_buffer) < batch_size:
            logger.warning("Not enough data in replay buffer to train.")
            return
            
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
        return total_metrics

    def save_checkpoint(self, path: str = "checkpoint.pth"):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Checkpoint loaded from {path}")

