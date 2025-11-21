import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from train import Trainer
from model import ChessNet
from replay_buffer import ReplayBuffer
from game import SelfPlaySample

class TestTrainer(unittest.TestCase):
    """
    Unit tests for the Trainer class in train.py.
    """

    def setUp(self):
        self.device = 'cpu'
        self.model = ChessNet(num_residual_blocks=2, num_channels=16)  # Small model for speed
        self.trainer = Trainer(self.model, device=self.device, learning_rate=0.01)
        self.replay_buffer = ReplayBuffer(capacity=100)

        # Populate replay buffer with some dummy data
        for _ in range(10):
            state = torch.randn(18, 8, 8)
            policy = torch.randn(8, 8, 73)
            policy = torch.softmax(policy.view(-1), dim=0).view(8, 8, 73)
            value = 0.5
            sample = SelfPlaySample(state=state, policy=policy, value=value)
            self.replay_buffer.add(sample)

    def test_trainer_initialization(self):
        """Test that the trainer initializes correctly."""
        self.assertIsInstance(self.trainer.model, ChessNet)
        self.assertIsInstance(self.trainer.optimizer, optim.SGD)
        self.assertIsInstance(self.trainer.scheduler, optim.lr_scheduler.StepLR)
        self.assertEqual(str(self.trainer.device), 'cpu')

    def test_compute_loss(self):
        """Test loss computation."""
        batch_size = 4
        # Simulate model output (log probabilities)
        pred_policy_logits = torch.randn(batch_size, 8, 8, 73)
        pred_policy_logits = torch.log_softmax(pred_policy_logits.view(batch_size, -1), dim=1).view(batch_size, 8, 8, 73)
        
        pred_value = torch.randn(batch_size, 1)
        target_policy = torch.randn(batch_size, 4672)
        target_policy = torch.softmax(target_policy, dim=1) # Normalize
        target_value = torch.randn(batch_size, 1)

        value_loss, policy_loss = self.trainer.compute_loss(
            pred_policy_logits, pred_value, target_policy, target_value
        )

        self.assertIsInstance(value_loss, torch.Tensor)
        self.assertIsInstance(policy_loss, torch.Tensor)
        self.assertTrue(value_loss.item() >= 0)
        # Policy loss can be negative if not strictly CE (depends on implementation specifics), 
        # but usually CE is positive. 
        # Implementation uses: -sum(target * log_pred). 
        # If log_pred comes from log_softmax, values are negative. 
        # target * negative = negative. -sum(negative) = positive.
        self.assertTrue(policy_loss.item() >= 0)

    def test_train_step(self):
        """Test a single training step."""
        # Sample a batch from the replay buffer
        # Note: ReplayBuffer.sample returns (states, policies, values) tensors
        batch = self.replay_buffer.sample(batch_size=4, device=self.device)
        
        # Flatten policies for the trainer (Trainer expects (B, 4672))
        # But wait, Trainer.compute_loss expects:
        # pred_policy_logits: (B, 8, 8, 73)
        # target_policy: (B, 4672)
        # But model output is (B, 8, 8, 73). 
        # ReplayBuffer stores (8, 8, 73).
        # Let's check ReplayBuffer output shape.
        # policies shape is (B, 8, 8, 73).
        # Trainer.compute_loss:
        # target_policy_flat = target_policy.view(target_policy.size(0), -1)
        # So Trainer handles reshaping internally if target_policy is passed as (B, 8, 8, 73) too.
        
        metrics = self.trainer.train_step(batch)
        
        self.assertIn("total_loss", metrics)
        self.assertIn("value_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertIn("lr", metrics)
        self.assertIsInstance(metrics["total_loss"], float)

    def test_train_epoch(self):
        """Test running a training epoch."""
        # We need enough data in buffer
        while len(self.replay_buffer) < 8:
             state = torch.randn(18, 8, 8)
             policy = torch.randn(8, 8, 73)
             value = 0.1
             sample = SelfPlaySample(state=state, policy=policy, value=value)
             self.replay_buffer.add(sample)

        metrics = self.trainer.train_epoch(self.replay_buffer, batch_size=4, num_batches=2)
        
        self.assertIsNotNone(metrics)
        if metrics:
            self.assertIn("total_loss", metrics)
            self.assertIn("value_loss", metrics)
            self.assertIn("policy_loss", metrics)
        
    def test_train_epoch_not_enough_data(self):
        """Test training epoch gracefully handles insufficient data."""
        empty_buffer = ReplayBuffer(capacity=100)
        metrics = self.trainer.train_epoch(empty_buffer, batch_size=32, num_batches=1)
        self.assertIsNone(metrics)

    def test_replay_buffer_state_dict(self):
        """Replay buffer can round-trip through state dict."""
        state = self.replay_buffer.state_dict()
        clone = ReplayBuffer(capacity=state["capacity"])
        clone.load_state_dict(state)
        self.assertEqual(len(clone), len(self.replay_buffer))

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints with replay buffer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pth")
            metadata = {"iteration": 3, "total_games": 12}
            self.trainer.save_checkpoint(
                checkpoint_path, replay_buffer=self.replay_buffer, metadata=metadata
            )
            self.assertTrue(os.path.exists(checkpoint_path))
            loaded_meta = self.trainer.load_checkpoint(
                checkpoint_path, replay_buffer=self.replay_buffer
            )
            self.assertEqual(metadata["iteration"], loaded_meta.get("iteration"))

if __name__ == '__main__':
    unittest.main()

