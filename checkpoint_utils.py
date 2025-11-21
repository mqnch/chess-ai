"""
checkpoint_utils.py

Helper functions for saving and loading training checkpoints, including model,
optimizer, scheduler, replay buffer, and metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from replay_buffer import ReplayBuffer


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    replay_buffer: Optional["ReplayBuffer"] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if replay_buffer is not None:
        checkpoint["replay_buffer"] = replay_buffer.state_dict()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    replay_buffer: Optional["ReplayBuffer"] = None,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location or "cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if replay_buffer is not None and "replay_buffer" in checkpoint:
        replay_buffer.load_state_dict(checkpoint["replay_buffer"])

    return checkpoint.get("metadata", {})

