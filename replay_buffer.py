import random
import threading
from typing import Dict, Iterable, List, Optional

import torch

from game import SelfPlaySample


class ReplayBuffer:
    """thread-safe fixed-size replay buffer for storing self-play samples."""

    def __init__(self, capacity: int = 200_000, seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.storage: List[SelfPlaySample] = []
        self.next_index = 0
        self.size = 0
        self.lock = threading.Lock()
        self.random = random.Random(seed)

    def __len__(self) -> int:
        with self.lock:
            return self.size

    def clear(self):
        """remove all stored samples."""
        with self.lock:
            self.storage.clear()
            self.next_index = 0
            self.size = 0

    def add(self, sample: SelfPlaySample):
        """add a single sample to the buffer."""
        with self.lock:
            if len(self.storage) < self.capacity:
                self.storage.append(sample)
            else:
                self.storage[self.next_index] = sample
            self.next_index = (self.next_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def extend(self, samples: Iterable[SelfPlaySample]):
        """add multiple samples to the buffer."""
        for sample in samples:
            self.add(sample)

    def state_dict(self) -> Dict:
        """serialize buffer contents for checkpointing."""
        with self.lock:
            return {
                "capacity": self.capacity,
                "next_index": self.next_index,
                "size": self.size,
                "storage": [
                    {
                        "state": sample.state.clone(),
                        "policy": sample.policy.clone(),
                        "value": sample.value,
                    }
                    for sample in self.storage
                ],
            }

    def load_state_dict(self, state: Dict):
        """restore buffer contents from serialized data."""
        with self.lock:
            self.capacity = state.get("capacity", self.capacity)
            self.next_index = state.get("next_index", 0)
            self.size = state.get("size", 0)
            self.storage = []
            for item in state.get("storage", []):
                sample = SelfPlaySample(
                    state=item["state"].clone(),
                    policy=item["policy"].clone(),
                    value=float(item["value"]),
                )
                self.storage.append(sample)
            # clamp values
            self.size = min(self.size, len(self.storage), self.capacity)
            self.next_index = self.size % self.capacity

    def save(self, path: str):
        """persist buffer to disk."""
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = None):
        """load buffer from disk."""
        state = torch.load(path, map_location=map_location or "cpu")
        self.load_state_dict(state)

    def sample(
        self,
        batch_size: int,
        *,
        device: Optional[str] = None,
        as_tensors: bool = True,
    ):
        """
        draw a random batch of samples.

        returns either a tuple of torch tensors (states, policies, values) if
        as_tensors is true, or a list of SelfPlaySample objects if false.
        """
        with self.lock:
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if batch_size > self.size:
                raise ValueError("not enough samples to draw from replay buffer")

            indices = self.random.sample(range(self.size), batch_size)
            batch = [self.storage[i] for i in indices]

        if not as_tensors:
            return batch

        device = device or "cpu"

        states = torch.stack([sample.state.clone() for sample in batch]).to(device)
        policies = torch.stack([sample.policy.clone() for sample in batch]).to(device)
        values = (
            torch.tensor([sample.value for sample in batch], dtype=torch.float32)
            .unsqueeze(1)
            .to(device)
        )

        return states, policies, values

