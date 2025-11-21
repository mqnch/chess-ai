"""
replay_buffer.py

Implements the ReplayBuffer for storing self-play data.
Stores experience tuples (state_tensor, policy_dist, value) to be sampled during training.

References:
    - Project Vision: "Core Components > replay_buffer.py"
"""

import numpy as np
from collections import deque
import random
from typing import List, Tuple, Any

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Args:
            capacity: Maximum number of game steps (positions) to store.
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state_tensor: np.ndarray, policy: np.ndarray, value: float):
        """
        Add a sample to the buffer.
        
        Args:
            state_tensor: Board representation (C, H, W).
            policy: Probability distribution over moves.
            value: Final game outcome (or estimated value) for this state.
        """
        self.buffer.append((state_tensor, policy, value))
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences.
        
        Returns:
            Tuple (states, policies, values) as numpy arrays.
        """
        batch = random.sample(self.buffer, batch_size)
        state_batch, policy_batch, value_batch = zip(*batch)
        
        return (
            np.array(state_batch), 
            np.array(policy_batch), 
            np.array(value_batch)
        )
    
    def __len__(self):
        return len(self.buffer)

