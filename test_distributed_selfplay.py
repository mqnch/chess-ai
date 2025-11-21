import unittest

import torch

from distributed_selfplay import run_distributed_self_play
from game import SelfPlayConfig
from model import ChessNet


class TestDistributedSelfPlay(unittest.TestCase):
    def test_multi_worker_self_play(self):
        model = ChessNet(num_residual_blocks=1, num_channels=16)
        config = SelfPlayConfig(
            num_simulations=8,
            batch_size=2,
            temperature_initial=1.25,
            temperature_final=0.5,
            temperature_switch_move=5,
            max_moves=40,
        )
        trajectories = run_distributed_self_play(
            model=model,
            config=config,
            total_games=2,
            device="cpu",
            num_workers=2,
            max_batch_size=2,
        )
        self.assertGreaterEqual(len(trajectories), 1)


if __name__ == "__main__":
    unittest.main()

