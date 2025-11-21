
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import chess

import numpy as np
import torch

from board import ChessGame
from encode import board_to_tensor, move_to_action
from mcts import BatchInference, MCTS, DEFAULT_C_PUCT, DEFAULT_DIRICHLET_ALPHA, DEFAULT_DIRICHLET_EPSILON

if TYPE_CHECKING:
    from replay_buffer import ReplayBuffer


@dataclass
class SelfPlayConfig:
    """configuration values for running self-play games."""

    num_simulations: int = 200
    batch_size: int = 8
    c_puct: float = DEFAULT_C_PUCT
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    temperature_initial: float = 1.25
    temperature_final: float = 0.1
    temperature_switch_move: int = 30
    max_moves: int = 512
    resignation_threshold: Optional[float] = None
    resignation_consecutive_moves: int = 2
    add_root_noise: bool = True


@dataclass
class SelfPlaySample:
    """single training sample captured during self-play."""

    state: torch.Tensor  # shape (18, 8, 8)
    policy: torch.Tensor  # shape (8, 8, 73)
    value: float


class SelfPlayGame:
    """coordinates self-play games using mcts and the neural network."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        config: Optional[SelfPlayConfig] = None,
    ):
        self.config = config or SelfPlayConfig()
        self.device = torch.device(device)
        self.inference = BatchInference(
            model=model,
            batch_size=self.config.batch_size,
            device=device,
        )
        self.mcts = MCTS(
            inference_service=self.inference,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """stop background inference threads."""
        self.inference.stop()

    def run_game(self) -> List[SelfPlaySample]:
        """run a complete self-play game and return recorded samples."""
        game = ChessGame()
        move_index = 0
        resign_counter = 0
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]] = []
        forced_result: Optional[int] = None

        while not game.is_terminal() and move_index < self.config.max_moves:
            board_snapshot = game.get_board().copy()
            state_tensor = board_to_tensor(board_snapshot)
            current_player = 1 if game.get_turn() else -1

            root_state = game.copy()
            root = self.mcts.search(
                root=root_state,
                num_simulations=self.config.num_simulations,
            )

            temperature = self._temperature_for_move(move_index)
            moves, probs = self.mcts.get_action_prob(root, temperature=temperature)

            if not moves:
                break

            policy_tensor = self._build_policy_tensor(
                moves=moves,
                probs=probs,
                board=board_snapshot,
            )

            selected_move = self._select_move(moves, probs)
            if selected_move is None:
                break

            samples.append((state_tensor, policy_tensor, current_player))

            if (
                self.config.resignation_threshold is not None
                and root.value <= self.config.resignation_threshold
            ):
                resign_counter += 1
            else:
                resign_counter = 0

            if (
                self.config.resignation_threshold is not None
                and resign_counter >= self.config.resignation_consecutive_moves
            ):
                forced_result = -current_player
                break

            game.make_move(selected_move)
            move_index += 1

        if forced_result is not None:
            result = forced_result
        elif game.is_terminal():
            result = game.get_winner() or 0
        else:
            result = 0  # reached max move cap; treat as draw

        trajectories: List[SelfPlaySample] = []
        for state_tensor, policy_tensor, player in samples:
            value = float(result * player)
            trajectories.append(
                SelfPlaySample(
                    state=state_tensor,
                    policy=policy_tensor,
                    value=value,
                )
            )

        return trajectories

    def _temperature_for_move(self, move_index: int) -> float:
        """determine temperature value for the given move index."""
        if move_index < self.config.temperature_switch_move:
            return self.config.temperature_initial
        return self.config.temperature_final

    def _build_policy_tensor(
        self,
        moves: List[chess.Move],
        probs: List[float],
        board: chess.Board,
    ) -> torch.Tensor:
        """convert move probabilities into the 8x8x73 tensor expected by the network."""
        policy_tensor = torch.zeros(8, 8, 73, dtype=torch.float32)
        total_prob = 0.0

        for move, prob in zip(moves, probs):
            action = move_to_action(move, board)
            if action is None:
                continue
            from_square, action_type = action
            row = from_square // 8
            col = from_square % 8
            policy_tensor[row, col, action_type] = prob
            total_prob += prob

        if total_prob > 0:
            policy_tensor /= total_prob

        return policy_tensor

    def _select_move(
        self,
        moves: List[chess.Move],
        probs: List[float],
    ) -> Optional[chess.Move]:
        """sample a move from the visit distribution."""
        if not moves or not probs:
            return None

        prob_array = np.array(probs, dtype=np.float64)
        prob_sum = prob_array.sum()
        if not math.isfinite(prob_sum) or prob_sum <= 0:
            prob_array = np.ones(len(moves), dtype=np.float64)
            prob_sum = prob_array.sum()

        prob_array /= prob_sum

        index = np.random.choice(len(moves), p=prob_array)
        return moves[index]


def generate_self_play_games(
    model: torch.nn.Module,
    num_games: int,
    config: Optional[SelfPlayConfig] = None,
    device: str = "cpu",
    replay_buffer: Optional["ReplayBuffer"] = None,
) -> List[List[SelfPlaySample]]:
    """
    helper function that runs multiple self-play games and optionally writes the
    resulting samples to a replay buffer.
    """
    trajectories: List[List[SelfPlaySample]] = []
    config = config or SelfPlayConfig()

    with SelfPlayGame(model=model, device=device, config=config) as runner:
        for _ in range(num_games):
            samples = runner.run_game()
            if not samples:
                continue
            trajectories.append(samples)
            if replay_buffer is not None:
                replay_buffer.extend(samples)

    return trajectories

