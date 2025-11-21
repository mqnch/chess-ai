"""
eval.py

Implements the Arena class for evaluating neural networks against each other.
This module manages the process of pitting a 'challenger' model against the
current 'best' model to determine if the challenger has improved.

References:
    - Project Vision: "Core Components > eval.py"
    - "Evaluator (eval.py): write an arena that pits two networks against each other"
"""

import time
import logging
import torch
import numpy as np
from typing import Dict, Tuple, Any
from game import ChessGame
from mcts import MCTS, BatchInference

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Arena:
    """
    Manages matches between two neural networks to evaluate their relative strength.
    """

    def __init__(self, best_model: torch.nn.Module, challenger_model: torch.nn.Module, 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 num_mcts_sims: int = 400):
        """
        Args:
            best_model: The current champion model.
            challenger_model: The new model to evaluate.
            device: Device to run inference on.
            num_mcts_sims: Number of MCTS simulations per move during evaluation.
        """
        self.best_model = best_model
        self.challenger_model = challenger_model
        self.device = device
        self.num_mcts_sims = num_mcts_sims

    def play_game(self) -> int:
        """
        Plays a single game between the two models.
        
        Returns:
            1 if Challenger wins (playing White or Black)
            -1 if Best Model wins
            0 if Draw
        """
        # Randomize who plays White
        # If challenger_is_white = True, Challenger is White (1), Best is Black (-1)
        challenger_is_white = np.random.choice([True, False])
        
        # Setup inference services
        # We use separate MCTS instances for each player to avoid state pollution
        # Ideally, we could share BatchInference if it supports multiple models, 
        # but for simplicity and safety, we create one for each model.
        # Note: In a highly optimized setup, we might batch across games.
        
        best_inference = BatchInference(self.best_model, batch_size=1, device=self.device)
        challenger_inference = BatchInference(self.challenger_model, batch_size=1, device=self.device)
        
        best_mcts = MCTS(best_inference)
        challenger_mcts = MCTS(challenger_inference)
        
        game = ChessGame()
        
        step = 0
        start_time = time.time()
        
        while not game.is_terminal():
            step += 1
            state = game # game object acts as state
            
            # Determine current player
            # game.get_turn() returns True for White, False for Black
            is_white_turn = (game.get_turn() == 1) # Assuming chess.WHITE is treated as truthy/1
            
            if is_white_turn == challenger_is_white:
                # Challenger's turn
                mcts = challenger_mcts
            else:
                # Best model's turn
                mcts = best_mcts
            
            # Run MCTS
            # For evaluation, we typically use a low temperature (deterministic-ish) 
            # to measure true strength, or a small temp like 0.1
            root = mcts.search(state, num_simulations=self.num_mcts_sims)
            actions, probs = mcts.get_action_prob(root, temperature=0.1)
            
            # Select move
            # With low temp, this picks the most visited move
            action_idx = np.random.choice(len(actions), p=probs)
            move = actions[action_idx]
            
            game.make_move(move)
            
            # Log progress occasionally
            if step % 20 == 0:
                logger.debug(f"Move {step} completed.")

        # Cleanup threads
        best_inference.stop()
        challenger_inference.stop()
        
        # Determine result
        # game.get_winner() returns 1 (White), -1 (Black), 0 (Draw)
        winner = game.get_winner()
        
        if winner == 0:
            return 0
        
        # If Challenger was White (True) and White won (1) -> Challenger wins (1)
        # If Challenger was White (True) and Black won (-1) -> Best wins (-1)
        # If Challenger was Black (False) and White won (1) -> Best wins (-1)
        # If Challenger was Black (False) and Black won (-1) -> Challenger wins (1)
        
        # Logic: if (challenger_is_white and winner == 1) or (not challenger_is_white and winner == -1):
        # Simplifies to: winner * (1 if challenger_is_white else -1)
        
        result_for_challenger = winner * (1 if challenger_is_white else -1)
        return result_for_challenger

    def evaluate(self, num_games: int = 20, update_threshold: float = 0.55) -> Tuple[bool, Dict[str, float]]:
        """
        Plays a series of games and decides if the challenger should replace the best model.
        
        Args:
            num_games: Total games to play.
            update_threshold: Fraction of wins (scoring) needed to replace best model.
                             Usually computed as (wins + 0.5 * draws) / total_games > threshold
                             OR just win rate. AlphaZero uses (wins + draws/2) / total > 0.55 usually.
        
        Returns:
            Tuple (accepted, metrics)
            accepted: True if challenger is better
            metrics: Dictionary of stats (win_rate, draw_rate, etc.)
        """
        logger.info(f"Starting evaluation: {num_games} games...")
        
        challenger_wins = 0
        best_wins = 0
        draws = 0
        game_lengths = []
        start_time = time.time()
        
        for i in range(num_games):
            game_start = time.time()
            result = self.play_game()
            duration = time.time() - game_start
            game_lengths.append(duration)
            
            if result == 1:
                challenger_wins += 1
            elif result == -1:
                best_wins += 1
            else:
                draws += 1
                
            logger.info(f"Game {i+1}/{num_games} finished. Result: {result} (1=Challenger, -1=Best, 0=Draw). Duration: {duration:.1f}s")
            
        total_time = time.time() - start_time
        avg_game_len = np.mean(game_lengths) if game_lengths else 0
        
        # Calculate score: Wins count as 1, Draws as 0.5 (optional, or just strict wins)
        # Vision.md says ">55%", usually implies strict win rate or score rate.
        # Let's use Score Rate: (Wins + 0.5 * Draws) / Total
        
        score = challenger_wins + 0.5 * draws
        win_rate = challenger_wins / num_games
        score_rate = score / num_games
        draw_rate = draws / num_games
        
        metrics = {
            "challenger_wins": challenger_wins,
            "best_wins": best_wins,
            "draws": draws,
            "win_rate": win_rate,
            "score_rate": score_rate,
            "draw_rate": draw_rate,
            "avg_game_length_s": avg_game_len,
            "total_eval_time_s": total_time
        }
        
        logger.info(f"Evaluation Complete. Score Rate: {score_rate:.2%} (Threshold: {update_threshold:.2%})")
        logger.info(f"Metrics: {metrics}")
        
        accepted = score_rate > update_threshold
        return accepted, metrics

