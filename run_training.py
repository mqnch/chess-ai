"""
run_training.py

Simple script to run the full training loop: generate self-play games, train the model,
and track progress. Designed to be run temporarily on any machine to see model growth.

Usage:
    python run_training.py

You can modify the parameters at the top of the script to adjust:
- Number of games per iteration
- Training batch size and batches per epoch
- MCTS simulations per move
- Device (cpu/cuda)
"""

import torch
import logging
import time
from pathlib import Path

from model import ChessNet
from game import SelfPlayConfig, generate_self_play_games
from replay_buffer import ReplayBuffer
from train import Trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
# Adjust these parameters based on available compute and time

# Training parameters
NUM_ITERATIONS = 10  # Number of training iterations (each: generate games + train)
GAMES_PER_ITERATION = 5  # Self-play games to generate per iteration
TRAIN_BATCH_SIZE = 16  # Batch size for training
TRAIN_BATCHES_PER_EPOCH = 50  # Number of training batches per epoch
VAL_SPLIT = 0.1  # Fraction of games to hold out for validation (0.0 to disable)

# MCTS parameters
MCTS_SIMULATIONS = 100  # Lower for faster games (50-200 recommended for quick testing)
MCTS_BATCH_SIZE = 4  # Batch size for neural network inference

# Model parameters
MODEL_CHANNELS = 128  # 128 for faster training, 256 for better quality
MODEL_RESIDUAL_BLOCKS = 6  # 6 for faster training, 8-10 for better quality

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAVE_CHECKPOINT_EVERY = 2  # Save checkpoint every N iterations

# ===== MAIN TRAINING LOOP =====

def main():
    logger.info("=" * 60)
    logger.info("Starting Chess AI Training")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Model: {MODEL_RESIDUAL_BLOCKS} residual blocks, {MODEL_CHANNELS} channels")
    logger.info(f"Config: {GAMES_PER_ITERATION} games/iter, {MCTS_SIMULATIONS} sims/move")
    logger.info("=" * 60)

    # Initialize model
    model = ChessNet(
        num_residual_blocks=MODEL_RESIDUAL_BLOCKS,
        num_channels=MODEL_CHANNELS
    )
    model.to(DEVICE)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=DEVICE,
        learning_rate=0.02,
        weight_decay=1e-4
    )
    
    # Initialize replay buffer
    replay_buffer = ReplayBuffer(capacity=50_000)  # Store up to 50k samples
    val_buffer = ReplayBuffer(capacity=10_000) if VAL_SPLIT > 0 else None
    
    # Self-play configuration
    self_play_config = SelfPlayConfig(
        num_simulations=MCTS_SIMULATIONS,
        batch_size=MCTS_BATCH_SIZE,
        temperature_initial=1.25,
        temperature_final=0.1,
        temperature_switch_move=30,
        max_moves=400,
    )
    
    total_games = 0
    total_samples = 0
    start_time = time.time()
    
    try:
        for iteration in range(1, NUM_ITERATIONS + 1):
            iter_start = time.time()
            logger.info("")
            logger.info(f"--- Iteration {iteration}/{NUM_ITERATIONS} ---")
            
            # ===== PHASE 1: Generate Self-Play Games =====
            logger.info(f"Generating {GAMES_PER_ITERATION} self-play games...")
            game_start = time.time()
            
            all_trajectories = generate_self_play_games(
                model=model,
                num_games=GAMES_PER_ITERATION,
                config=self_play_config,
                device=DEVICE,
                replay_buffer=replay_buffer,
            )
            
            games_generated = len(all_trajectories)
            samples_generated = sum(len(traj) for traj in all_trajectories)
            total_games += games_generated
            total_samples += samples_generated
            
            game_time = time.time() - game_start
            logger.info(
                f"Generated {games_generated} games ({samples_generated} samples) "
                f"in {game_time:.1f}s ({game_time/games_generated:.1f}s/game)"
            )
            
            # Split validation data if enabled
            if val_buffer is not None and samples_generated > 0:
                val_samples = int(samples_generated * VAL_SPLIT)
                if val_samples > 0:
                    # Take last val_samples from the buffer (most recent games)
                    # This is a simple approach; in production you might want more sophisticated splitting
                    recent_samples = []
                    for traj in all_trajectories[-val_samples:]:
                        recent_samples.extend(traj)
                    if len(recent_samples) > val_samples:
                        recent_samples = recent_samples[-val_samples:]
                    val_buffer.extend(recent_samples)
                    logger.info(f"Added {len(recent_samples)} samples to validation buffer")
            
            # ===== PHASE 2: Train on Replay Buffer =====
            if len(replay_buffer) < TRAIN_BATCH_SIZE:
                logger.warning(f"Not enough samples ({len(replay_buffer)}) to train. Skipping.")
                continue
            
            logger.info(f"Training on {len(replay_buffer)} samples...")
            train_start = time.time()
            
            metrics = trainer.train_epoch(
                replay_buffer=replay_buffer,
                batch_size=TRAIN_BATCH_SIZE,
                num_batches=TRAIN_BATCHES_PER_EPOCH,
                val_buffer=val_buffer,
                val_batches=10 if val_buffer and len(val_buffer) > 0 else 0,
            )
            
            train_time = time.time() - train_start
            
            if metrics:
                if isinstance(metrics, dict) and "train" in metrics:
                    # Validation was run
                    train_metrics = metrics["train"]
                    val_metrics = metrics.get("validation")
                    logger.info(
                        f"Training complete in {train_time:.1f}s. "
                        f"Train Loss: {train_metrics['total_loss']:.4f} "
                        f"(Pol: {train_metrics['policy_loss']:.4f}, "
                        f"Val: {train_metrics['value_loss']:.4f})"
                    )
                    if val_metrics:
                        logger.info(
                            f"Validation Loss: {val_metrics['total_loss']:.4f} "
                            f"(Pol: {val_metrics['policy_loss']:.4f}, "
                            f"Val: {val_metrics['value_loss']:.4f})"
                        )
                else:
                    # No validation
                    logger.info(
                        f"Training complete in {train_time:.1f}s. "
                        f"Loss: {metrics['total_loss']:.4f} "
                        f"(Pol: {metrics['policy_loss']:.4f}, "
                        f"Val: {metrics['value_loss']:.4f})"
                    )
            
            # ===== PHASE 3: Save Checkpoint =====
            if iteration % SAVE_CHECKPOINT_EVERY == 0:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_iter_{iteration}.pth"
                trainer.save_checkpoint(str(checkpoint_path))
            
            iter_time = time.time() - iter_start
            logger.info(f"Iteration {iteration} complete in {iter_time:.1f}s")
            
            # Progress summary
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {total_games} games, {total_samples} samples, "
                f"{elapsed/60:.1f} min elapsed"
            )
    
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Saving final checkpoint...")
        final_checkpoint = CHECKPOINT_DIR / "checkpoint_final.pth"
        trainer.save_checkpoint(str(final_checkpoint))
        logger.info(f"Final checkpoint saved to {final_checkpoint}")
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average: {total_time/total_games:.1f}s per game" if total_games > 0 else "")
    logger.info(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

