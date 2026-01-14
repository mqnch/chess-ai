"""
run_training.py

End-to-end training loop for the AlphaZero-style chess engine. Supports
single-process or distributed self-play, checkpointing, and configuration
overrides for quick experimentation or remote jobs (e.g., WATCLOUD).
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from model import ChessNet
from game import SelfPlayConfig, SelfPlaySample, generate_self_play_games
from replay_buffer import ReplayBuffer
from train import Trainer
from distributed_selfplay import run_distributed_self_play

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device setup
if torch.cuda.is_available():
    NUM_GPUS = torch.cuda.device_count()
    DEVICE = "cuda"
    DEVICES = [f"cuda:{i}" for i in range(NUM_GPUS)]
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
    DEVICES = ["mps"]
    NUM_GPUS = 1
else:
    DEVICE = "cpu"
    DEVICES = ["cpu"]
    NUM_GPUS = 0

# ===== CONFIGURATION =====
# Adjust these parameters based on available compute and time

# Training parameters
NUM_ITERATIONS = 100  # More iterations for deep learning
GAMES_PER_ITERATION = 100  # More games to keep 4x3090 busy
TRAIN_BATCH_SIZE = 512  # 128 per GPU
TRAIN_BATCHES_PER_EPOCH = 400
VAL_SPLIT = 0.1

# MCTS parameters
MCTS_SIMULATIONS = 400
MCTS_BATCH_SIZE = 16

# Distributed self-play
NUM_SELFPLAY_WORKERS = 48 # Scaled for Threadripper 64-cores
DISTRIBUTED_MAX_BATCH = 64
INFERENCE_DEVICE = DEVICES  # Use all GPUs for inference

# Model parameters
MODEL_CHANNELS = 256
MODEL_RESIDUAL_BLOCKS = 10

# Loss balancing
VALUE_LOSS_WEIGHT = 1.0

# Checkpointing
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAVE_CHECKPOINT_EVERY = 2


def _train_worker(rank, world_size, model_config, state_dict, batch_data, settings, results_dict):
    """
    Worker process for distributed training using DDP.
    """
    import os
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import TensorDataset, DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = ChessNet(
        num_residual_blocks=model_config["num_residual_blocks"],
        num_channels=model_config["num_channels"],
    ).to(device)
    model.load_state_dict(state_dict)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        device=str(device),
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        value_loss_weight=settings.get("value_loss_weight", 1.0),
    )
    
    # Prepare data
    states, policies, values = batch_data
    dataset = TensorDataset(states, policies, values)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=settings["train_batch_size"], sampler=sampler, num_workers=0, pin_memory=True)
    
    # Training loop for this epoch
    model.train()
    total_metrics = {"total_loss": 0.0, "value_loss": 0.0, "policy_loss": 0.0}
    num_batches = 0
    
    # We want to run for settings["train_batches_per_epoch"] total batches across all GPUs
    # so each GPU runs for settings["train_batches_per_epoch"] // world_size
    batches_to_run = settings["train_batches_per_epoch"] // world_size
    
    for _ in range((batches_to_run // len(dataloader)) + 1):
        for batch in dataloader:
            if num_batches >= batches_to_run:
                break
            metrics = trainer.train_step(batch)
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            num_batches += 1
        if num_batches >= batches_to_run:
            break
            
    # Average metrics
    if num_batches > 0:
        for k in total_metrics:
            total_metrics[k] /= num_batches
            
    # Gather metrics (simplified: only rank 0 returns)
    if rank == 0:
        results_dict["metrics"] = total_metrics
        # Save state dict from rank 0 to return to main process
        results_dict["state_dict"] = {k: v.cpu() for k, v in model.module.state_dict().items()}
        
    dist.destroy_process_group()


def build_default_settings() -> Dict[str, Any]:
    return {
        "num_iterations": NUM_ITERATIONS,
        "games_per_iteration": GAMES_PER_ITERATION,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "train_batches_per_epoch": TRAIN_BATCHES_PER_EPOCH,
        "val_split": VAL_SPLIT,
        "mcts_simulations": MCTS_SIMULATIONS,
        "mcts_batch_size": MCTS_BATCH_SIZE,
        "model_channels": MODEL_CHANNELS,
        "model_residual_blocks": MODEL_RESIDUAL_BLOCKS,
        "learning_rate": 0.02,
        "weight_decay": 1e-4,
        "value_loss_weight": VALUE_LOSS_WEIGHT,
        "replay_buffer_capacity": 200_000,
        "val_buffer_capacity": 20_000,
        "temperature_initial": 1.25,
        "temperature_final": 0.1,
        "temperature_switch_move": 30,
        "max_moves": 400,
        "num_selfplay_workers": NUM_SELFPLAY_WORKERS,
        "distributed_max_batch": DISTRIBUTED_MAX_BATCH,
        "inference_device": INFERENCE_DEVICE,
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "save_checkpoint_every": SAVE_CHECKPOINT_EVERY,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlphaZero-style chess training loop")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, help="Path to JSON config overriding defaults")
    parser.add_argument("--num-workers", type=int, help="Override number of self-play workers")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ===== MAIN TRAINING LOOP =====

def main():
    args = parse_args()
    settings = build_default_settings()
    if args.config:
        settings.update(load_config(args.config))
    if args.num_workers is not None:
        settings["num_selfplay_workers"] = args.num_workers

    training_device = settings.get("device", DEVICE)
    inference_device = settings.get("inference_device") or training_device
    checkpoint_dir = Path(settings.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_iterations = settings["num_iterations"]
    games_per_iteration = settings["games_per_iteration"]
    train_batch_size = settings["train_batch_size"]
    train_batches = settings["train_batches_per_epoch"]
    val_split = settings["val_split"]
    num_workers = settings["num_selfplay_workers"]
    save_every = settings["save_checkpoint_every"]

    logger.info("=" * 60)
    logger.info("Starting Chess AI Training")
    logger.info("=" * 60)
    logger.info(f"Device: {training_device}")
    logger.info(
        f"Model: {settings['model_residual_blocks']} residual blocks, "
        f"{settings['model_channels']} channels"
    )
    logger.info(
        f"Config: {games_per_iteration} games/iter, "
        f"{settings['mcts_simulations']} sims/move, "
        f"{num_workers} workers"
    )
    logger.info(
        f"Loss weights -> value: {settings.get('value_loss_weight', 1.0):.2f}"
    )
    logger.info("=" * 60)

    model = ChessNet(
        num_residual_blocks=settings["model_residual_blocks"],
        num_channels=settings["model_channels"],
    )
    model.to(training_device)

    trainer = Trainer(
        model=model,
        device=training_device,
        learning_rate=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
        value_loss_weight=settings.get("value_loss_weight", 1.0),
    )

    replay_buffer = ReplayBuffer(capacity=settings["replay_buffer_capacity"])
    val_buffer = (
        ReplayBuffer(capacity=settings["val_buffer_capacity"]) if val_split > 0 else None
    )

    self_play_config = SelfPlayConfig(
        num_simulations=settings["mcts_simulations"],
        batch_size=settings["mcts_batch_size"],
        temperature_initial=settings["temperature_initial"],
        temperature_final=settings["temperature_final"],
        temperature_switch_move=settings["temperature_switch_move"],
        max_moves=settings["max_moves"],
        c_puct=settings.get("c_puct", 1.0),
        dirichlet_alpha=settings.get("dirichlet_alpha", 0.3),
        dirichlet_epsilon=settings.get("dirichlet_epsilon", 0.25),
    )

    total_games = 0
    total_samples = 0
    start_iteration = 1

    if args.resume:
        metadata = trainer.load_checkpoint(args.resume, replay_buffer=replay_buffer)
        start_iteration = metadata.get("iteration", 0) + 1
        total_games = metadata.get("total_games", 0)
        total_samples = metadata.get("total_samples", 0)
        logger.info(
            f"Resuming from iteration {start_iteration} "
            f"(games={total_games}, samples={total_samples})"
        )

    if start_iteration > num_iterations:
        logger.info("All iterations already completed. Exiting.")
        return

    start_time = time.time()
    iteration = start_iteration - 1

    try:
        for iteration in range(start_iteration, num_iterations + 1):
            iter_start = time.time()
            logger.info("")
            logger.info(f"--- Iteration {iteration}/{num_iterations} ---")

            # ===== PHASE 1: Generate Self-Play Games =====
            logger.info(f"Generating {games_per_iteration} self-play games...")
            game_start = time.time()

            if num_workers > 1:
                logger.info(
                    f"Using distributed self-play ({num_workers} workers, inference on {inference_device})"
                )
                all_trajectories = run_distributed_self_play(
                    model=model,
                    config=self_play_config,
                    total_games=games_per_iteration,
                    device=training_device,
                    num_workers=num_workers,
                    inference_device=inference_device,
                    max_batch_size=settings["distributed_max_batch"],
                )
                for traj in all_trajectories:
                    if traj:
                        replay_buffer.extend(traj)
            else:
                all_trajectories = generate_self_play_games(
                    model=model,
                    num_games=games_per_iteration,
                    config=self_play_config,
                    device=training_device,
                    replay_buffer=replay_buffer,
                )

            games_generated = len(all_trajectories)
            samples_generated = sum(len(traj) for traj in all_trajectories)
            total_games += games_generated
            total_samples += samples_generated

            game_time = time.time() - game_start
            avg_time = game_time / games_generated if games_generated else 0.0
            logger.info(
                f"Generated {games_generated} games ({samples_generated} samples) "
                f"in {game_time:.1f}s ({avg_time:.1f}s/game)"
            )

            if val_buffer is not None and samples_generated > 0:
                val_samples = int(samples_generated * val_split)
                if val_samples > 0:
                    recent_samples: List[SelfPlaySample] = []
                    for traj in all_trajectories[-val_samples:]:
                        recent_samples.extend(traj)
                    if len(recent_samples) > val_samples:
                        recent_samples = recent_samples[-val_samples:]
                    val_buffer.extend(recent_samples)
                    logger.info(f"Added {len(recent_samples)} samples to validation buffer")

            # ===== PHASE 2: Train on Replay Buffer =====
            if len(replay_buffer) < train_batch_size:
                logger.warning(
                    f"Not enough samples ({len(replay_buffer)}) to train. Skipping."
                )
                continue

            logger.info(f"Training on {len(replay_buffer)} samples across {NUM_GPUS or 1} GPUs...")
            train_start = time.time()

            if NUM_GPUS > 1:
                # Prepare data for all workers
                states, policies, values = replay_buffer.sample(len(replay_buffer), as_tensors=True)
                batch_data = (states, policies, values)
                
                model_config = {
                    "num_residual_blocks": settings["model_residual_blocks"],
                    "num_channels": settings["model_channels"],
                }
                state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
                
                # Use a manager for results
                ctx = torch.multiprocessing.get_context("spawn")
                manager = ctx.Manager()
                results_dict = manager.dict()
                
                torch.multiprocessing.spawn(
                    _train_worker,
                    args=(NUM_GPUS, model_config, state_dict, batch_data, settings, results_dict),
                    nprocs=NUM_GPUS,
                    join=True
                )
                
                metrics = results_dict.get("metrics")
                if "state_dict" in results_dict:
                    model.load_state_dict(results_dict["state_dict"])
            else:
                val_batches = 10 if val_buffer and len(val_buffer) > 0 else 0
                metrics = trainer.train_epoch(
                    replay_buffer=replay_buffer,
                    batch_size=train_batch_size,
                    num_batches=train_batches,
                    val_buffer=val_buffer,
                    val_batches=val_batches,
                )

            train_time = time.time() - train_start

            if metrics:
                if isinstance(metrics, dict) and "train" in metrics:
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
                    logger.info(
                        f"Training complete in {train_time:.1f}s. "
                        f"Loss: {metrics['total_loss']:.4f} "
                        f"(Pol: {metrics['policy_loss']:.4f}, "
                        f"Val: {metrics['value_loss']:.4f})"
                    )

            # ===== PHASE 3: Save Checkpoint =====
            if iteration % save_every == 0:
                metadata = {
                    "iteration": iteration,
                    "total_games": total_games,
                    "total_samples": total_samples,
                    "settings": settings,
                    "timestamp": time.time(),
                }
                checkpoint_path = checkpoint_dir / f"checkpoint_iter_{iteration}.pth"
                trainer.save_checkpoint(
                    str(checkpoint_path), replay_buffer=replay_buffer, metadata=metadata
                )

            iter_time = time.time() - iter_start
            logger.info(f"Iteration {iteration} complete in {iter_time:.1f}s")

            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {total_games} games, {total_samples} samples, "
                f"{elapsed/60:.1f} min elapsed"
            )

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user. Saving final checkpoint...")
        metadata = {
            "iteration": iteration,
            "total_games": total_games,
            "total_samples": total_samples,
            "settings": settings,
            "timestamp": time.time(),
        }
        final_checkpoint = checkpoint_dir / "checkpoint_final.pth"
        trainer.save_checkpoint(
            str(final_checkpoint), replay_buffer=replay_buffer, metadata=metadata
        )
        logger.info(f"Final checkpoint saved to {final_checkpoint}")

    total_time = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Total games: {total_games}")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Average: {total_time/total_games:.1f}s per game" if total_games > 0 else "")
    logger.info(f"Checkpoints saved in: {checkpoint_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

