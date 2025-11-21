# Quick Start: Training the Chess AI

This guide shows you how to quickly run training on any machine to see the model improve.

## Prerequisites

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

## Running Training

### Basic Usage

Simply run:

```bash
python run_training.py
```

This will:
1. Generate self-play games using MCTS
2. Train the neural network on those games
3. Show progress metrics (loss values, game counts, etc.)
4. Save checkpoints periodically

### Adjusting Parameters

Edit the configuration section at the top of `run_training.py`:

**For Faster Training (Quick Testing):**
```python
GAMES_PER_ITERATION = 3
MCTS_SIMULATIONS = 50
MODEL_CHANNELS = 128
MODEL_RESIDUAL_BLOCKS = 4
TRAIN_BATCHES_PER_EPOCH = 20
```

**For Better Quality (Longer Training):**
```python
GAMES_PER_ITERATION = 10
MCTS_SIMULATIONS = 200
MODEL_CHANNELS = 256
MODEL_RESIDUAL_BLOCKS = 8
TRAIN_BATCHES_PER_EPOCH = 100
```

### What to Expect

**First Iteration:**
- Model plays randomly (no training yet)
- Games may be long or end in draws
- Loss values will be high (6-8+)

**After Several Iterations:**
- Model starts learning basic patterns
- Games become more strategic
- Loss values should decrease (aim for < 4.0)

**Progress Indicators:**
- **Total Loss**: Should decrease over time (lower is better)
- **Policy Loss**: Measures move prediction accuracy
- **Value Loss**: Measures position evaluation accuracy
- **Validation Loss**: Should track training loss (if enabled)

### Monitoring Progress

The script logs:
- Number of games generated
- Training loss after each epoch
- Validation loss (if enabled)
- Time per game/iteration
- Checkpoint saves

### Stopping and Resuming

- Press `Ctrl+C` to stop safely (saves a final checkpoint)
- Checkpoints are saved in `checkpoints/` directory
- To resume, modify the script to load a checkpoint:

```python
# Add after model initialization:
trainer.load_checkpoint("checkpoints/checkpoint_iter_X.pth")
```

### Tips

1. **Start Small**: Use lower parameters first to verify everything works
2. **Device Selection**: 
   - **M1 Mac**: The script auto-detects MPS (Metal Performance Shaders) for ~2-3x speedup
   - **NVIDIA GPU**: Set `DEVICE = "cuda"` for fastest training
   - **CPU only**: Set `DEVICE = "cpu"` (slowest but works everywhere)
   - MPS on M1 Mac typically gives ~1-2 min/iteration vs ~3-6 min on CPU
3. **Memory**: If you run out of memory, reduce:
   - `GAMES_PER_ITERATION`
   - `TRAIN_BATCH_SIZE`
   - `replay_buffer.capacity`
4. **Time**: Each iteration takes (with default settings):
   - **M1 Mac (CPU)**: ~3-6 minutes per iteration
     - Game generation: ~2-4 minutes (5 games × ~30-50s each)
     - Training: ~1-2 minutes (50 batches)
   - **M1 Mac (MPS/GPU)**: ~1-2 minutes per iteration
     - Game generation: ~45-90 seconds
     - Training: ~15-30 seconds
   - **Intel CPU**: ~5-10 minutes per iteration
   - **NVIDIA GPU**: ~30-60 seconds per iteration
   
   **Breakdown per game on M1 Mac:**
   - With 100 MCTS simulations: ~30-50 seconds per game
   - With 50 MCTS simulations: ~15-25 seconds per game (faster but lower quality)
   - With 200 MCTS simulations: ~60-100 seconds per game (slower but better)

### Expected Timeline

- **Iteration 1-5**: Model learns basic rules and piece movement
- **Iteration 5-20**: Model develops opening preferences
- **Iteration 20+**: Model improves tactical and strategic play

For meaningful improvement, plan to run 20-50+ iterations.

### Troubleshooting

**"Not enough data in replay buffer"**
- Wait for more games to be generated
- Reduce `TRAIN_BATCH_SIZE`

**Out of Memory**
- Reduce `MODEL_CHANNELS` or `MODEL_RESIDUAL_BLOCKS`
- Reduce `TRAIN_BATCH_SIZE`
- Reduce `replay_buffer.capacity`

**Games are too slow**
- Reduce `MCTS_SIMULATIONS` (try 50-100)
- Reduce `GAMES_PER_ITERATION`

**Loss not decreasing**
- Model may need more training iterations
- Try increasing `TRAIN_BATCHES_PER_EPOCH`
- Check that validation loss also decreases (if enabled)

