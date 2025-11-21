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
2. **GPU vs CPU**: 
   - GPU: Much faster (set `DEVICE = "cuda"`)
   - CPU: Works but slower (set `DEVICE = "cpu"`)
3. **Memory**: If you run out of memory, reduce:
   - `GAMES_PER_ITERATION`
   - `TRAIN_BATCH_SIZE`
   - `replay_buffer.capacity`
4. **Time**: Each iteration takes:
   - ~30-60 seconds on CPU (with default settings)
   - ~10-20 seconds on GPU (with default settings)

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

