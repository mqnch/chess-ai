"""
graph_metrics.py

Extracts metrics from training checkpoints and creates visualization plots.
Reads checkpoint metadata (iteration, games, samples) and optionally TensorBoard logs.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import glob
import re
from datetime import datetime

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available. Only checkpoint metrics will be plotted.")


def load_checkpoint_metadata(checkpoint_path: str) -> Optional[Dict]:
    """load metadata from a checkpoint file."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        return metadata
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def extract_iteration_number(filename: str) -> int:
    """extract iteration number from checkpoint filename."""
    match = re.search(r'iter_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def load_all_checkpoints(checkpoint_dir: str = "checkpoints") -> List[Dict]:
    """load metadata from all checkpoints in directory."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return []
    
    # find all checkpoint files
    checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.pth"))
    checkpoint_files.sort(key=lambda x: extract_iteration_number(x.name))
    
    all_metadata = []
    for ckpt_path in checkpoint_files:
        metadata = load_checkpoint_metadata(str(ckpt_path))
        if metadata:
            metadata['checkpoint_path'] = str(ckpt_path)
            metadata['filename'] = ckpt_path.name
            all_metadata.append(metadata)
    
    return all_metadata


def load_tensorboard_logs(log_dir: str = "logs/train") -> Optional[Dict]:
    """load metrics from tensorboard logs if available."""
    if not HAS_TENSORBOARD:
        return None
    
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    
    # find event files
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None
    
    # use most recent event file
    event_file = sorted(event_files)[-1]
    
    try:
        ea = EventAccumulator(str(event_file.parent))
        ea.Reload()
        
        tags = ea.Tags().get('scalars', [])
        metrics = {}
        
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            metrics[tag] = {'steps': steps, 'values': values}
        
        return metrics
    except Exception as e:
        print(f"Error loading TensorBoard logs: {e}")
        return None


def plot_checkpoint_metrics(metadata_list: List[Dict], output_file: str = "training_metrics.png"):
    """create plots from checkpoint metadata."""
    if not metadata_list:
        print("No checkpoint metadata to plot.")
        return
    
    # extract data
    iterations = []
    total_games = []
    total_samples = []
    timestamps = []
    
    for meta in metadata_list:
        iterations.append(meta.get('iteration', 0))
        total_games.append(meta.get('total_games', 0))
        total_samples.append(meta.get('total_samples', 0))
        timestamps.append(meta.get('timestamp', 0))
    
    # calculate time elapsed
    if timestamps and all(t > 0 for t in timestamps):
        start_time = min(timestamps)
        hours_elapsed = [(t - start_time) / 3600 for t in timestamps]
    else:
        hours_elapsed = iterations
    
    # calculate games/samples per iteration
    games_per_iter = []
    samples_per_iter = []
    for i in range(len(iterations)):
        if i == 0:
            games_per_iter.append(total_games[i])
            samples_per_iter.append(total_samples[i])
        else:
            games_per_iter.append(total_games[i] - total_games[i-1])
            samples_per_iter.append(total_samples[i] - total_samples[i-1])
    
    # create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Progress Metrics', fontsize=16, fontweight='bold')
    
    # plot 1: total games over iterations
    axes[0, 0].plot(iterations, total_games, 'b-o', markersize=6, linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Total Games')
    axes[0, 0].set_title('Total Games Generated')
    axes[0, 0].grid(True, alpha=0.3)
    
    # plot 2: total samples over iterations
    axes[0, 1].plot(iterations, total_samples, 'g-o', markersize=6, linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Total Samples')
    axes[0, 1].set_title('Total Training Samples')
    axes[0, 1].grid(True, alpha=0.3)
    
    # plot 3: games per iteration
    axes[0, 2].bar(iterations, games_per_iter, alpha=0.7, color='orange')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Games')
    axes[0, 2].set_title('Games per Iteration')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # plot 4: samples per iteration
    axes[1, 0].bar(iterations, samples_per_iter, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Samples')
    axes[1, 0].set_title('Samples per Iteration')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # plot 5: games over time
    if hours_elapsed:
        axes[1, 1].plot(hours_elapsed, total_games, 'r-o', markersize=6, linewidth=2)
        axes[1, 1].set_xlabel('Hours Elapsed')
        axes[1, 1].set_ylabel('Total Games')
        axes[1, 1].set_title('Games Generated Over Time')
        axes[1, 1].grid(True, alpha=0.3)
    
    # plot 6: average samples per game
    avg_samples_per_game = []
    for i in range(len(iterations)):
        if total_games[i] > 0:
            avg_samples_per_game.append(total_samples[i] / total_games[i])
        else:
            avg_samples_per_game.append(0)
    
    axes[1, 2].plot(iterations, avg_samples_per_game, 'm-o', markersize=6, linewidth=2)
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('Samples per Game')
    axes[1, 2].set_title('Average Samples per Game')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved checkpoint metrics plot to {output_file}")


def plot_tensorboard_metrics(tb_metrics: Dict, output_file: str = "tensorboard_metrics.png"):
    """create plots from tensorboard metrics."""
    if not tb_metrics:
        return
    
    # filter for common metrics
    loss_metrics = {k: v for k, v in tb_metrics.items() if 'loss' in k.lower()}
    
    if not loss_metrics:
        print("No loss metrics found in TensorBoard logs.")
        return
    
    num_metrics = len(loss_metrics)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else [axes]
    
    fig.suptitle('TensorBoard Training Metrics', fontsize=16, fontweight='bold')
    
    for idx, (metric_name, data) in enumerate(loss_metrics.items()):
        ax = axes[idx] if num_metrics > 1 else axes[0]
        ax.plot(data['steps'], data['values'], linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        ax.set_title(metric_name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    # hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved TensorBoard metrics plot to {output_file}")


def create_combined_plot(metadata_list: List[Dict], tb_metrics: Optional[Dict] = None, 
                        output_file: str = "combined_metrics.png"):
    """create a comprehensive combined plot."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    if not metadata_list:
        print("No data to plot.")
        return
    
    # extract checkpoint data
    iterations = [m.get('iteration', 0) for m in metadata_list]
    total_games = [m.get('total_games', 0) for m in metadata_list]
    total_samples = [m.get('total_samples', 0) for m in metadata_list]
    
    # checkpoint plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(iterations, total_games, 'b-o', markersize=6, linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Total Games')
    ax1.set_title('Total Games Generated')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(iterations, total_samples, 'g-o', markersize=6, linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Total Samples')
    ax2.set_title('Total Training Samples')
    ax2.grid(True, alpha=0.3)
    
    # calculate samples per game
    samples_per_game = []
    for i in range(len(iterations)):
        if total_games[i] > 0:
            samples_per_game.append(total_samples[i] / total_games[i])
        else:
            samples_per_game.append(0)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(iterations, samples_per_game, 'm-o', markersize=6, linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Samples per Game')
    ax3.set_title('Average Samples per Game')
    ax3.grid(True, alpha=0.3)
    
    # tensorboard metrics if available
    if tb_metrics:
        loss_metrics = {k: v for k, v in tb_metrics.items() if 'loss' in k.lower()}
        for idx, (metric_name, data) in enumerate(list(loss_metrics.items())[:3]):
            ax = fig.add_subplot(gs[1, idx])
            ax.plot(data['steps'], data['values'], linewidth=2)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.set_title(metric_name.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
    
    # summary statistics
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    summary_text = f"""
    Training Summary:
    • Total Iterations: {len(iterations)}
    • Total Games: {total_games[-1] if total_games else 0}
    • Total Samples: {total_samples[-1] if total_samples else 0}
    • Average Samples per Game: {np.mean(samples_per_game):.1f}
    • Checkpoints Found: {len(metadata_list)}
    """
    
    if tb_metrics:
        summary_text += f"• TensorBoard Metrics: {len(tb_metrics)} metrics logged\n"
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Complete Training Metrics Overview', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved combined metrics plot to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph training metrics from checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/train",
                       help="Directory containing TensorBoard logs")
    parser.add_argument("--output", type=str, default="training_metrics.png",
                       help="Output file for plots")
    parser.add_argument("--combined", action="store_true",
                       help="Create combined plot with all metrics")
    
    args = parser.parse_args()
    
    # load checkpoint metadata
    print(f"Loading checkpoints from {args.checkpoint_dir}...")
    metadata_list = load_all_checkpoints(args.checkpoint_dir)
    
    if not metadata_list:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(metadata_list)} checkpoints")
    for meta in metadata_list:
        print(f"  Iteration {meta.get('iteration', '?')}: "
              f"{meta.get('total_games', 0)} games, "
              f"{meta.get('total_samples', 0)} samples")
    
    # load tensorboard logs if available
    tb_metrics = None
    if HAS_TENSORBOARD:
        print(f"\nLoading TensorBoard logs from {args.log_dir}...")
        tb_metrics = load_tensorboard_logs(args.log_dir)
        if tb_metrics:
            print(f"Found {len(tb_metrics)} TensorBoard metrics")
        else:
            print("No TensorBoard logs found (this is okay)")
    
    # create plots
    if args.combined:
        create_combined_plot(metadata_list, tb_metrics, args.output)
    else:
        plot_checkpoint_metrics(metadata_list, args.output)
        if tb_metrics:
            plot_tensorboard_metrics(tb_metrics, "tensorboard_metrics.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()