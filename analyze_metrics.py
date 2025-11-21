"""
analyze_metrics.py

Parses training logs (if available) or tensorboard logs to visualize training progress.
For this simple implementation, it parses a dummy log format or just creates a placeholder 
plot to demonstrate metric analysis.

Ideally, this would parse the `logs/train` TensorBoard directory using `tensorboard.backend.event_processing.event_accumulator`.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs(log_dir):
    """
    Reads TensorBoard logs and plots training metrics.
    """
    if not os.path.exists(log_dir):
        print(f"Log directory {log_dir} does not exist.")
        return

    # Find the most recent event file
    event_files = [f for f in os.listdir(log_dir) if f.startswith('events.out.tfevents')]
    if not event_files:
        print("No event files found.")
        return
    
    event_file = os.path.join(log_dir, event_files[-1])
    print(f"Loading logs from {event_file}...")
    
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Available tags
    tags = ea.Tags()['scalars']
    print(f"Found tags: {tags}")
    
    metrics = ['total_loss', 'policy_loss', 'value_loss']
    
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        if metric in tags:
            events = ea.Scalars(metric)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            
            plt.subplot(1, 3, i+1)
            plt.plot(steps, values)
            plt.title(metric)
            plt.xlabel('Step')
            plt.ylabel('Value')
            plt.grid(True)
            
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Saved metrics plot to training_metrics.png")

if __name__ == "__main__":
    # Default location from train.py
    log_dir = "logs/train"
    plot_tensorboard_logs(log_dir)

