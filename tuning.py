"""
tuning.py

Hyper-parameter tuning utilities for the AlphaZero-style chess trainer.
Supports grid search, random sampling, and Bayesian optimization (via
scikit-optimize) to propose new configuration files for run_training.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from skopt import Optimizer
    from skopt.space import Integer, Real
except ImportError:  # pragma: no cover
    Optimizer = None  # type: ignore

from run_training import build_default_settings  # reuse default config structure

HYPERPARAM_SPACE = {
    "mcts_simulations": {"type": "int", "bounds": [200, 800]},
    "replay_buffer_capacity": {"type": "int", "bounds": [40_000, 200_000]},
    "train_batch_size": {"type": "choice", "values": [32, 48, 64, 96, 128]},
    "learning_rate": {"type": "float", "bounds": [5e-3, 3e-2]},
    "dirichlet_alpha": {"type": "float", "bounds": [0.1, 0.4]},
    "num_selfplay_workers": {"type": "choice", "values": [1, 2, 4, 8]},
    "distributed_max_batch": {"type": "choice", "values": [8, 16, 32]},
    "temperature_switch_move": {"type": "choice", "values": [20, 30, 40]},
}

GRID_KEYS = ["mcts_simulations", "train_batch_size", "learning_rate"]

BAYES_DIMENSIONS = [
    Integer(200, 800, name="mcts_simulations"),
    Integer(40_000, 200_000, name="replay_buffer_capacity"),
    Integer(32, 128, name="train_batch_size"),
    Real(5e-3, 3e-2, name="learning_rate"),
    Real(0.1, 0.4, name="dirichlet_alpha"),
    Integer(1, 8, name="num_selfplay_workers"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyper-parameter tuning helper")
    parser.add_argument(
        "--action",
        choices=["suggest", "record"],
        default="suggest",
        help="Suggest new configs or record the result of a completed run.",
    )
    parser.add_argument(
        "--mode",
        choices=["grid", "random", "bayes"],
        default="random",
        help="Search strategy for suggestions.",
    )
    parser.add_argument("--samples", type=int, default=1, help="Number of configs to suggest.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_configs",
        help="Directory to write generated config files.",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="tuning_results.json",
        help="JSON file tracking past configs and scores (for Bayesian mode).",
    )
    parser.add_argument("--config-path", type=str, help="Config file path when recording results.")
    parser.add_argument("--score", type=float, help="Observed score (lower is better) for recording.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def load_results(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(path: str, results: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def record_result(path: str, config_path: str, score: float):
    results = load_results(path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    results.append(
        {
            "timestamp": time.time(),
            "config_path": config_path,
            "config": config,
            "score": score,
        }
    )
    save_results(path, results)
    print(f"Recorded score {score:.4f} for config {config_path}")


def sample_random_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key, spec in HYPERPARAM_SPACE.items():
        if spec["type"] == "choice":
            cfg[key] = random.choice(spec["values"])
        elif spec["type"] == "int":
            low, high = spec["bounds"]
            cfg[key] = random.randint(low, high)
        elif spec["type"] == "float":
            low, high = spec["bounds"]
            cfg[key] = random.uniform(low, high)
    return cfg


def generate_grid_configs(limit: int) -> List[Dict[str, Any]]:
    values = [HYPERPARAM_SPACE[key]["values"] for key in GRID_KEYS]
    combos = list(itertools.product(*values))
    configs = []
    for combo in combos[:limit]:
        cfg = {}
        for key, value in zip(GRID_KEYS, combo):
            cfg[key] = value
        configs.append(cfg)
    return configs


def build_bayes_optimizer(seed: int) -> "Optimizer":
    if Optimizer is None:  # pragma: no cover
        raise RuntimeError("scikit-optimize is required for Bayesian optimization (pip install scikit-optimize)")
    return Optimizer(
        dimensions=BAYES_DIMENSIONS,
        base_estimator="GP",
        acq_func="EI",
        random_state=seed,
    )


def suggest_bayesian_configs(results_path: str, samples: int, seed: int) -> List[Dict[str, Any]]:
    optimizer = build_bayes_optimizer(seed)
    existing = load_results(results_path)
    dimension_names = [dim.name for dim in BAYES_DIMENSIONS]
    for entry in existing:
        params = [entry["config"].get(name, build_default_settings().get(name)) for name in dimension_names]
        optimizer.tell(params, entry["score"])
    suggestions = optimizer.ask(n_points=samples)
    configs = []
    for params in suggestions:
        cfg = {name: value for name, value in zip(dimension_names, params)}
        configs.append(cfg)
    return configs


def merge_with_defaults(overrides: Dict[str, Any]) -> Dict[str, Any]:
    config = build_default_settings()
    config.update(overrides)
    return config


def write_configs(configs: List[Dict[str, Any]], output_dir: str) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, cfg in enumerate(configs, start=1):
        timestamp = int(time.time())
        path = Path(output_dir) / f"config_{timestamp}_{idx}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        paths.append(str(path))
    return paths


def suggest_configs(args: argparse.Namespace):
    random.seed(args.seed)
    if args.mode == "grid":
        overrides = generate_grid_configs(args.samples)
    elif args.mode == "random":
        overrides = [sample_random_config() for _ in range(args.samples)]
    else:
        overrides = suggest_bayesian_configs(args.results, args.samples, args.seed)

    configs = [merge_with_defaults(cfg) for cfg in overrides]
    paths = write_configs(configs, args.output_dir)
    print("Generated config files:")
    for path in paths:
        print(f"  {path}")
    print("Run training with: python run_training.py --config <path>")


def main():
    args = parse_args()
    if args.action == "record":
        if not args.config_path or args.score is None:
            raise SystemExit("Recording requires --config-path and --score.")
        record_result(args.results, args.config_path, args.score)
    else:
        suggest_configs(args)


if __name__ == "__main__":
    main()

