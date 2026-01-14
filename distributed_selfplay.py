"""
distributed_selfplay.py

Utilities for running multi-process self-play with a shared inference server.
Designed to scale across multiple CPU workers on a single machine or remote
GPU servers (e.g., WATCLOUD) by routing inference requests through a central
queue.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from encode import board_to_tensor
from game import SelfPlayConfig, SelfPlayGame, SelfPlaySample
from model import ChessNet

# ---------------------------------------------------------------------------
# Remote inference plumbing
# ---------------------------------------------------------------------------


class RemoteInferenceClient:
    """
    lightweight proxy that forwards inference requests to the central server.
    each worker process creates its own client with a dedicated response queue.
    """

    def __init__(self, request_queue: mp.Queue, response_queue: mp.Queue):
        self.request_queue = request_queue
        self.response_queue: mp.Queue = response_queue
        self._next_request_id = 0
        self._closed = False

    def predict(self, state) -> Tuple[np.ndarray, np.ndarray]:
        """match BatchInference API used by SelfPlayGame."""
        if self._closed:
            raise RuntimeError("remote inference client already closed")

        board = state.get_board()
        tensor = board_to_tensor(board)  # torch tensor on cpu
        request_id = (id(self), self._next_request_id)
        self._next_request_id += 1

        self.request_queue.put(
            {
                "type": "PREDICT",
                "request_id": request_id,
                "state": tensor.numpy(),
                "response_queue": self.response_queue,
            }
        )

        policy, value = self.response_queue.get()
        return policy, value

    def close(self):
        if not self._closed:
            self._closed = True
            # Queue proxies from multiprocessing.Manager don't have close()
            # They're automatically cleaned up by the manager
            if hasattr(self.response_queue, 'close'):
                self.response_queue.close()


def _inference_server_main(
    model_config: Dict[str, int],
    state_dict: Dict[str, torch.Tensor],
    device: str,
    request_queue: mp.Queue,
    max_batch_size: int = 16,
):
    """
    runs inside a dedicated process and batches inference requests coming from
    multiple self-play workers.
    """
    torch.set_num_threads(1)
    model = ChessNet(
        num_residual_blocks=model_config["num_residual_blocks"],
        num_channels=model_config["num_channels"],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    stop_requested = False

    while not stop_requested:
        try:
            message = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if message.get("type") == "STOP":
            stop_requested = True
            break

        batch = [message]

        # try to build a batch without blocking too long
        while len(batch) < max_batch_size:
            try:
                msg = request_queue.get_nowait()
                if msg.get("type") == "STOP":
                    stop_requested = True
                    break
                batch.append(msg)
            except queue.Empty:
                break

        states = np.stack([item["state"] for item in batch])
        state_tensor = torch.from_numpy(states).to(device)

        with torch.no_grad():
            policy_logits, values = model(state_tensor)
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()

        for item, policy, value in zip(batch, policy_logits, values):
            response_queue: mp.Queue = item["response_queue"]
            response_queue.put((policy, value))

    # drain outstanding requests gracefully
    while True:
        try:
            item = request_queue.get_nowait()
        except queue.Empty:
            break
        if item.get("type") == "PREDICT":
            item["response_queue"].put((np.zeros((8, 8, 73), dtype=np.float32), np.zeros(1, dtype=np.float32)))


def _self_play_worker_main(
    worker_id: int,
    num_games: int,
    config_dict: Dict,
    request_queue: mp.Queue,
    result_queue: mp.Queue,
    response_queue: mp.Queue,
):
    """
    worker process: runs self-play games using RemoteInferenceClient.
    """
    config = SelfPlayConfig(**config_dict)
    client = RemoteInferenceClient(request_queue, response_queue)
    runner = SelfPlayGame(
        model=None,
        device="cpu",
        config=config,
        inference_service=client,
        owns_inference=False,
    )

    games_played = 0
    try:
        while games_played < num_games:
            samples = runner.run_game()
            games_played += 1
            if samples:
                result_queue.put(
                    {
                        "type": "SAMPLES",
                        "worker_id": worker_id,
                        "samples": samples,
                    }
                )
    finally:
        client.close()
        runner.close()
        result_queue.put({"type": "DONE", "worker_id": worker_id, "games": games_played})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_distributed_self_play(
    model: torch.nn.Module,
    config: SelfPlayConfig,
    total_games: int,
    device: str,
    num_workers: int = 4,
    inference_device: Optional[str | List[str]] = None,
    max_batch_size: int = 16,
) -> List[List[SelfPlaySample]]:
    """
    spawn shared inference servers (one per device) and multiple self-play workers,
    returning the collected trajectories.
    """
    if total_games <= 0 or num_workers <= 0:
        return []

    # Handle multiple inference devices
    if inference_device is None:
        inference_devices = [device]
    elif isinstance(inference_device, str):
        inference_devices = [inference_device]
    else:
        inference_devices = inference_device

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    
    # One request queue per inference server
    request_queues: List[mp.Queue] = [
        manager.Queue(maxsize=num_workers * max_batch_size * 2)
        for _ in inference_devices
    ]
    result_queue: mp.Queue = manager.Queue()

    model_config = {
        "num_residual_blocks": getattr(model, "num_residual_blocks", 6),
        "num_channels": getattr(model, "num_channels", 256),
    }
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    servers: List[mp.Process] = []
    for i, dev in enumerate(inference_devices):
        server = ctx.Process(
            target=_inference_server_main,
            args=(model_config, state_dict, dev, request_queues[i], max_batch_size),
            name=f"InferenceServer-{i}",
        )
        server.start()
        servers.append(server)

    games_per_worker = _split_work(total_games, num_workers)
    workers: List[mp.Process] = []
    config_dict = asdict(config)
    
    response_queues: List[mp.Queue] = []

    for worker_id, games_to_play in enumerate(games_per_worker):
        if games_to_play <= 0:
            continue
        response_queue = manager.Queue()
        response_queues.append(response_queue)
        
        # Round-robin assignment of workers to inference servers
        q_idx = worker_id % len(request_queues)
        
        proc = ctx.Process(
            target=_self_play_worker_main,
            args=(worker_id, games_to_play, config_dict, request_queues[q_idx], result_queue, response_queue),
            name=f"SelfPlayWorker-{worker_id}",
        )
        proc.start()
        workers.append(proc)

    trajectories: List[List[SelfPlaySample]] = []
    finished_workers = 0
    expected_workers = len(workers)

    try:
        while finished_workers < expected_workers:
            try:
                message = result_queue.get(timeout=1.0)
                if message["type"] == "SAMPLES":
                    trajectories.append(message["samples"])
                elif message["type"] == "DONE":
                    finished_workers += 1
            except queue.Empty:
                continue
    finally:
        for proc in workers:
            proc.join()
        for q in request_queues:
            q.put({"type": "STOP"})
        for server in servers:
            server.join()

    return trajectories


def _split_work(total: int, workers: int) -> List[int]:
    base = total // workers
    remainder = total % workers
    allocation = []
    for i in range(workers):
        extra = 1 if i < remainder else 0
        allocation.append(base + extra)
    return allocation


if __name__ == "__main__":
    print("distributed_selfplay is a utility module. Use run_training.py to launch training.")

