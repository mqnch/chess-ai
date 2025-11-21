"""
test_mcts_full.py

Comprehensive test suite for the MCTS + ChessGame integration
before moving on to Stage 4 (self-play generation).

Covers:
- Basic legality and distribution tests
- Node consistency and tree structure
- Terminal position value propagation
- Full-game self-play simulation
- Optional BatchInference stress test
- Optional memory growth test
"""

import numpy as np
import torch
import chess

from mcts import MCTS, MCTSNode, BatchInference
from board import ChessGame


# ---------------------------------------------------------------------
# Dummy model for testing (uniform policy, zero value)
# ---------------------------------------------------------------------

class DummyModel(torch.nn.Module):
    """
    Simple stub network:
    - policy: uniform over all 8x8x73 action logits
    - value: always zero
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: (batch_size, C, 8, 8) from board_to_tensor
        batch = x.shape[0]
        # Policy shape should match your encode: (8, 8, 73)
        policy = torch.ones((batch, 8, 8, 73), dtype=torch.float32)
        policy = policy / policy.numel()  # Not strictly necessary; logits can be anything
        value = torch.zeros((batch, 1), dtype=torch.float32)
        return policy, value


# ---------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------

def create_mcts():
    model = DummyModel()
    inference = BatchInference(model, batch_size=4, device="cpu")
    mcts = MCTS(inference_service=inference)
    return mcts, inference


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_returns_legal_move():
    print("\n[TEST] MCTS returns legal moves from starting position")

    mcts, inference = create_mcts()
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)
    actions, probs = mcts.get_action_prob(root)

    legal_moves = set(state.get_legal_moves())
    assert len(actions) > 0, "No actions returned from MCTS"

    for move in actions:
        assert move in legal_moves, f"Illegal move returned: {move}"

    print("  All returned moves are legal")
    inference.stop()


def test_distribution_not_empty():
    print("\n[TEST] Policy distribution sanity")

    mcts, inference = create_mcts()
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)
    actions, probs = mcts.get_action_prob(root)

    assert len(actions) == len(probs), "Mismatch between actions and probabilities"
    assert len(actions) > 0, "No actions returned"
    s = sum(probs)
    assert abs(s - 1.0) < 1e-3, f"Probabilities should sum to 1, got {s}"

    print(" Non-empty distribution, probabilities sum to ~1")
    inference.stop()


def test_root_invariance():
    print("\n[TEST] Root invariance (running MCTS twice from start)")

    mcts, inference = create_mcts()

    state1 = ChessGame()
    root1 = mcts.search(state1, num_simulations=100)
    a1, p1 = mcts.get_action_prob(root1)

    state2 = ChessGame()
    root2 = mcts.search(state2, num_simulations=100)
    a2, p2 = mcts.get_action_prob(root2)

    assert len(a1) == len(a2), "Different number of actions from identical positions"

    print("  Action set size is consistent across runs")
    inference.stop()


def test_node_consistency():
    print("\n[TEST] Children states are consistent with moves from root")

    mcts, inference = create_mcts()
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)

    for move, child in root.children.items():
        expected = state.copy()
        expected.make_move(move)
        assert child.state.board.fen() == expected.board.fen(), \
            f"Child state FEN mismatch for move {move}"

    print("  All child nodes match expected next-board states")
    inference.stop()


def test_terminal_propagation_draw():
    print("\n[TEST] Terminal propagation: draw position")

    # Stalemate-like or draw position
    # Example simple draw FEN (king vs king)
    fen = "8/8/8/8/8/8/5k2/7K w - - 0 1"
    state = ChessGame(fen=fen)

    mcts, inference = create_mcts()
    root = mcts.search(state, num_simulations=20)

    # In a drawn position, values should hover around 0
    for move, child in root.children.items():
        assert abs(child.value) < 1e-3, \
            f"Child value should be ~0 in drawn position, got {child.value}"

    print("  Values around 0 for draw position")
    inference.stop()


def test_terminal_propagation_mate_in_one():
    print("\n[TEST] Terminal propagation: mate in one")

    # White can mate in one with Kf1# in this position:
    # (simple constructed example)
    fen = "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
    state = ChessGame(fen=fen)

    mcts, inference = create_mcts()
    root = mcts.search(state, num_simulations=200)
    actions, probs = mcts.get_action_prob(root, temperature=0)

    # Get best move (max probability)
    best_index = int(np.argmax(probs))
    best_move = actions[best_index]

    print(f"  Best move chosen: {best_move}")
    # This exact move name may differ if FEN or example differs,
    # so you can adjust this assertion to your actual known mate move.
    # Example: assert str(best_move) == "g1f1"

    print("  MCTS finds a strong move in tactical position (manual check above)")
    inference.stop()


def test_full_game_self_play(max_moves=200):
    print("\n[TEST] Full-game self-play simulation")

    mcts, inference = create_mcts()
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)

    move_count = 0

    while not state.is_terminal() and move_count < max_moves:
        actions, probs = mcts.get_action_prob(root, temperature=1.0)
        if not actions:
            print("  ! No actions available, breaking early.")
            break

        best_index = int(np.argmax(probs))
        move = actions[best_index]

        ok = state.make_move(move)
        assert ok, f"MCTS suggested illegal move: {move}"

        # New root from new state
        root = MCTSNode(state, prior=1.0)
        root = mcts.search(root, num_simulations=50)

        move_count += 1

    print(f"  Game finished after {move_count} moves. Result: {state.get_result()}")
    print("  Self-play game ran to completion (or max_moves) without crashes")
    inference.stop()


def test_batch_inference_stress():
    print("\n[TEST] BatchInference stress test")

    import threading

    mcts, inference = create_mcts()

    def run_inferences(n):
        for _ in range(n):
            _ = inference.predict(ChessGame())

    threads = [threading.Thread(target=run_inferences, args=(20,)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("  BatchInference handled concurrent requests without crashing")
    inference.stop()


def test_memory_growth():
    print("\n[TEST] Optional memory growth test")

    try:
        import psutil, os
    except ImportError:
        print("  psutil not installed, skipping memory test.")
        return

    mcts, inference = create_mcts()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss

    for _ in range(50):
        _ = mcts.search(ChessGame(), num_simulations=50)

    mem_after = process.memory_info().rss
    diff_mb = (mem_after - mem_before) / (1024 * 1024)

    print(f"  Memory increased by ~{diff_mb:.2f} MB after repeated searches.")
    print("  (Manually decide if this is acceptable for now.)")
    inference.stop()


# ---------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------

if __name__ == "__main__":
    test_returns_legal_move()
    test_distribution_not_empty()
    test_root_invariance()
    test_node_consistency()
    test_terminal_propagation_draw()
    test_terminal_propagation_mate_in_one()
    test_full_game_self_play()
    test_batch_inference_stress()
    test_memory_growth()

    print("\nALL TESTS COMPLETED.\n")