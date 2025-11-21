import chess
from board import ChessGame
from mcts import MCTS, MCTSNode, BatchInference
from dummy_model import DummyModel

# Setup
model = DummyModel()
inference = BatchInference(model, batch_size=1)
mcts = MCTS(inference_service=inference)

def test_returns_legal_move():
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)
    actions, probs = mcts.get_action_prob(root)

    for move in actions:
        assert move in state.get_legal_moves(), \
            f"Illegal move returned: {move}"

    print("✓ test_returns_legal_move passed")

def test_distribution_not_empty():
    state = ChessGame()
    root = mcts.search(state, num_simulations=50)
    actions, probs = mcts.get_action_prob(root)

    assert len(actions) > 0, "No actions returned"
    assert sum(probs) > 0.99, "Probabilities do not sum to 1"

    print("✓ test_distribution_not_empty passed")

def test_mcts_finds_mate_in_one():
    # Mate in 1 (White to move, Kf1#)
    fen = "6k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
    state = ChessGame(fen=fen)

    root = mcts.search(state, num_simulations=200)
    actions, probs = mcts.get_action_prob(root, temperature=0)

    best = actions[probs.index(max(probs))]
    print("Best move:", best)

    assert str(best) == "g1f1", "MCTS failed to find forced mate"

    print("✓ test_mcts_finds_mate_in_one passed")

def test_mcts_consistency():
    state = ChessGame()
    root1 = mcts.search(state, num_simulations=50)
    actions1, probs1 = mcts.get_action_prob(root1)

    root2 = mcts.search(state, num_simulations=50)
    actions2, probs2 = mcts.get_action_prob(root2)

    assert len(actions1) == len(actions2), "Inconsistent branching"

    print("✓ test_mcts_consistency passed")


if __name__ == "__main__":
    test_returns_legal_move()
    test_distribution_not_empty()
    test_mcts_finds_mate_in_one()
    test_mcts_consistency()

    print("\nALL TESTS PASSED")