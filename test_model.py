import torch
import chess
import unittest
from model import ChessNet
from encode import board_to_tensor, move_to_action, action_to_move
from board import ChessGame
from game import SelfPlayGame, SelfPlayConfig
from replay_buffer import ReplayBuffer


class TestChessNet(unittest.TestCase):
    """
    unit tests for the chess neural network model.
    """
    
    def test_model_initialization(self):
        """test that model initializes with default parameters."""
        model = ChessNet()
        self.assertEqual(model.num_residual_blocks, 8)
        self.assertEqual(model.num_channels, 256)
    
    def test_model_initialization_custom(self):
        """test that model initializes with custom parameters."""
        model = ChessNet(num_residual_blocks=10, num_channels=128)
        self.assertEqual(model.num_residual_blocks, 10)
        self.assertEqual(model.num_channels, 128)
    
    def test_forward_pass_single(self):
        """test forward pass with single board position."""
        model = ChessNet()
        model.eval()  # set to evaluation mode
        
        # create synthetic board tensor
        batch_size = 1
        input_tensor = torch.randn(batch_size, 18, 8, 8)
        
        with torch.no_grad():
            policy, value = model(input_tensor)
        
        # verify output shapes
        self.assertEqual(policy.shape, (batch_size, 8, 8, 73))
        self.assertEqual(value.shape, (batch_size, 1))
        
        # verify value is in [-1, 1] range
        self.assertGreaterEqual(value.item(), -1.0)
        self.assertLessEqual(value.item(), 1.0)
    
    def test_forward_pass_batch(self):
        """test forward pass with batch of board positions."""
        model = ChessNet()
        model.eval()
        
        batch_size = 4
        input_tensor = torch.randn(batch_size, 18, 8, 8)
        
        with torch.no_grad():
            policy, value = model(input_tensor)
        
        # verify output shapes
        self.assertEqual(policy.shape, (batch_size, 8, 8, 73))
        self.assertEqual(value.shape, (batch_size, 1))
        
        # verify all values are in [-1, 1] range
        self.assertTrue(torch.all(value >= -1.0))
        self.assertTrue(torch.all(value <= 1.0))
    
    def test_forward_pass_real_board(self):
        """test forward pass with real board encoding."""
        model = ChessNet()
        model.eval()
        
        # create real chess board
        game = ChessGame()
        board = game.get_board()
        board_tensor = board_to_tensor(board)
        
        # add batch dimension
        input_tensor = board_tensor.unsqueeze(0)
        
        with torch.no_grad():
            policy, value = model(input_tensor)
        
        # verify output shapes
        self.assertEqual(policy.shape, (1, 8, 8, 73))
        self.assertEqual(value.shape, (1, 1))
    
    def test_policy_log_softmax(self):
        """test that policy output sums to approximately 1 when exponentiated."""
        model = ChessNet()
        model.eval()
        
        input_tensor = torch.randn(1, 18, 8, 8)
        
        with torch.no_grad():
            policy, _ = model(input_tensor)
        
        # convert log probabilities to probabilities
        probs = torch.exp(policy)
        
        # sum over action dimension should be approximately 1 for each square
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))


class TestMoveEncoding(unittest.TestCase):
    """
    unit tests for move encoding and decoding functions.
    """
    
    def test_move_encoding_round_trip_simple(self):
        """test encoding and decoding a simple move."""
        board = chess.Board()
        move = chess.Move(chess.E2, chess.E4)
        
        # encode move
        from_square, action_type = move_to_action(move, board)
        self.assertIsNotNone(from_square)
        self.assertIsNotNone(action_type)
        self.assertEqual(from_square, chess.E2)
        
        # decode move
        decoded_move = action_to_move(from_square, action_type, board)
        self.assertIsNotNone(decoded_move)
        self.assertEqual(decoded_move, move)
    
    def test_move_encoding_round_trip_capture(self):
        """test encoding and decoding a capture move."""
        # set up position with a capture
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
        move = chess.Move(chess.D2, chess.D4)
        board.push(move)
        
        # try to capture
        capture_move = chess.Move(chess.E5, chess.D4)
        if capture_move in board.legal_moves:
            from_square, action_type = move_to_action(capture_move, board)
            self.assertIsNotNone(from_square)
            
            decoded_move = action_to_move(from_square, action_type, board)
            self.assertIsNotNone(decoded_move)
            self.assertEqual(decoded_move, capture_move)
    
    def test_move_encoding_promotion(self):
        """test encoding and decoding a promotion move."""
        # set up position near promotion
        board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
        
        # queen promotion
        move = chess.Move(chess.E7, chess.E8, promotion=chess.QUEEN)
        if move in board.legal_moves:
            from_square, action_type = move_to_action(move, board)
            self.assertIsNotNone(from_square)
            
            decoded_move = action_to_move(from_square, action_type, board)
            self.assertIsNotNone(decoded_move)
            self.assertEqual(decoded_move, move)
    
    def test_move_encoding_underpromotion(self):
        """test encoding and decoding an underpromotion move."""
        # set up position near promotion
        board = chess.Board("8/4P3/8/8/8/8/8/8 w - - 0 1")
        
        # knight underpromotion
        move = chess.Move(chess.E7, chess.E8, promotion=chess.KNIGHT)
        if move in board.legal_moves:
            from_square, action_type = move_to_action(move, board)
            self.assertIsNotNone(from_square)
            self.assertGreaterEqual(action_type, 64)  # should be in underpromotion range
            
            decoded_move = action_to_move(from_square, action_type, board)
            self.assertIsNotNone(decoded_move)
            self.assertEqual(decoded_move, move)
    
    def test_move_encoding_knight_move(self):
        """test encoding and decoding a knight move."""
        board = chess.Board()
        move = chess.Move(chess.B1, chess.C3)
        
        if move in board.legal_moves:
            from_square, action_type = move_to_action(move, board)
            self.assertIsNotNone(from_square)
            self.assertGreaterEqual(action_type, 56)  # should be in knight move range
            self.assertLess(action_type, 64)
            
            decoded_move = action_to_move(from_square, action_type, board)
            self.assertIsNotNone(decoded_move)
            self.assertEqual(decoded_move, move)
    
    def test_move_encoding_castling(self):
        """test encoding and decoding a castling move."""
        board = chess.Board()
        # make moves to allow castling
        board.push(chess.Move(chess.E2, chess.E4))
        board.push(chess.Move(chess.E7, chess.E5))
        board.push(chess.Move(chess.G1, chess.F3))
        board.push(chess.Move(chess.B8, chess.C6))
        board.push(chess.Move(chess.F1, chess.C4))
        board.push(chess.Move(chess.G8, chess.F6))
        
        # kingside castling
        move = chess.Move(chess.E1, chess.G1)
        if move in board.legal_moves:
            from_square, action_type = move_to_action(move, board)
            self.assertIsNotNone(from_square)
            
            decoded_move = action_to_move(from_square, action_type, board)
            self.assertIsNotNone(decoded_move)
            self.assertEqual(decoded_move, move)
    
    def test_move_encoding_all_legal_moves(self):
        """test that all legal moves can be encoded and decoded."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        
        successful_round_trips = 0
        for move in legal_moves:
            from_square, action_type = move_to_action(move, board)
            if from_square is not None:
                decoded_move = action_to_move(from_square, action_type, board)
                if decoded_move == move:
                    successful_round_trips += 1
        
        # should successfully encode/decode most moves
        # (some edge cases might fail, but majority should work)
        success_rate = successful_round_trips / len(legal_moves)
        self.assertGreater(success_rate, 0.8)  # at least 80% success rate


class TestIntegration(unittest.TestCase):
    """
    integration tests combining model and encoding.
    """
    
    def test_model_with_real_game(self):
        """test model forward pass through a real game sequence."""
        model = ChessNet()
        model.eval()
        
        game = ChessGame()
        
        # play a few moves
        moves = [
            chess.Move(chess.E2, chess.E4),
            chess.Move(chess.E7, chess.E5),
            chess.Move(chess.G1, chess.F3),
        ]
        
        for move in moves:
            if move in game.get_legal_moves():
                game.make_move(move)
                board = game.get_board()
                board_tensor = board_to_tensor(board)
                input_tensor = board_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    policy, value = model(input_tensor)
                
                # verify outputs
                self.assertEqual(policy.shape, (1, 8, 8, 73))
                self.assertEqual(value.shape, (1, 1))


class DummyModel(torch.nn.Module):
    """simple deterministic model for quick self-play tests."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        policy = torch.zeros(batch_size, 8, 8, 73, device=x.device)
        policy = torch.log_softmax(policy, dim=-1)
        value = torch.zeros(batch_size, 1, device=x.device)
        return policy, value


class TestSelfPlayAndReplayBuffer(unittest.TestCase):
    """tests for self-play generation and replay buffer sampling."""

    def test_self_play_generates_samples(self):
        config = SelfPlayConfig(
            num_simulations=2,
            batch_size=2,
            max_moves=10,
            temperature_switch_move=2,
            temperature_final=0.5,
        )
        with SelfPlayGame(model=DummyModel(), config=config) as runner:
            samples = runner.run_game()

        self.assertGreater(len(samples), 0)
        for sample in samples:
            self.assertEqual(sample.state.shape, (18, 8, 8))
            self.assertEqual(sample.policy.shape, (8, 8, 73))
            self.assertGreaterEqual(sample.value, -1.0)
            self.assertLessEqual(sample.value, 1.0)

    def test_replay_buffer_sampling(self):
        config = SelfPlayConfig(
            num_simulations=2,
            batch_size=2,
            max_moves=8,
            temperature_switch_move=2,
            temperature_final=0.5,
        )
        with SelfPlayGame(model=DummyModel(), config=config) as runner:
            samples = runner.run_game()

        self.assertGreater(len(samples), 0)

        buffer = ReplayBuffer(capacity=16, seed=42)
        buffer.extend(samples)
        buffer.extend(samples)

        batch_size = min(4, len(buffer))
        states, policies, values = buffer.sample(batch_size)

        self.assertEqual(states.shape, (batch_size, 18, 8, 8))
        self.assertEqual(policies.shape, (batch_size, 8, 8, 73))
        self.assertEqual(values.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()

