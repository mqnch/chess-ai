import pygame
import sys
import io
import torch
import threading
from pathlib import Path
from board import ChessGame
import chess
import chess.svg
from model import ChessNet
from mcts import MCTS, BatchInference
try:
    import cairosvg
    HAS_CAIROSVG = True
except (ImportError, OSError):
    # OSError catches missing GTK+ DLLs on Windows
    HAS_CAIROSVG = False
    try:
        from PIL import Image
        import xml.etree.ElementTree as ET
        HAS_PIL = True
    except ImportError:
        HAS_PIL = False


# color constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (247, 247, 105)
MOVE_HIGHLIGHT = (186, 202, 68)
SELECTED = (255, 255, 0)
TEXT_COLOR = (50, 50, 50)

# board dimensions
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
MARGIN = 40
INFO_WIDTH = 200
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 720

# piece image cache will be populated during initialization


class ChessVisualizer:
    """
    pygame-based visual interface for playing chess.
    supports move selection, legal move highlighting, and game status display.
    """
    
    def __init__(self, model_path=None):
        """initialize the chess visualizer.
        
        args:
            model_path: optional path to checkpoint file to load AI opponent
        """
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI - Visual Interface")
        
        # initialize font
        self.font_large = pygame.font.Font("public/inter.ttf", 72)
        self.font_medium = pygame.font.Font("public/inter.ttf", 36)
        self.font_small = pygame.font.Font("public/inter.ttf", 18)
        
        # game state
        self.game = ChessGame()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        
        # AI setup
        self.ai_model = None
        self.ai_mcts = None
        self.ai_inference = None
        self.game_mode = 'human_vs_human'
        
        # auto-detect latest checkpoint if no path provided
        if model_path is None:
            model_path = self._find_latest_checkpoint()
        
        if model_path:
            self._load_ai_model(model_path)
            self.game_mode = 'human_vs_ai'
        
        # load and cache piece images from chess.svg
        self.piece_images = {}
        self._load_piece_images()
    
    def square_to_pixel(self, square):
        """
        convert chess square index to pixel coordinates on screen.
        
        args:
            square: chess square index (0-63)
        
        returns:
            tuple (x, y) pixel coordinates
        """
        row = square // 8
        col = square % 8
        x = MARGIN + col * SQUARE_SIZE
        y = MARGIN + row * SQUARE_SIZE
        return (x, y)
    
    def pixel_to_square(self, x, y):
        """
        convert pixel coordinates to chess square index.
        
        args:
            x, y: pixel coordinates
        
        returns:
            chess square index (0-63) or None if outside board
        """
        if x < MARGIN or y < MARGIN:
            return None
        if x >= MARGIN + BOARD_SIZE or y >= MARGIN + BOARD_SIZE:
            return None
        
        col = (x - MARGIN) // SQUARE_SIZE
        row = (y - MARGIN) // SQUARE_SIZE
        
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            return None
        
        return row * 8 + col
    
    def _find_latest_checkpoint(self):
        """find the latest checkpoint in the checkpoints directory."""
        checkpoint_dir = Path("checkpoints")
        if not checkpoint_dir.exists():
            return None
        
        # find all checkpoint files
        checkpoint_files = list(checkpoint_dir.glob("checkpoint_iter_*.pth"))
        if not checkpoint_files:
            return None
        
        # sort by iteration number (extract from filename)
        def get_iteration(path):
            try:
                # extract number from "checkpoint_iter_10.pth" -> 10
                return int(path.stem.split("_")[-1])
            except:
                return 0
        
        checkpoint_files.sort(key=get_iteration, reverse=True)
        latest = checkpoint_files[0]
        print(f"Auto-detected latest checkpoint: {latest}")
        return str(latest)
    
    def _load_ai_model(self, checkpoint_path):
        """load a trained model from checkpoint for AI opponent."""
        try:
            # use Metal (MPS) on Mac, fallback to CUDA or CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"  # Metal Performance Shaders for M1/M2 Macs
            else:
                device = "cpu"
            
            print(f"Using device: {device}")
            
            # load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # determine model architecture from checkpoint or use defaults
            metadata = checkpoint.get("metadata", {})
            settings = metadata.get("settings", {})
            num_blocks = settings.get("model_residual_blocks", 6)
            num_channels = settings.get("model_channels", 128)
            
            # create and load model
            self.ai_model = ChessNet(
                num_residual_blocks=num_blocks,
                num_channels=num_channels
            )
            self.ai_model.load_state_dict(checkpoint["model_state_dict"])
            self.ai_model.to(device)
            self.ai_model.eval()
            
            # setup MCTS with optimized batch size for Metal
            batch_size = 8 if device == "mps" else 4
            self.ai_inference = BatchInference(
                model=self.ai_model,
                batch_size=batch_size,
                device=device
            )
            self.ai_mcts = MCTS(
                inference_service=self.ai_inference,
                c_puct=1.0
            )
            
            print(f"AI model loaded from {checkpoint_path}")
            print(f"Model: {num_blocks} blocks, {num_channels} channels")
            print(f"Device: {device}, Batch size: {batch_size}")
        except Exception as e:
            print(f"Error loading AI model: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to human vs human mode")
            self.ai_model = None
            self.ai_mcts = None
            self.ai_inference = None
            self.game_mode = 'human_vs_human'
    
    def _get_ai_move(self):
        """get the AI's move using MCTS."""
        if not self.ai_mcts:
            return None
        
        try:
            # adjust simulations for 5-10 second moves
            # with MPS: ~100-150 sims should give 5-10 seconds
            # with CPU: ~50-100 sims should give 5-10 seconds
            device = next(self.ai_model.parameters()).device
            if str(device) == "mps":
                num_simulations = 250  # optimized for Metal GPU
            else:
                num_simulations = 80   # optimized for CPU
            
            # run MCTS search
            root = self.ai_mcts.search(
                root=self.game.copy(),
                num_simulations=num_simulations
            )
            
            # get best move (temperature=0 for deterministic)
            moves, probs = self.ai_mcts.get_action_prob(root, temperature=0.0)
            
            if not moves:
                return None
            
            # return most visited move
            return moves[0]
        except Exception as e:
            print(f"Error getting AI move: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_piece_images(self):
        """load chess piece images from chess.svg and cache them as pygame surfaces."""
        piece_size = int(SQUARE_SIZE * 0.9)  # slightly smaller than square
        
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        colors = [chess.WHITE, chess.BLACK]
        
        for piece_type in piece_types:
            for color in colors:
                svg_string = chess.svg.piece(chess.Piece(piece_type, color), size=piece_size)

                if HAS_CAIROSVG:
                    png_bytes = cairosvg.svg2png(bytestring=svg_string.encode('utf-8'))
                    png_file = io.BytesIO(png_bytes)

                    surface = pygame.image.load(png_file)
                    surface = surface.convert_alpha()

                else:
                    # fallback to unicode
                    surface = self._create_unicode_piece(piece_type, color, piece_size)
                    
                self.piece_images[(piece_type, color)] = surface
        
                    
    def _create_unicode_piece(self, piece_type, color, size):
        """create a pygame surface with unicode piece symbol as fallback."""
        unicode_symbols = {
            (chess.PAWN, chess.WHITE): '♙',
            (chess.PAWN, chess.BLACK): '♟',
            (chess.KNIGHT, chess.WHITE): '♘',
            (chess.KNIGHT, chess.BLACK): '♞',
            (chess.BISHOP, chess.WHITE): '♗',
            (chess.BISHOP, chess.BLACK): '♝',
            (chess.ROOK, chess.WHITE): '♖',
            (chess.ROOK, chess.BLACK): '♜',
            (chess.QUEEN, chess.WHITE): '♕',
            (chess.QUEEN, chess.BLACK): '♛',
            (chess.KING, chess.WHITE): '♔',
            (chess.KING, chess.BLACK): '♚',
        }
        
        symbol = unicode_symbols.get((piece_type, color), '?')
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        font = pygame.font.Font(None, size)
        text_surface = font.render(symbol, True, BLACK if color == chess.WHITE else WHITE)
        text_rect = text_surface.get_rect(center=(size // 2, size // 2))
        surface.blit(text_surface, text_rect)
        return surface
    
    def draw_board(self):
        """draw the chess board with alternating square colors."""
        for row in range(8):
            for col in range(8):
                x = MARGIN + col * SQUARE_SIZE
                y = MARGIN + row * SQUARE_SIZE
                
                # determine square color
                is_light = (row + col) % 2 == 0
                color = LIGHT_SQUARE if is_light else DARK_SQUARE
                
                # highlight selected square
                square = row * 8 + col
                if square == self.selected_square:
                    color = SELECTED
                elif square in [move.to_square for move in self.legal_moves_for_selected]:
                    color = MOVE_HIGHLIGHT
                
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
    
    def draw_pieces(self):
        """draw chess pieces on the board using chess.svg icons."""
        board = self.game.get_board()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # get cached piece image
                piece_image = self.piece_images.get((piece.piece_type, piece.color))
                if piece_image is not None:
                    x, y = self.square_to_pixel(square)
                    
                    # center the piece in the square
                    piece_rect = piece_image.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                    self.screen.blit(piece_image, piece_rect)
    
    def draw_info_panel(self):
        """draw information panel on the right side."""
        x_start = BOARD_SIZE + MARGIN * 2
        y_start = MARGIN
        
        # game status
        status_text = "Game Status"
        text = self.font_medium.render(status_text, True, TEXT_COLOR)
        self.screen.blit(text, (x_start, y_start))
        y_start += 40
        
        # turn indicator
        turn = "White" if self.game.get_turn() else "Black"
        turn_text = f"Turn: {turn}"
        text = self.font_small.render(turn_text, True, TEXT_COLOR)
        self.screen.blit(text, (x_start, y_start))
        y_start += 30
        
        # game state
        if self.game.is_terminal():
            if self.game.is_checkmate():
                status = "Checkmate!"
            elif self.game.is_stalemate():
                status = "Stalemate"
            elif self.game.is_draw():
                status = "Draw"
            else:
                status = "Game Over"
            
            text = self.font_small.render(status, True, TEXT_COLOR)
            self.screen.blit(text, (x_start, y_start))
            y_start += 30
            
            # winner
            winner = self.game.get_winner()
            if winner == 1:
                winner_text = "White wins!"
            elif winner == -1:
                winner_text = "Black wins!"
            else:
                winner_text = "Draw"
            
            text = self.font_small.render(winner_text, True, TEXT_COLOR)
            self.screen.blit(text, (x_start, y_start))
            y_start += 30
        else:
            # check indicator
            board = self.game.get_board()
            if board.is_check():
                check_text = "Check!"
                text = self.font_small.render(check_text, True, (255, 0, 0))
                self.screen.blit(text, (x_start, y_start))
                y_start += 30
        
        y_start += 20
        
        # move count
        move_count_text = f"Move: {len(self.move_history)}"
        text = self.font_small.render(move_count_text, True, TEXT_COLOR)
        self.screen.blit(text, (x_start, y_start))
        y_start += 40
        
        # game mode indicator
        if self.game_mode == 'human_vs_ai':
            mode_text = "Mode: Human vs AI"
            text = self.font_small.render(mode_text, True, (0, 128, 0))
            self.screen.blit(text, (x_start, y_start))
            y_start += 25
        
        # instructions
        instructions = [
            "Click piece to select",
            "Click destination to move",
            "Press R to reset",
            "Press U to undo"
        ]
        
        for instruction in instructions:
            text = self.font_small.render(instruction, True, TEXT_COLOR)
            self.screen.blit(text, (x_start, y_start))
            y_start += 25
    
    def handle_click(self, x, y):
        """handle mouse click on the board."""
        # don't allow moves if it's AI's turn
        if self.game_mode == 'human_vs_ai' and self.game.get_turn() == chess.BLACK:
            return
        
        square = self.pixel_to_square(x, y)
        if square is None:
            return
        
        board = self.game.get_board()
        
        # if no square selected, try to select this square
        if self.selected_square is None:
            piece = board.piece_at(square)
            # only select if it's a piece of the current player's color
            if piece is not None and piece.color == board.turn:
                self.selected_square = square
                # get legal moves for this piece
                self.legal_moves_for_selected = [
                    move for move in board.legal_moves
                    if move.from_square == square
                ]
        else:
            # try to make a move
            move = None
            for legal_move in self.legal_moves_for_selected:
                if legal_move.to_square == square:
                    move = legal_move
                    break
            
            if move is not None:
                # make the move
                if self.game.make_move(move):
                    self.move_history.append(move)
                    self.selected_square = None
                    self.legal_moves_for_selected = []
            else:
                # clicked on different square - deselect or select new piece
                piece = board.piece_at(square)
                if piece is not None and piece.color == board.turn:
                    # select new piece
                    self.selected_square = square
                    self.legal_moves_for_selected = [
                        move for move in board.legal_moves
                        if move.from_square == square
                    ]
                else:
                    # clicked on empty square or opponent piece - deselect
                    self.selected_square = None
                    self.legal_moves_for_selected = []
    
    def reset_game(self):
        """reset the game to starting position."""
        self.game = ChessGame()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
    
    def undo_move(self):
        """undo the last move."""
        if len(self.move_history) > 0:
            self.game.undo_move()
            self.move_history.pop()
            self.selected_square = None
            self.legal_moves_for_selected = []
    
    def _calculate_ai_move_async(self):
        """calculate AI move in background thread."""
        try:
            ai_move = self._get_ai_move()
            # store result in thread-safe way
            self._ai_move_result = ai_move
            self._ai_move_ready = True
        except Exception as e:
            print(f"Error in AI move calculation: {e}")
            self._ai_move_result = None
            self._ai_move_ready = True
    
    def run(self):
        """main game loop."""
        clock = pygame.time.Clock()
        running = True
        ai_thinking = False
        ai_thread = None
        self._ai_move_result = None
        self._ai_move_ready = False
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left click
                        if not ai_thinking:
                            self.handle_click(event.pos[0], event.pos[1])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                        # cancel AI thinking if reset
                        if ai_thread and ai_thread.is_alive():
                            ai_thinking = False
                    elif event.key == pygame.K_u:
                        self.undo_move()
                        # cancel AI thinking if undo
                        if ai_thread and ai_thread.is_alive():
                            ai_thinking = False
            
            # check if AI move is ready
            if ai_thinking and self._ai_move_ready:
                ai_move = self._ai_move_result
                if ai_move:
                    self.game.make_move(ai_move)
                    self.move_history.append(ai_move)
                    # clear selection state immediately after AI move
                    self.selected_square = None
                    self.legal_moves_for_selected = []
                # reset AI state immediately
                ai_thinking = False
                self._ai_move_ready = False
                self._ai_move_result = None
                ai_thread = None
            
            # start AI thinking if it's AI's turn
            if (self.game_mode == 'human_vs_ai' and 
                not self.game.is_terminal() and 
                not ai_thinking and
                self.game.get_turn() == chess.BLACK):  # AI plays black
                ai_thinking = True
                self._ai_move_ready = False
                self._ai_move_result = None
                # start AI calculation in background thread
                ai_thread = threading.Thread(target=self._calculate_ai_move_async, daemon=True)
                ai_thread.start()
            
            # draw everything (this happens every frame, keeping UI responsive)
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()
            
            # show "AI thinking..." if needed
            if ai_thinking:
                thinking_text = self.font_small.render("AI thinking...", True, (255, 0, 0))
                x_start = BOARD_SIZE + MARGIN * 2
                self.screen.blit(thinking_text, (x_start, WINDOW_HEIGHT - 50))
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS - keeps UI smooth
        
        # cleanup
        if self.ai_inference:
            self.ai_inference.stop()
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Loading AI model from: {model_path}")
    
    visualizer = ChessVisualizer(model_path=model_path)
    visualizer.run()

