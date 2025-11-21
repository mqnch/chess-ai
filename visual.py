import pygame
import sys
from board import ChessGame
import chess


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
MARGIN = 20
INFO_WIDTH = 200
WINDOW_WIDTH = BOARD_SIZE + MARGIN * 2 + INFO_WIDTH
WINDOW_HEIGHT = BOARD_SIZE + MARGIN * 2

# unicode chess piece symbols
PIECE_SYMBOLS = {
    chess.PAWN: {chess.WHITE: '♙', chess.BLACK: '♟'},
    chess.KNIGHT: {chess.WHITE: '♘', chess.BLACK: '♞'},
    chess.BISHOP: {chess.WHITE: '♗', chess.BLACK: '♝'},
    chess.ROOK: {chess.WHITE: '♖', chess.BLACK: '♜'},
    chess.QUEEN: {chess.WHITE: '♕', chess.BLACK: '♛'},
    chess.KING: {chess.WHITE: '♔', chess.BLACK: '♚'},
}


class ChessVisualizer:
    """
    pygame-based visual interface for playing chess.
    supports move selection, legal move highlighting, and game status display.
    """
    
    def __init__(self):
        """initialize the chess visualizer."""
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Chess AI - Visual Interface")
        
        # initialize font
        self.font_large = pygame.font.Font(None, 72)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
        # game state
        self.game = ChessGame()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        
        # game mode: 'human_vs_human' or 'human_vs_ai'
        self.game_mode = 'human_vs_human'
    
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
        """draw chess pieces on the board."""
        board = self.game.get_board()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                symbol = PIECE_SYMBOLS[piece.piece_type][piece.color]
                x, y = self.square_to_pixel(square)
                
                # center the piece in the square
                text_surface = self.font_large.render(symbol, True, BLACK if piece.color == chess.WHITE else WHITE)
                text_rect = text_surface.get_rect(center=(x + SQUARE_SIZE // 2, y + SQUARE_SIZE // 2))
                self.screen.blit(text_surface, text_rect)
    
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
                # deselect or select new piece
                piece = board.piece_at(square)
                if piece is not None and piece.color == board.turn:
                    self.selected_square = square
                    self.legal_moves_for_selected = [
                        move for move in board.legal_moves
                        if move.from_square == square
                    ]
                else:
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
    
    def run(self):
        """main game loop."""
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # left click
                        self.handle_click(event.pos[0], event.pos[1])
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.reset_game()
                    elif event.key == pygame.K_u:
                        self.undo_move()
            
            # draw everything
            self.screen.fill(WHITE)
            self.draw_board()
            self.draw_pieces()
            self.draw_info_panel()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()


if __name__ == '__main__':
    visualizer = ChessVisualizer()
    visualizer.run()

