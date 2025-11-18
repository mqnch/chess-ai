import chess


class ChessGame:
    # fen is a string representation of the board position
    def __init__(self, fen=None): 
        if fen is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen)
    
    def get_legal_moves(self):
        return list(self.board.legal_moves)
    
    def make_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False
    
    def undo_move(self):
        if len(self.board.move_stack) > 0:
            self.board.pop()
            return True
        return False
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def get_result(self):
        return self.board.result()
    
    def get_winner(self):
        if not self.is_terminal():
            return None
        
        result = self.get_result()
        if result == '1-0':
            return 1  # white wins
        elif result == '0-1':
            return -1  # black wins
        else:
            return 0  # draw
    
    def get_board(self):
        return self.board
    
    def is_checkmate(self):
        return self.board.is_checkmate()
    
    def is_stalemate(self):
        return self.board.is_stalemate()
    
    def is_draw(self):
        return self.board.is_game_over() and self.get_result() == '1/2-1/2'
    
    def get_turn(self):
        return self.board.turn
    
    # creates a deep copy of the current game state
    def copy(self): 
        return ChessGame(fen=self.board.fen())
