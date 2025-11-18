import torch
import chess


def board_to_tensor(board): # converts a chess board position to a pytorch tensor representation
    """
    encoding scheme:
    - planes 0-5: white pieces (pawn, knight, bishop, rook, queen, king)
    - planes 6-11: black pieces (pawn, knight, bishop, rook, queen, king)
    - plane 12: side to move (all 1s if white, all 0s if black)
    - plane 13: white kingside castling right
    - plane 14: white queenside castling right
    - plane 15: black kingside castling right
    - plane 16: black queenside castling right
    - plane 17: move count (normalized, represents game phase)
    
    coordinate system: (0,0) is at bottom-left (a1 from white's perspective).
    each plane is an 8x8 array with 1.0 where the feature is present, 0.0 elsewhere.
    
    args:
        board: chess.Board object representing the position to encode.
    
    returns:
        pytorch tensor of shape (18, 8, 8) with dtype float32.
    """
    # initialize tensor with zeros: 18 planes, 8 rows, 8 columns
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
    
    # piece type mapping: order is pawn, knight, bishop, rook, queen, king
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING
    ]
    
    # encode piece positions (planes 0-11)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # convert square index to (row, col) with (0,0) at bottom-left
            # in python-chess: square 0 = a1, square 56 = a8
            # square // 8 gives rank-1 (0 for rank 1, 7 for rank 8)
            # square % 8 gives file index (0 for a, 7 for h)
            row = square // 8  # rank 1 is row 0, rank 8 is row 7
            col = square % 8   # file a is col 0, file h is col 7
            
            # find piece type index (0-5)
            piece_type_idx = piece_types.index(piece.piece_type)
            
            # determine plane index based on color
            if piece.color == chess.WHITE:
                plane_idx = piece_type_idx  # planes 0-5 for white
            else:
                plane_idx = piece_type_idx + 6  # planes 6-11 for black
            
            tensor[plane_idx, row, col] = 1.0
    
    # encode side to move (plane 12)
    # all 1s if white to move, all 0s if black to move
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    
    # encode castling rights (planes 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0  # white kingside
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0  # white queenside
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0  # black kingside
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0  # black queenside
    
    # encode move count (plane 17)
    # normalize move count to represent game phase (0.0 to 1.0)
    # using fullmove_number which counts full moves (both white and black)
    # typical game length is around 40-60 moves, so normalize by 100
    normalized_move_count = min(board.fullmove_number / 100.0, 1.0)
    tensor[17, :, :] = normalized_move_count
    
    return tensor


def tensor_to_board_state(tensor):
    """
    reconstruct a chess board state from a tensor representation.
    this is primarily useful for debugging and verification.
    
    note: this function reconstructs the position but may not perfectly
    restore all metadata (e.g., move history, repetition counts).
    the reconstructed board should match the position encoded in the tensor.
    
    args:
        tensor: pytorch tensor of shape (18, 8, 8) representing a board position.
    
    returns:
        chess.Board object representing the reconstructed position, or None if invalid.
    """
    if tensor.shape != (18, 8, 8):
        return None
    
    # create empty board
    board = chess.Board(chess.Board.empty())
    
    # piece type mapping (same order as encoding)
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING
    ]
    
    # reconstruct piece positions from planes 0-11
    for row in range(8):
        for col in range(8):
            # convert (row, col) back to square index
            # row 0 is rank 1, row 7 is rank 8
            # square = row * 8 + col
            square = row * 8 + col
            
            # check white pieces (planes 0-5)
            for piece_type_idx in range(6):
                if tensor[piece_type_idx, row, col] > 0.5:  # threshold for binary values
                    piece_type = piece_types[piece_type_idx]
                    piece = chess.Piece(piece_type, chess.WHITE)
                    board.set_piece_at(square, piece)
                    break
            
            # check black pieces (planes 6-11)
            if board.piece_at(square) is None:  # only if no white piece found
                for piece_type_idx in range(6):
                    if tensor[piece_type_idx + 6, row, col] > 0.5:
                        piece_type = piece_types[piece_type_idx]
                        piece = chess.Piece(piece_type, chess.BLACK)
                        board.set_piece_at(square, piece)
                        break
    
    # reconstruct side to move (plane 12)
    # if any cell in plane 12 is > 0.5, it's white's turn
    if tensor[12, 0, 0] > 0.5:
        board.turn = chess.WHITE
    else:
        board.turn = chess.BLACK
    
    # reconstruct castling rights (planes 13-16)
    if tensor[13, 0, 0] > 0.5:  # white kingside
        # ensure king and rook are in correct positions
        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
                board.castling_rights |= chess.BB_H1
    if tensor[14, 0, 0] > 0.5:  # white queenside
        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
                board.castling_rights |= chess.BB_A1
    if tensor[15, 0, 0] > 0.5:  # black kingside
        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
                board.castling_rights |= chess.BB_H8
    if tensor[16, 0, 0] > 0.5:  # black queenside
        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
                board.castling_rights |= chess.BB_A8
    
    # move count (plane 17) and other metadata cannot be fully restored
    # the board will have default values for these
    
    return board
