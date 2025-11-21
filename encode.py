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


# move encoding constants for 8×8×73 policy representation
# 73 action types per square:
# - 0-55: queen-like moves (8 directions × 7 squares max)
# - 56-63: knight moves (8 directions)
# - 64-72: underpromotions (3 piece types × 3 directions)
QUEEN_MOVE_OFFSET = 0
KNIGHT_MOVE_OFFSET = 56
UNDERPROMOTION_OFFSET = 64

# directions for queen moves: N, NE, E, SE, S, SW, W, NW
QUEEN_DIRECTIONS = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1)   # NW
]

# directions for knight moves
KNIGHT_MOVES = [
    (-2, -1), (-2, 1), (-1, -2), (-1, 2),
    (1, -2), (1, 2), (2, -1), (2, 1)
]

# underpromotion piece types (knight, bishop, rook)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
UNDERPROMOTION_DIRECTIONS = [(-1, 0), (-1, -1), (-1, 1)]  # straight, left-diagonal, right-diagonal


def move_to_action(move, board):
    """
    convert a chess.Move to (from_square, action_type) tuple for 8×8×73 encoding.
    
    args:
        move: chess.Move object
        board: chess.Board object (needed for context like promotions)
    
    returns:
        tuple (from_square, action_type) where:
        - from_square: 0-63 (square index)
        - action_type: 0-72 (action type index)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # convert squares to (row, col) coordinates
    from_row = from_square // 8
    from_col = from_square % 8
    to_row = to_square // 8
    to_col = to_square % 8
    
    # calculate direction and distance
    dr = to_row - from_row
    dc = to_col - from_col
    
    # handle promotions (including underpromotions)
    if move.promotion and move.promotion != chess.QUEEN:
        # underpromotion: piece type and direction
        piece_idx = UNDERPROMOTION_PIECES.index(move.promotion)
        # determine direction: straight, left-diagonal, or right-diagonal
        if dc == 0:
            dir_idx = 0  # straight
        elif dc < 0:
            dir_idx = 1  # left-diagonal
        else:
            dir_idx = 2  # right-diagonal
        action_type = UNDERPROMOTION_OFFSET + piece_idx * 3 + dir_idx
        return (from_square, action_type)
    
    # handle knight moves
    if abs(dr) == 2 and abs(dc) == 1 or abs(dr) == 1 and abs(dc) == 2:
        # find matching knight move direction
        for i, (kr, kc) in enumerate(KNIGHT_MOVES):
            if kr == dr and kc == dc:
                action_type = KNIGHT_MOVE_OFFSET + i
                return (from_square, action_type)
    
    # handle queen-like moves (including regular moves and queen promotions)
    # normalize direction
    if dr != 0:
        dr_norm = dr // abs(dr)
    else:
        dr_norm = 0
    if dc != 0:
        dc_norm = dc // abs(dc)
    else:
        dc_norm = 0
    
    # find matching direction
    dir_idx = -1
    for i, (qr, qc) in enumerate(QUEEN_DIRECTIONS):
        if qr == dr_norm and qc == dc_norm:
            dir_idx = i
            break
    
    if dir_idx == -1:
        # fallback: should not happen for valid moves
        return None
    
    # calculate distance (1-7)
    distance = max(abs(dr), abs(dc))
    if distance < 1 or distance > 7:
        return None
    
    action_type = QUEEN_MOVE_OFFSET + dir_idx * 7 + (distance - 1)
    return (from_square, action_type)


def action_to_move(from_square, action_type, board):
    """
    convert (from_square, action_type) back to a chess.Move object.
    
    args:
        from_square: 0-63 (square index)
        action_type: 0-72 (action type index)
        board: chess.Board object (needed for legal move validation)
    
    returns:
        chess.Move object, or None if invalid
    """
    from_row = from_square // 8
    from_col = from_square % 8
    
    # handle underpromotions (64-72)
    if action_type >= UNDERPROMOTION_OFFSET:
        idx = action_type - UNDERPROMOTION_OFFSET
        piece_idx = idx // 3
        dir_idx = idx % 3
        
        promotion_piece = UNDERPROMOTION_PIECES[piece_idx]
        dr, dc = UNDERPROMOTION_DIRECTIONS[dir_idx]
        
        to_row = from_row + dr
        to_col = from_col + dc
        
        # underpromotions only happen on rank 7 (for white) or rank 0 (for black)
        if to_row < 0 or to_row >= 8 or to_col < 0 or to_col >= 8:
            return None
        
        to_square = to_row * 8 + to_col
        
        # try to create move with promotion
        try:
            move = chess.Move(from_square, to_square, promotion=promotion_piece)
            if move in board.legal_moves:
                return move
        except:
            pass
        return None
    
    # handle knight moves (56-63)
    if action_type >= KNIGHT_MOVE_OFFSET and action_type < UNDERPROMOTION_OFFSET:
        idx = action_type - KNIGHT_MOVE_OFFSET
        dr, dc = KNIGHT_MOVES[idx]
        to_row = from_row + dr
        to_col = from_col + dc
        
        if to_row < 0 or to_row >= 8 or to_col < 0 or to_col >= 8:
            return None
        
        to_square = to_row * 8 + to_col
        
        # check if it's a promotion (knight move to promotion square)
        if (to_row == 0 or to_row == 7) and board.piece_at(from_square) and \
           board.piece_at(from_square).piece_type == chess.PAWN:
            # try with queen promotion first
            try:
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    return move
            except:
                pass
        
        try:
            move = chess.Move(from_square, to_square)
            if move in board.legal_moves:
                return move
        except:
            pass
        return None
    
    # handle queen-like moves (0-55)
    if action_type < KNIGHT_MOVE_OFFSET:
        dir_idx = action_type // 7
        distance = (action_type % 7) + 1
        
        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        to_row = from_row + dr * distance
        to_col = from_col + dc * distance
        
        if to_row < 0 or to_row >= 8 or to_col < 0 or to_col >= 8:
            return None
        
        to_square = to_row * 8 + to_col
        
        # check if it's a promotion (pawn move to promotion square)
        if (to_row == 0 or to_row == 7) and board.piece_at(from_square) and \
           board.piece_at(from_square).piece_type == chess.PAWN:
            # try with queen promotion
            try:
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    return move
            except:
                pass
        
        try:
            move = chess.Move(from_square, to_square)
            if move in board.legal_moves:
                return move
        except:
            pass
        return None
    
    return None
