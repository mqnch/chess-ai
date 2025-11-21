import torch
import chess

################################################################################
# BOARD ENCODING (18×8×8 planes)
################################################################################

def board_to_tensor(board):
    """
    Convert chess.Board → (18, 8, 8) tensor.
    """

    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    # Piece planes 0–11
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = square // 8
            col = square % 8
            idx = piece_types.index(piece.piece_type)
            if piece.color == chess.WHITE:
                tensor[idx, row, col] = 1.0
            else:
                tensor[idx + 6, row, col] = 1.0

    # Side to move plane (12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Castling planes 13–16
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # Move count plane (17)
    tensor[17, :, :] = min(board.fullmove_number / 100.0, 1.0)

    return tensor


################################################################################
# MOVE ENCODING CONSTANTS (8×8×73 policy)
################################################################################

QUEEN_DIRECTIONS = [
    (-1, 0),  # N
    (-1, 1),  # NE
    (0, 1),   # E
    (1, 1),   # SE
    (1, 0),   # S
    (1, -1),  # SW
    (0, -1),  # W
    (-1, -1)  # NW
]

KNIGHT_MOVES = [
    (-2, -1), (-2, 1),
    (-1, -2), (-1, 2),
    (1, -2),  (1, 2),
    (2, -1),  (2, 1)
]

# Action type ranges:
# 0–55  : queen-like (sliding) moves (8 dirs × 7 steps)
# 56–63 : knight moves
# 64–67 : quiet promotions (Q,R,B,N)
# 68–71 : capture promotions (Q,R,B,N)
# 72    : unused/reserved (AlphaZero uses only 0–71)


################################################################################
# MOVE → ACTION ENCODING
################################################################################

def move_to_action(move, board):
    """
    Convert chess.Move → (from_square, action_type).
    """

    from_sq = move.from_square
    to_sq = move.to_square

    fr = from_sq // 8
    fc = from_sq % 8
    tr = to_sq // 8
    tc = to_sq % 8

    dr = tr - fr
    dc = tc - fc

    # --- PROMOTIONS (all 8 promotion action types) ---
    if move.promotion is not None:
        prom_map = {
            chess.QUEEN: 0,
            chess.ROOK: 1,
            chess.BISHOP: 2,
            chess.KNIGHT: 3,
        }
        idx = prom_map[move.promotion]
        base = 64

        is_capture = board.is_capture(move)

        if is_capture:
            action_type = base + 4 + idx   # 68–71
        else:
            action_type = base + idx       # 64–67

        return from_sq, action_type

    # --- KNIGHT MOVES ---
    if (abs(dr), abs(dc)) in [(2,1), (1,2)]:
        for i, (kr, kc) in enumerate(KNIGHT_MOVES):
            if kr == dr and kc == dc:
                return from_sq, 56 + i

    # --- SLIDING (QUEEN-LIKE) MOVES ---
    # normalize direction
    drn = 0 if dr == 0 else (dr // abs(dr))
    dcn = 0 if dc == 0 else (dc // abs(dc))

    # find direction index
    for i, (qr, qc) in enumerate(QUEEN_DIRECTIONS):
        if (qr, qc) == (drn, dcn):
            distance = max(abs(dr), abs(dc))
            if 1 <= distance <= 7:
                action_type = i * 7 + (distance - 1)
                return from_sq, action_type

    return None


################################################################################
# ACTION → MOVE DECODING
################################################################################

def action_to_move(from_square, action_type, board):
    """
    Convert (from_square, action_type) → chess.Move.
    """
    fr = from_square // 8
    fc = from_square % 8

    # --- PROMOTIONS ---
    if 64 <= action_type <= 71:
        prom_index = action_type % 4
        prom_piece = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT][prom_index]

        piece = board.piece_at(from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return None

        direction = 8 if board.turn == chess.WHITE else -8
        to_sq = from_square + direction

        if not (0 <= to_sq < 64):
            return None

        move = chess.Move(from_square, to_sq, promotion=prom_piece)
        return move if move in board.legal_moves else None

    # --- KNIGHT MOVES (56–63) ---
    if 56 <= action_type <= 63:
        idx = action_type - 56
        dr, dc = KNIGHT_MOVES[idx]
        tr = fr + dr
        tc = fc + dc

        if 0 <= tr < 8 and 0 <= tc < 8:
            to_sq = tr * 8 + tc
            move = chess.Move(from_square, to_sq)
            return move if move in board.legal_moves else None
        return None

    # --- SLIDING (0–55) ---
    if 0 <= action_type <= 55:
        dir_idx = action_type // 7
        distance = (action_type % 7) + 1

        dr, dc = QUEEN_DIRECTIONS[dir_idx]
        tr = fr + dr * distance
        tc = fc + dc * distance

        if 0 <= tr < 8 and 0 <= tc < 8:
            to_sq = tr * 8 + tc
            move = chess.Move(from_square, to_sq)
            return move if move in board.legal_moves else None

        return None

    return None