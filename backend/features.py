import chess
import numpy as np

CENTER_SQUARES = [chess.D4, chess.E4, chess.D5, chess.E5]
EXTENDED_CENTER = [
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6
]

# ---------- MATERIAL ----------
def count_material(board, color):
    return {
        chess.PAWN: len(board.pieces(chess.PAWN, color)),
        chess.KNIGHT: len(board.pieces(chess.KNIGHT, color)),
        chess.BISHOP: len(board.pieces(chess.BISHOP, color)),
        chess.ROOK: len(board.pieces(chess.ROOK, color)),
        chess.QUEEN: len(board.pieces(chess.QUEEN, color)),
    }

# ---------- KING SAFETY ----------
def king_safety(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return 0

    k_file = chess.square_file(king_sq)
    k_rank = chess.square_rank(king_sq)

    danger = 0
    for df in [-1, 0, 1]:
        for dr in [-1, 0, 1]:
            if df == 0 and dr == 0:
                continue
            f = k_file + df
            r = k_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                danger += len(board.attackers(not color, sq))

    return max(0, 100 - danger * 8)

# ---------- CENTER CONTROL ----------
def center_control(board, color):
    return sum(len(board.attackers(color, sq)) for sq in CENTER_SQUARES)

# ---------- MOBILITY ----------
def mobility(board, color):
    temp = board.copy()
    temp.turn = color
    return temp.legal_moves.count()

# ---------- PAWN STRUCTURE ----------
def pawn_structure(board, color):
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(p) for p in pawns]

    isolated = 0
    doubled = 0

    # doubled pawns counted per file
    for f in set(files):
        if files.count(f) > 1:
            doubled += 1

    # isolated pawns
    for p in pawns:
        f = chess.square_file(p)
        if (f - 1 not in files) and (f + 1 not in files):
            isolated += 1

    # passed pawns
    passed = 0
    opponent_pawns = board.pieces(chess.PAWN, not color)
    for p in pawns:
        pf = chess.square_file(p)
        pr = chess.square_rank(p)
        is_passed = True

        for op in opponent_pawns:
            if abs(chess.square_file(op) - pf) <= 1:
                if (color == chess.WHITE and chess.square_rank(op) > pr) or \
                   (color == chess.BLACK and chess.square_rank(op) < pr):
                    is_passed = False
                    break

        if is_passed:
            passed += 1

    return isolated, doubled, passed

# ---------- SPACE ----------
def space_control(board, color):
    return sum(len(board.attackers(color, sq)) > 0 for sq in EXTENDED_CENTER)

# ---------- PIECE ACTIVITY ----------
def piece_activity(board, color):
    minors = len(board.pieces(chess.KNIGHT, color)) + len(board.pieces(chess.BISHOP, color))
    rooks = len(board.pieces(chess.ROOK, color))
    queens = len(board.pieces(chess.QUEEN, color))
    return minors, rooks, queens

# ---------- MAIN FEATURE EXTRACTION ----------
def extract_features(board: chess.Board):

    wm = count_material(board, chess.WHITE)
    bm = count_material(board, chess.BLACK)

    w_iso, w_dbl, w_pass = pawn_structure(board, chess.WHITE)
    b_iso, b_dbl, b_pass = pawn_structure(board, chess.BLACK)

    w_min, w_rook, w_q = piece_activity(board, chess.WHITE)
    b_min, b_rook, b_q = piece_activity(board, chess.BLACK)

    features = [
        wm[chess.PAWN], wm[chess.KNIGHT], wm[chess.BISHOP], wm[chess.ROOK], wm[chess.QUEEN],
        bm[chess.PAWN], bm[chess.KNIGHT], bm[chess.BISHOP], bm[chess.ROOK], bm[chess.QUEEN],

        king_safety(board, chess.WHITE),
        king_safety(board, chess.BLACK),

        center_control(board, chess.WHITE),
        center_control(board, chess.BLACK),

        mobility(board, chess.WHITE),
        mobility(board, chess.BLACK),

        w_iso, b_iso,
        w_pass, b_pass,
        w_dbl, b_dbl,

        space_control(board, chess.WHITE),
        space_control(board, chess.BLACK),

        w_min - b_min,
        w_rook - b_rook,
        w_q - b_q,

        1 if board.turn == chess.WHITE else 0
    ]

    return np.array(features, dtype=np.float32)
