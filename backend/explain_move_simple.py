import os
import chess
import chess.engine
import shap
import numpy as np
import pandas as pd
import torch
from features import extract_features
from model import ChessEvalNet

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BACKEND_DIR)
ENGINE_PATH = "/Users/gaurangpandey/stockfish/stockfish"
MODEL_PATH = os.path.join(BACKEND_DIR, "chess_eval_nn.pt")
DATA_PATH = os.path.join(PROJECT_DIR, "data", "positions.csv")

SQUARE_NAMES = ["a1","b1","c1","d1","e1","f1","g1","h1",
                "a2","b2","c2","d2","e2","f2","g2","h2",
                "a3","b3","c3","d3","e3","f3","g3","h3",
                "a4","b4","c4","d4","e4","f4","g4","h4",
                "a5","b5","c5","d5","e5","f5","g5","h5",
                "a6","b6","c6","d6","e6","f6","g6","h6",
                "a7","b7","c7","d7","e7","f7","g7","h7",
                "a8","b8","c8","d8","e8","f8","g8","h8"]

PIECE_SYMBOLS = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king",
                 "P": "pawn", "N": "knight", "B": "bishop", "R": "rook", "Q": "queen", "K": "king"}

PIECE_VALUES = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 100}

FEATURE_NAMES = [
    "white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens",
    "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens",
    "white_king_safety", "black_king_safety",
    "white_center_control", "black_center_control",
    "white_mobility", "black_mobility",
    "white_isolated_pawns", "black_isolated_pawns",
    "white_passed_pawns", "black_passed_pawns",
    "white_doubled_pawns", "black_doubled_pawns",
    "white_space", "black_space",
    "minor_piece_diff", "rook_diff", "queen_diff",
    "white_turn"
]

EQUIVALENCE_THRESHOLD = 0.3

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

_model = None
_explainer = None
_mean = None
_std = None


def _load_xai_components():
    global _model, _explainer, _mean, _std
    if _model is not None:
        return _model, _explainer, _mean, _std
    df = pd.read_csv(DATA_PATH)
    X = df.iloc[:, :-1].values.astype(np.float32)
    _mean = X.mean(axis=0)
    _std = X.std(axis=0) + 1e-6
    _model = ChessEvalNet()
    _model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    _model.eval()
    background = np.zeros((1, len(FEATURE_NAMES)))
    background_norm = (background - _mean) / _std
    background_tensor = torch.tensor(background_norm).float()
    _explainer = shap.DeepExplainer(_model, background_tensor)
    return _model, _explainer, _mean, _std


def _get_shap_explanation(board):
    model, explainer, mean, std = _load_xai_components()
    features = extract_features(board)
    features_norm = (np.array(features) - mean) / std
    x = torch.tensor(features_norm).float().unsqueeze(0)
    shap_values = explainer.shap_values(x)
    prediction = model(x).item()
    return shap_values[0], prediction, features


def square_to_name(sq):
    return SQUARE_NAMES[sq]


# ═══════════════════════════════════════════════════════════════════
#  POSITION CONTEXT
# ═══════════════════════════════════════════════════════════════════

def get_game_phase(board):
    total = len(board.piece_map())
    if total > 28:
        return "opening"
    elif total > 14:
        return "middlegame"
    return "endgame"


def get_center_pawn_structure(board):
    center_files = [chess.D4, chess.E4, chess.D5, chess.E5,
                    chess.C3, chess.D3, chess.E3, chess.F3,
                    chess.C4, chess.D4, chess.E4, chess.F4,
                    chess.C5, chess.D5, chess.E5, chess.F5,
                    chess.C6, chess.D6, chess.E6, chess.F6]
    pawns_in_center = 0
    for sq in center_files:
        p = board.piece_at(sq)
        if p and p.piece_type == chess.PAWN:
            pawns_in_center += 1
    if pawns_in_center >= 4:
        return "closed", "pawn-heavy center — the position is closed and requires careful maneuvering"
    elif pawns_in_center >= 2:
        return "semi-open", "semi-open center with some pawn tension — piece activity matters"
    else:
        return "open", "open center with few pawns — piece activity and initiative are paramount"


def get_pawn_structure_description(board, color):
    """Identify pawn structure features with descriptive text."""
    pawns = board.pieces(chess.PAWN, color)
    files_with_pawns = set(chess.square_file(p) for p in pawns)

    has_doubled = False
    for f in files_with_pawns:
        if sum(1 for p in pawns if chess.square_file(p) == f) > 1:
            has_doubled = True
            break

    isolated = 0
    for p in pawns:
        f = chess.square_file(p)
        if (f - 1 not in files_with_pawns) and (f + 1 not in files_with_pawns):
            isolated += 1

    passed = 0
    enemy = not color
    enemy_pawns = board.pieces(chess.PAWN, enemy)
    for p in pawns:
        pf = chess.square_file(p)
        pr = chess.square_rank(p)
        is_passed = True
        for op in enemy_pawns:
            if abs(chess.square_file(op) - pf) <= 1:
                if (color == chess.WHITE and chess.square_rank(op) > pr) or \
                   (color == chess.BLACK and chess.square_rank(op) < pr):
                    is_passed = False
                    break
        if is_passed:
            passed += 1

    parts = []
    if isolated > 0:
        parts.append(f"{isolated} isolated")
    if has_doubled:
        parts.append("doubled")
    if passed > 0:
        parts.append(f"{passed} passed")
    if not parts:
        return "sound"
    return ", ".join(parts)


# ═══════════════════════════════════════════════════════════════════
#  KING SAFETY
# ═══════════════════════════════════════════════════════════════════

def analyze_king_safety(board, color):
    king_sq = board.king(color)
    if king_sq is None:
        return {"status": "error", "description": "No king found", "square": "?"}

    k_file = chess.square_file(king_sq)
    k_rank = chess.square_rank(king_sq)
    enemy = not color

    pawn_shield = 0
    for df in [-1, 0, 1]:
        f = k_file + df
        if 0 <= f <= 7:
            for dr in range(1, 4):
                r = k_rank + (dr if color == chess.WHITE else -dr)
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        pawn_shield += 1

    flank_cover = 0
    for df in [-1, 0, 1]:
        f = k_file + df
        if 0 <= f <= 7:
            for dr in range(-1, 2):
                r = k_rank + dr
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.color == color and p.piece_type != chess.KING:
                        flank_cover += 0.5

    attackers_near = 0
    for df in [-2, -1, 0, 1, 2]:
        for dr in [-2, -1, 0, 1, 2]:
            if df == 0 and dr == 0:
                continue
            f = k_file + df
            r = k_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                attackers_near += len(board.attackers(enemy, sq))

    open_files = 0
    for df in [-1, 0, 1]:
        f = k_file + df
        if 0 <= f <= 7:
            own_pawn_on_file = False
            for r in range(8):
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    own_pawn_on_file = True
                    break
            if not own_pawn_on_file:
                open_files += 1

    has_castled = (color == chess.WHITE and king_sq in (chess.G1, chess.C1)) or \
                  (color == chess.BLACK and king_sq in (chess.G8, chess.C8))

    total_shelter = pawn_shield + flank_cover

    lines = []
    if has_castled:
        side = "kingside" if king_sq in (chess.G1, chess.G8) else "queenside"
        lines.append(f"castled {side}")
    else:
        if total_shelter >= 2:
            lines.append("still in the center but has some cover")
        else:
            lines.append("still in the center and exposed")

    if pawn_shield >= 3:
        lines.append(f"protected by {pawn_shield} pawns nearby")
    elif pawn_shield >= 1:
        lines.append(f"only {pawn_shield} nearby shield pawn{'s' if pawn_shield > 1 else ''}")
    else:
        lines.append("no pawn shield at all")

    if open_files > 0:
        lines.append(f"{open_files} open file{'s' if open_files > 1 else ''} adjacent")
    if attackers_near > 4:
        lines.append(f"under threat from {attackers_near} enemy attacks nearby")

    if total_shelter >= 4 and attackers_near <= 2 and open_files <= 1:
        status = "fortress"
        desc = "a fortress — well-fortified behind a solid pawn shield"
    elif total_shelter >= 3 and attackers_near <= 4:
        status = "safe"
        desc = "comfortable — good pawn cover keeps the king safe"
    elif total_shelter >= 2:
        if attackers_near > 6:
            status = "pressured"
            desc = "under pressure — enemy pieces are circling despite some cover"
        else:
            status = "adequate"
            desc = "adequate — the shield is thin but holding for now"
    elif total_shelter >= 1:
        status = "exposed"
        desc = "exposed — minimal pawn cover, consider improving king safety"
    else:
        status = "vulnerable"
        desc = "vulnerable — critically exposed with zero pawn shelter"

    if has_castled:
        side = "kingside" if king_sq in (chess.G1, chess.G8) else "queenside"
        desc += f" ({side} castle)"

    return {
        "status": status,
        "description": desc,
        "summary": "; ".join(lines),
        "shelterPawns": int(pawn_shield),
        "flankCover": int(flank_cover),
        "attackersNear": attackers_near,
        "openFilesNear": open_files,
        "hasCastled": has_castled,
        "square": square_to_name(king_sq),
    }


# ═══════════════════════════════════════════════════════════════════
#  TACTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════

def detect_forks(board, move, temp_board, is_white, piece_name):
    """Detect if the moved piece forks multiple enemy pieces."""
    moved = move.to_square
    to_rank = chess.square_rank(moved)
    to_file = chess.square_file(moved)
    attacked_pieces = []
    for df in (-2, -1, 0, 1, 2):
        for dr in (-2, -1, 0, 1, 2):
            if df == 0 and dr == 0:
                continue
            f, r = to_file + df, to_rank + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                sq = chess.square(f, r)
                if board.is_attacked_by(not is_white, sq):
                    continue
                if temp_board.is_attacked_by(is_white, sq):
                    p = board.piece_at(sq)
                    if p and p.color != is_white:
                        attacked_pieces.append((sq, PIECE_SYMBOLS.get(p.piece_type, "?"), PIECE_VALUES.get(PIECE_SYMBOLS.get(p.piece_type, "piece"), 0)))
    undef_high_value = [(s, n, v) for s, n, v in attacked_pieces if v >= 5 or (v >= 3 and n != "pawn")]
    targets = list(set(n for _, n, _ in attacked_pieces))
    if len(undef_high_value) >= 2:
        names = ", ".join(n for _, n, _ in undef_high_value[:3])
        return f"Fork! The {piece_name} attacks multiple valuable pieces: {names}"
    if len(targets) >= 3:
        return f"The {piece_name} attacks multiple enemy pieces from its new square"
    return None


def detect_pins(board, move, temp_board, is_white):
    """Detect pins created by the move."""
    enemy = not is_white
    enemy_king = board.king(enemy)
    if enemy_king is None:
        return None
    pins = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == enemy and p.piece_type != chess.KING:
            if temp_board.is_pinned(enemy, sq):
                name = PIECE_SYMBOLS.get(p.piece_type, "piece")
                pins.append((sq, name))
    if pins:
        return f"Creates a pin on {', '.join(n for _, n in pins[:2])}"
    return None


def detect_discovered_attack(board, move, temp_board, is_white):
    """Detect if the move uncovers an attack behind the moved piece."""
    enemy = not is_white
    enemy_king = board.king(enemy)
    from_sq = move.from_square

    behind = None
    f = chess.square_file(from_sq)
    r = chess.square_rank(from_sq)
    mf = chess.square_file(move.to_square)
    mr = chess.square_rank(move.to_square)

    if f == mf:
        dr = 1 if mr > r else -1
        for step in range(1, 8):
            sq = chess.square(f, r + dr * step)
            if 0 <= chess.square_rank(sq) <= 7:
                p = board.piece_at(sq)
                if p and p.color == is_white:
                    behind = sq
                    break
            else:
                break
    elif r == mr:
        df = 1 if mf > f else -1
        for step in range(1, 8):
            sq = chess.square(f + df * step, r)
            if 0 <= chess.square_file(sq) <= 7:
                p = board.piece_at(sq)
                if p and p.color == is_white:
                    behind = sq
                    break
            else:
                break

    if behind is not None:
        piece_behind = board.piece_at(behind)
        if piece_behind and temp_board.is_attacked_by(is_white, enemy_king):
            return "Discovered check! The moving piece reveals an attack on the enemy king"
        for sq2 in chess.SQUARES:
            p2 = board.piece_at(sq2)
            if p2 and p2.color == enemy and p2.piece_type != chess.KING:
                if temp_board.is_attacked_by(is_white, sq2) and not board.is_attacked_by(is_white, sq2):
                    name2 = PIECE_SYMBOLS.get(p2.piece_type, "piece")
                    return f"Discovered attack on the enemy {name2}"
    return None


def find_undefended_pieces(temp_board, is_white):
    """Find undefended enemy pieces after the move."""
    enemy = not is_white
    undefended = []
    for sq in chess.SQUARES:
        p = temp_board.piece_at(sq)
        if p and p.color == enemy and p.piece_type != chess.KING:
            if len(temp_board.attackers(enemy, sq)) == 0 and len(temp_board.attackers(is_white, sq)) > 0:
                name = PIECE_SYMBOLS.get(p.piece_type, "piece")
                val = PIECE_VALUES.get(name, 0)
                undefended.append((name, val, sq))
    undefended.sort(key=lambda x: -x[1])
    return undefended


def analyze_tactics(board, move, temp_board, is_white, piece_name):
    """Comprehensive tactical analysis of a move. Returns list of finding strings."""
    findings = []

    fork = detect_forks(board, move, temp_board, is_white, piece_name)
    if fork:
        findings.append(fork)

    pin = detect_pins(board, move, temp_board, is_white)
    if pin:
        findings.append(pin)

    discovery = detect_discovered_attack(board, move, temp_board, is_white)
    if discovery:
        findings.append(discovery)

    undefended = find_undefended_pieces(temp_board, is_white)
    if undefended:
        targets = [f"{name} on {square_to_name(sq)}" for name, val, sq in undefended[:2]]
        undef_str = ", ".join(targets)
        if undefended[0][1] >= 5:
            findings.append(f"Targets undefended high-value piece: {undef_str}")
        else:
            findings.append(f"Leaves enemy {undef_str} hanging (undefended)")

    return findings


# ═══════════════════════════════════════════════════════════════════
#  PIECE ACTIVITY
# ═══════════════════════════════════════════════════════════════════

def get_piece_activity(board, color):
    """Return a description of how active each piece type is."""
    pieces = []
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color and p.piece_type not in (chess.KING, chess.PAWN):
            name = PIECE_SYMBOLS.get(p.piece_type, "piece")
            sq_name = SQUARE_NAMES[sq]
            mobility_on_sq = len(board.attacks(sq))
            attackers = len(board.attackers(not color, sq))
            defenders = len(board.attackers(color, sq))
            pieces.append({
                "type": name,
                "square": sq_name,
                "mobility": mobility_on_sq,
                "attacked": attackers,
                "defended": defenders,
            })

    piece_types = set(p["type"] for p in pieces)
    lines = []
    for pt in ["knight", "bishop", "rook", "queen"]:
        same = [p for p in pieces if p["type"] == pt]
        if not same:
            continue
        active = sum(1 for p in same if p["mobility"] >= 3)
        passive = sum(1 for p in same if p["mobility"] <= 1)
        if len(same) == 1:
            s = same[0]
            if s["mobility"] >= 5:
                lines.append(f"{pt} on {s['square']} is very active ({s['mobility']} moves)")
            elif s["mobility"] >= 3:
                lines.append(f"{pt} on {s['square']} is reasonably active ({s['mobility']} moves)")
            elif s["mobility"] <= 1:
                lines.append(f"{pt} on {s['square']} is poorly placed (only {s['mobility']} move{'s' if s['mobility'] != 1 else ''})")
        else:
            if active > passive:
                lines.append(f"the {pt}s are active overall")
            elif passive > active:
                lines.append(f"some {pt}s are passively placed")

    return lines


# ═══════════════════════════════════════════════════════════════════
#  MOVE COMPARISON
# ═══════════════════════════════════════════════════════════════════

def compare_top_moves(board, best_move, top_moves_info):
    """Generate comparative reasoning between the best move and alternatives."""
    if not top_moves_info or len(top_moves_info) <= 1:
        return []

    is_white = board.turn
    best_eval = top_moves_info[0]["eval"]
    comparions = []

    for alt in top_moves_info[1:]:
        alt_move = alt["move"]
        diff = abs(best_eval - alt["eval"])

        alt_eval_adj = alt["eval"] if is_white else -alt["eval"]

        if diff <= EQUIVALENCE_THRESHOLD:
            comparions.append(
                f"{alt['san']} ({alt_eval_adj:+.2f}) is equally strong — both moves lead to similar evaluations"
            )
        elif diff <= 0.6:
            comparions.append(
                f"{alt['san']} ({alt_eval_adj:+.2f}) is a reasonable alternative but slightly less precise ({diff:.2f} diff)"
            )
        else:
            comparions.append(
                f"{alt['san']} ({alt_eval_adj:+.2f}) is notably weaker — the chosen move is clearly better"
            )

    return comparions


# ═══════════════════════════════════════════════════════════════════
#  MOVE REASONING
# ═══════════════════════════════════════════════════════════════════

def analyze_move_reasoning(board, move, top_moves_info=None):
    reasons = []
    is_white = board.turn
    opponent = "Black" if is_white else "White"

    from_sq = square_to_name(move.from_square)
    to_sq = square_to_name(move.to_square)
    moving_piece = board.piece_at(move.from_square)
    piece_name = PIECE_SYMBOLS.get(moving_piece.piece_type, "piece") if moving_piece else "piece"

    temp_board = board.copy()
    temp_board.push(move)

    # ── Checkmate / Stalemate ──
    if temp_board.is_checkmate():
        reasons.append(f"Checkmate! Forces {opponent} to resign immediately")
        return reasons
    if temp_board.is_stalemate():
        reasons.append(f"Stalemate! {opponent} has no legal moves — the game is drawn")
        return reasons

    # ── Position context ──
    phase = get_game_phase(board)
    center_type, center_desc = get_center_pawn_structure(board)
    if phase == "opening":
        reasons.append(f"{center_desc.capitalize()}")
    elif phase == "endgame":
        reasons.append(f"Endgame — material imbalances and king activity become decisive")

    # ── Tactical analysis ──
    tactics = analyze_tactics(board, move, temp_board, is_white, piece_name)
    reasons.extend(tactics)

    # ── Check ──
    if temp_board.is_check():
        k_sq = square_to_name(temp_board.king(not is_white))
        reasons.append(f"Delivers check — the {opponent.lower()} king on {k_sq} must respond")

    # ── Captures ──
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            captured_name = PIECE_SYMBOLS.get(captured.piece_type, "piece")
            our_val = PIECE_VALUES.get(piece_name, 0)
            their_val = PIECE_VALUES.get(captured_name, 0)
            diff = their_val - our_val
            if their_val > our_val:
                reasons.append(f"Winning material: {piece_name} captures {captured_name} (net +{diff})")
            elif their_val == our_val:
                reasons.append(f"Equal trade: {piece_name} exchanges for {captured_name}")
            else:
                reasons.append(f"Losing trade: trades down {piece_name} for {captured_name} (net {diff})")

    # ── Promotions ──
    if move.promotion:
        promo_name = PIECE_SYMBOLS.get(move.promotion, "queen")
        reasons.append(f"Pawn promotes to {promo_name} — a decisive material gain")

    # ── En passant ──
    if board.is_en_passant(move):
        ep_sq = square_to_name(move.to_square + (-8 if is_white else 8))
        reasons.append(f"En passant capture — removes the enemy pawn on {ep_sq}")

    # ── Center control impact ──
    center_sqs = [chess.D4, chess.E4, chess.D5, chess.E5]
    before_ctrl = sum(1 for sq in center_sqs if board.is_attacked_by(is_white, sq))
    after_ctrl = sum(1 for sq in center_sqs if temp_board.is_attacked_by(is_white, sq))
    center_gain = after_ctrl - before_ctrl
    if center_gain > 0:
        reasons.append(f"Gains center influence (now attacks {after_ctrl}/4 center squares, was {before_ctrl})")
    elif center_gain < 0 and after_ctrl <= 1:
        reasons.append(f"Gives up center presence — only {after_ctrl}/4 center squares now covered")

    # ── Extended center spatial control ──
    ext_sqs = [chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.D4, chess.E4, chess.F4,
               chess.C5, chess.D5, chess.E5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6]
    ext_ctrl = sum(1 for sq in ext_sqs if temp_board.is_attacked_by(is_white, sq))
    if ext_ctrl >= 10:
        reasons.append(f"Strong spatial grip — controls {ext_ctrl}/16 extended center squares")

    # ── Piece development ──
    total_pieces = len(board.piece_map())
    if piece_name == "king" and abs(chess.square_file(move.from_square) - chess.square_file(move.to_square)) == 2:
        side = "kingside" if chess.square_file(move.to_square) > chess.square_file(move.from_square) else "queenside"
        reasons.append(f"Castles {side} — king to safety, rook activates on the {side[0]} file")
    elif piece_name == "rook" and from_sq in ("h1", "a1", "h8", "a8") and not temp_board.has_castling_rights(is_white):
        reasons.append(f"Voluntarily forfeits castling rights — the rook moves out early")
    elif total_pieces > 20 and piece_name in ("knight", "bishop"):
        starting_sqs = {"b1", "g1", "c1", "f1", "b8", "g8", "c8", "f8"}
        if from_sq in starting_sqs:
            reasons.append(f"Develops the {piece_name} to {to_sq} — brings a piece into the game")
        elif temp_board.is_attacked_by(is_white, move.to_square):
            reasons.append(f"Centralizes the {piece_name} to an active square with options")

    # ── King safety change ──
    old_king = board.king(is_white)
    new_king = temp_board.king(is_white)
    old_shelter = _count_pawn_shield(board, is_white, old_king)
    new_shelter = _count_pawn_shield(temp_board, is_white, new_king)
    shelter_diff = new_shelter - old_shelter
    if shelter_diff > 1:
        reasons.append(f"Bolsters king safety — the pawn shield gets stronger")
    elif shelter_diff < -1:
        reasons.append(f"Damages the king's pawn shield — exposes the king slightly")

    # ── Alternative moves context ──
    if top_moves_info and len(top_moves_info) > 1:
        compars = compare_top_moves(board, move, top_moves_info)
        reasons.extend(compars)

    if not reasons:
        reasons.append(f"Develops {piece_name} to {to_sq} — a solid positional move")

    return reasons


def _count_pawn_shield(board, color, king_sq):
    if king_sq is None:
        return 0
    kf = chess.square_file(king_sq)
    kr = chess.square_rank(king_sq)
    count = 0
    for df in (-1, 0, 1):
        f = kf + df
        if 0 <= f <= 7:
            for dr in (1, 2, 3):
                r = kr + (dr if color == chess.WHITE else -dr)
                if 0 <= r <= 7:
                    sq = chess.square(f, r)
                    p = board.piece_at(sq)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        count += 1
    return count


# ═══════════════════════════════════════════════════════════════════
#  STOCKFISH MULTI-PV
# ═══════════════════════════════════════════════════════════════════

def get_top_moves(board, num_moves=3, depth=14):
    results = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=num_moves)
    top_moves = []
    for result in results:
        if "pv" in result and result["pv"]:
            move = result["pv"][0]
            score_obj = result["score"].white()
            score = score_obj.score(mate_score=10000)
            p = board.piece_at(move.from_square)
            pn = PIECE_SYMBOLS.get(p.piece_type, "piece") if p else "piece"
            top_moves.append({
                "move": move,
                "san": board.san(move),
                "eval": score / 100.0,
                "piece": pn,
                "from": SQUARE_NAMES[move.from_square],
                "to": SQUARE_NAMES[move.to_square],
            })
    return top_moves


def get_grouped_top_moves(board, num_moves=3, depth=14, threshold=EQUIVALENCE_THRESHOLD):
    top = get_top_moves(board, num_moves, depth)
    if not top:
        return [], [], ""
    best_eval = top[0]["eval"]
    equivalent = [m for m in top if abs(m["eval"] - best_eval) <= threshold]
    other = [m for m in top if abs(m["eval"] - best_eval) > threshold]
    if len(equivalent) > 1:
        label = " or ".join(m["san"] for m in equivalent)
    else:
        label = equivalent[0]["san"] if equivalent else top[0]["san"]
    return equivalent, other, label


# ═══════════════════════════════════════════════════════════════════
#  SHAP INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

def _interpret_shap_features(shap_vals, features, board, is_white):
    pairs = sorted(zip(FEATURE_NAMES, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    explanations = []

    white_material = sum(features[i] * v for i, v in enumerate([1, 3, 3, 5, 9]))
    black_material = sum(features[5 + i] * v for i, v in enumerate([1, 3, 3, 5, 9]))
    material_diff = white_material - black_material

    king_info = {}
    for color, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        king_info[name] = analyze_king_safety(board, color)

    # ── Material ──
    if material_diff > 2:
        explanations.append(
            f"Material (+{material_diff:.1f}): {_describe_material(float(material_diff))} — an important factor in the evaluation"
        )
    elif material_diff < -2:
        explanations.append(
            f"Material ({material_diff:.1f}): {_describe_material(float(-material_diff))} — an important factor in the evaluation"
        )
    elif abs(material_diff) <= 1:
        explanations.append(f"Material is even — the evaluation rests on positional factors")

    # ── Top SHAP features ──
    for name, val in pairs:
        if abs(val) < 0.08:
            continue

        if name == "white_king_safety":
            ki = king_info["white"]
            if val > 0.2:
                explanations.append(f"White king on {ki['square']}: {ki['description']} (+{val:.2f} contribution)")
            elif val < -0.2:
                explanations.append(f"White king on {ki['square']}: {ki['description']} ({val:.2f} contribution — a liability)")

        elif name == "black_king_safety":
            ki = king_info["black"]
            if val > 0.2:
                explanations.append(f"Black king on {ki['square']}: {ki['description']} (+{val:.2f} contribution)")
            elif val < -0.2:
                explanations.append(f"Black king on {ki['square']}: {ki['description']} ({val:.2f} contribution — a liability)")

        elif name == "white_center_control":
            wc, bc = int(features[12]), int(features[13])
            if wc > bc + 1:
                explanations.append(f"Center control ({wc} vs {bc}): White dominates the center squares")
            elif val > 0.15:
                explanations.append(f"Center: White exerts influence ({wc}/4 center squares)")

        elif name == "black_center_control":
            wc, bc = int(features[12]), int(features[13])
            if bc > wc + 1:
                explanations.append(f"Center control ({bc} vs {wc}): Black controls the center")
            elif val > 0.15:
                explanations.append(f"Center: Black exerts influence ({bc}/4 center squares)")

        elif name == "white_mobility":
            wm, bm = int(features[14]), int(features[15])
            if wm > bm + 5:
                explanations.append(f"Activity gap ({wm} vs {bm}): White's pieces are significantly more mobile")
            elif val > 0.15:
                explanations.append(f"Piece activity: White has {wm} legal moves available")

        elif name == "black_mobility":
            wm, bm = int(features[14]), int(features[15])
            if bm > wm + 5:
                explanations.append(f"Activity gap ({bm} vs {wm}): Black's pieces are significantly more mobile")
            elif val > 0.15:
                explanations.append(f"Piece activity: Black has {bm} legal moves available")

        elif name == "white_passed_pawns" and val > 0.15 and features[18] > 0:
            n = int(features[18])
            explanations.append(f"Passed pawn{'s' if n > 1 else ''}: White has {n} passed pawn{'s' if n > 1 else ''} — a long-term asset")

        elif name == "black_passed_pawns" and val > 0.15 and features[19] > 0:
            n = int(features[19])
            explanations.append(f"Passed pawn{'s' if n > 1 else ''}: Black has {n} passed pawn{'s' if n > 1 else ''} — a long-term asset")

        elif name == "white_isolated_pawns" and val < -0.15 and features[16] > 0:
            n = int(features[16])
            explanations.append(f"Weakness: White has {n} isolated pawn{'s' if n > 1 else ''} that could be targeted")

        elif name == "black_isolated_pawns" and val < -0.15 and features[17] > 0:
            n = int(features[17])
            explanations.append(f"Weakness: Black has {n} isolated pawn{'s' if n > 1 else ''} that could be targeted")

        elif name == "white_doubled_pawns" and val < -0.15 and features[20] > 0:
            n = int(features[20])
            explanations.append(f"Structural weakness: White has doubled pawn{'s' if n > 1 else ''}")

        elif name == "black_doubled_pawns" and val < -0.15 and features[21] > 0:
            n = int(features[21])
            explanations.append(f"Structural weakness: Black has doubled pawn{'s' if n > 1 else ''}")

        elif name == "white_space" and abs(val) > 0.15:
            ws, bs = int(features[22]), int(features[23])
            if ws > bs + 3:
                explanations.append(f"Space advantage ({ws} vs {bs}): White has more breathing room")
            elif val > 0.1:
                explanations.append(f"Space: White controls {ws} extended center squares")

        elif name == "black_space" and abs(val) > 0.15:
            ws, bs = int(features[22]), int(features[23])
            if bs > ws + 3:
                explanations.append(f"Space advantage ({bs} vs {ws}): Black has more breathing room")
            elif val > 0.1:
                explanations.append(f"Space: Black controls {bs} extended center squares")

        elif name == "minor_piece_diff" and abs(val) > 0.2:
            diff = int(features[24])
            if diff > 1:
                explanations.append(f"Knights and bishops: White has +{diff} more active minor piece{'s' if diff > 1 else ''}")
            elif diff < -1:
                explanations.append(f"Knights and bishops: Black has +{-diff} more active minor piece{'s' if -diff > 1 else ''}")

        elif name == "rook_diff" and abs(val) > 0.2:
            diff = int(features[25])
            if diff > 0:
                explanations.append(f"Rook activity favors White (+{diff})")
            elif diff < 0:
                explanations.append(f"Rook activity favors Black ({diff})")

        elif name == "queen_diff" and abs(val) > 0.2:
            diff = int(features[26])
            if diff > 0:
                explanations.append(f"Queen positioning favors White (+{diff})")
            elif diff < 0:
                explanations.append(f"Queen positioning favors Black ({diff})")

        elif name in ("white_pawns", "white_knights", "white_bishops", "white_rooks", "white_queens",
                      "black_pawns", "black_knights", "black_bishops", "black_rooks", "black_queens"):
            idx_map = {"white_pawns": 0, "white_knights": 1, "white_bishops": 2, "white_rooks": 3, "white_queens": 4,
                       "black_pawns": 5, "black_knights": 6, "black_bishops": 7, "black_rooks": 8, "black_queens": 9}
            idx = idx_map[name]
            if abs(val) > 0.3 and features[idx] > 0:
                side = "White" if "white" in name else "Black"
                piece = name.split("_", 1)[1]
                explanations.append(f"{side} has {int(features[idx])} {piece}{'s' if features[idx] > 1 else ''} on the board")

    return explanations


def _describe_material(advantage):
    if advantage > 5:
        return "a decisive material advantage"
    elif advantage > 3:
        return "the equivalent of a minor piece"
    elif advantage > 1:
        return "at least a pawn to the good"
    return "slightly ahead in material"


# ═══════════════════════════════════════════════════════════════════
#  MAIN API
# ═══════════════════════════════════════════════════════════════════

def explain_best_move(fen):
    return _build_explanation_text(fen)


def get_stockfish_evaluation(board, depth=15):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=10000)
    if score is None:
        return 0.0
    return score / 100.0


def explain_best_move_dict(fen, beforeFen=None, lastPlayed=None):
    board = chess.Board(fen)
    is_white = board.turn
    current_player = "White" if is_white else "Black"

    equivalent_moves, other_moves, primary_label = get_grouped_top_moves(board, num_moves=3, depth=14)
    if not equivalent_moves:
        return {"error": "Failed to find any moves"}

    best = equivalent_moves[0]
    best_move = best["move"]
    eval_before = best["eval"]

    played_move_info = None
    before_top_suggestions = []
    if lastPlayed and beforeFen:
        try:
            before = chess.Board(beforeFen)
            played_by = "White" if before.turn else "Black"
            eval_before_player = get_stockfish_evaluation(before, depth=12)
            best_from_before = get_grouped_top_moves(before, num_moves=3, depth=14)
            eval_best_from_before = eval_before_player
            if best_from_before and best_from_before[0]:
                t2 = chess.Board(before.fen())
                t2.push(best_from_before[0][0]["move"])
                eval_best_from_before = get_stockfish_evaluation(t2, depth=12)
            for bm in (best_from_before[0] + best_from_before[1]) if best_from_before else []:
                if bm["san"] != lastPlayed:
                    before_top_suggestions.append({"san": bm["san"], "eval": round(bm["eval"], 2)})
            after = chess.Board(fen)
            eval_after_player = get_stockfish_evaluation(after, depth=12)
            played_move_info = {
                "san": lastPlayed,
                "playedBy": played_by,
                "evalBefore": round(eval_before_player, 2),
                "evalAfter": round(eval_after_player, 2),
                "bestEvalAfter": round(eval_best_from_before, 2),
            }
        except Exception as e:
            played_move_info = {"san": lastPlayed, "evalAfter": None}

    move_reasons = analyze_move_reasoning(board, best_move, top_moves_info=equivalent_moves + other_moves)

    board.push(best_move)
    eval_after = get_stockfish_evaluation(board, depth=12)

    shap_vals, nn_pred, features = _get_shap_explanation(board)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.flatten()
    shap_explanations = _interpret_shap_features(shap_vals, features, board, is_white)

    king_white = analyze_king_safety(board, chess.WHITE)
    king_black = analyze_king_safety(board, chess.BLACK)

    eval_change = eval_after - eval_before

    def format_eval(es):
        if es is None:
            return "0.00 (equal)"
        if abs(es) > 4:
            s = "winning"
        elif abs(es) > 2:
            s = "strong advantage"
        elif abs(es) > 1:
            s = "moderate advantage"
        elif abs(es) > 0.5:
            s = "slight advantage"
        else:
            s = "roughly equal"
        prefix = "+" if es > 0 else ""
        return f"{prefix}{es:.2f} ({s})"

    all_top = equivalent_moves + other_moves
    top_moves_data = [
        {"san": m["san"], "eval": round(m["eval"], 2), "piece": m["piece"], "from": m["from"], "to": m["to"]}
        for m in all_top
    ]

    return {
        "bestMove": {
            "san": best["san"],
            "piece": best["piece"],
            "from": best["from"],
            "to": best["to"],
        },
        "equivalentMoves": [m["san"] for m in equivalent_moves],
        "topMoves": top_moves_data,
        "playedMove": played_move_info,
        "beforeTopSuggestions": before_top_suggestions,
        "currentPlayer": current_player,
        "evaluationBefore": round(eval_before, 2),
        "evaluationAfter": round(eval_after, 2),
        "evalChange": round(eval_change, 2),
        "formattedEvalBefore": format_eval(eval_before),
        "formattedEvalAfter": format_eval(eval_after),
        "nnPrediction": round(nn_pred, 2),
        "moveReasoning": move_reasons,
        "shapExplanations": shap_explanations,
        "kingSafety": {
            "white": king_white,
            "black": king_black,
        },
        "positionDetails": {
            "whiteKing": king_white["square"],
            "blackKing": king_black["square"],
            "whiteMaterial": f"{features[0]:.0f}p+{features[1]:.0f}N+{features[2]:.0f}B+{features[3]:.0f}R+{features[4]:.0f}Q",
            "blackMaterial": f"{features[5]:.0f}p+{features[6]:.0f}N+{features[7]:.0f}B+{features[8]:.0f}R+{features[9]:.0f}Q",
            "centerControlWhite": int(features[12]),
            "centerControlBlack": int(features[13]),
            "mobilityWhite": int(features[14]),
            "mobilityBlack": int(features[15]),
        },
    }


# ═══════════════════════════════════════════════════════════════════
#  LEGACY TEXT BUILDER
# ═══════════════════════════════════════════════════════════════════

def _build_explanation_text(fen):
    board = chess.Board(fen)
    is_white = board.turn
    current_player = "White" if is_white else "Black"

    equivalent_moves, other_moves, primary_label = get_grouped_top_moves(board, num_moves=3, depth=14)
    if not equivalent_moves:
        return "Error: Could not analyze position."

    best = equivalent_moves[0]
    best_move = best["move"]
    eval_before = best["eval"]

    piece_name = best["piece"]
    from_sq = best["from"]
    to_sq = best["to"]

    move_reasons = analyze_move_reasoning(board, best_move, top_moves_info=equivalent_moves + other_moves)

    board.push(best_move)
    eval_after = get_stockfish_evaluation(board, depth=12)

    def fmt(es):
        if abs(es) > 4:
            return f"{'+' if es > 0 else ''}{es:.2f} (winning)"
        return f"{'+' if es > 0 else ''}{es:.2f}"

    eval_change = eval_after - eval_before

    lines = [f"Best: {piece_name.capitalize()} {from_sq}-{to_sq} ({best['san']})"]
    if len(equivalent_moves) > 1:
        alts = ", ".join(m["san"] for m in equivalent_moves[1:])
        lines.append(f"Also good: {alts}")
    lines.append(f"Eval: {fmt(eval_before)} -> {fmt(eval_after)} ({'+' if eval_change > 0 else ''}{eval_change:.2f})")
    for r in move_reasons:
        lines.append(f"\n  - {r}")
    return "\n".join(lines)
