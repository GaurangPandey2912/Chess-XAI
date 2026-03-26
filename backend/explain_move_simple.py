import chess
import chess.engine
from features import extract_features

ENGINE_PATH = "/Users/gaurangpandey/stockfish/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

SQUARE_NAMES = ["a8","b8","c8","d8","e8","f8","g8","h8",
                "a7","b7","c7","d7","e7","f7","g7","h7",
                "a6","b6","c6","d6","e6","f6","g6","h6",
                "a5","b5","c5","d5","e5","f5","g5","h5",
                "a4","b4","c4","d4","e4","f4","g4","h4",
                "a3","b3","c3","d3","e3","f3","g3","h3",
                "a2","b2","c2","d2","e2","f2","g2","h2",
                "a1","b1","c1","d1","e1","f1","g1","h1"]

PIECE_SYMBOLS = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king", 
                 "P": "pawn", "N": "knight", "B": "bishop", "R": "rook", "Q": "queen", "K": "king"}

CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]
EXTENDED_CENTER = [chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.D4, chess.E4, chess.F4,
                   chess.C5, chess.D5, chess.E5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6]

def square_to_name(sq):
    return SQUARE_NAMES[sq]

def get_stockfish_evaluation(board, depth=15):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=10000)
    if score is None:
        return 0.0
    return score / 100.0

def analyze_position_features(board):
    """Analyze position features for basic insights without SHAP."""
    features = extract_features(board)
    feature_names = [
        "white_pawns","white_knights","white_bishops","white_rooks","white_queens",
        "black_pawns","black_knights","black_bishops","black_rooks","black_queens",
        "white_king_safety","black_king_safety",
        "white_center_control","black_center_control",
        "white_mobility","black_mobility",
        "white_isolated_pawns","black_isolated_pawns",
        "white_passed_pawns","black_passed_pawns",
        "white_doubled_pawns","black_doubled_pawns",
        "white_space","black_space",
        "minor_piece_diff","rook_diff","queen_diff",
        "white_turn"
    ]
    
    insights = []
    
    # Material analysis
    white_material = features[0] + 3*features[1] + 3*features[2] + 5*features[3] + 9*features[4]
    black_material = features[5] + 3*features[6] + 3*features[7] + 5*features[8] + 9*features[9]
    material_diff = white_material - black_material
    
    if material_diff > 3:
        insights.append("White has significant material advantage")
    elif material_diff < -3:
        insights.append("Black has significant material advantage")
    elif abs(material_diff) > 0:
        insights.append(f"{'White' if material_diff > 0 else 'Black'} leads by {abs(material_diff):.1f} points")
    else:
        insights.append("Material is equal")
    
    # Add specific material advantages
    if features[1] > features[6] + 0:
        insights.append(f"White leads {features[1]-features[6]:.0f} knights")
    elif features[6] > features[1] + 0:
        insights.append(f"Black leads {features[6]-features[1]:.0f} knights")
    
    if features[3] > features[8] + 0:
        insights.append(f"White leads {features[3]-features[8]:.0f} rooks")
    elif features[8] > features[3] + 0:
        insights.append(f"Black leads {features[8]-features[3]:.0f} rooks")
    
    if features[2] > features[7] + 0:
        insights.append(f"White leads {features[2]-features[7]:.0f} bishops")
    elif features[7] > features[2] + 0:
        insights.append(f"Black leads {features[7]-features[2]:.0f} bishops")
    
    if features[4] > features[9] + 0:
        insights.append(f"White leads {features[4]-features[9]:.0f} queens")
    elif features[9] > features[4] + 0:
        insights.append(f"Black leads {features[9]-features[4]:.0f} queens")
    
    # King safety
    if features[10] > 80:  # white king safety
        insights.append("White's king position is very safe")
    elif features[11] > 80:  # black king safety
        insights.append("Black's king position is very safe")
    elif features[10] < 50:
        insights.append("White's king safety is compromised")
    elif features[11] < 50:
        insights.append("Black's king safety is compromised")
    
    # Center control
    if features[12] > features[13] + 2:
        insights.append("White dominates the center")
    elif features[13] > features[12] + 2:
        insights.append("Black dominates the center")
    elif abs(features[12] - features[13]) <= 1:
        insights.append("Center control is balanced")
    
    # Mobility
    if features[14] > features[15] + 5:
        insights.append("White has much more active pieces")
    elif features[15] > features[14] + 5:
        insights.append("Black has much more active pieces")
    
    # Pawn structure
    if features[18] > features[19]:  # white passed pawns vs black
        insights.append("White has dangerous passed pawns")
    elif features[19] > features[18]:
        insights.append("Black has dangerous passed pawns")
    
    return insights

def analyze_move_reasoning(board, move):
    """Analyze why a move is good with specific details."""
    reasons = []
    
    is_white = board.turn
    opponent = "Black" if is_white else "White"
    
    from_sq = square_to_name(move.from_square)
    to_sq = square_to_name(move.to_square)
    moving_piece = board.piece_at(move.from_square)
    piece_name = PIECE_SYMBOLS.get(moving_piece.piece_type, "piece") if moving_piece else "piece"
    
    temp_board = board.copy()
    temp_board.push(move)
    
    if temp_board.is_check():
        if temp_board.is_checkmate():
            reasons.append(f"🎯 Checkmate! Forces {opponent} to resign")
        elif temp_board.is_stalemate():
            reasons.append(f"🎯 Stalemate trap - {opponent} has no legal moves")
        else:
            reasons.append(f"🎯 Check on {opponent}'s king at {square_to_name(temp_board.king(not is_white))}")
    
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            captured_name = PIECE_SYMBOLS.get(captured.piece_type, "piece")
            move_captured = f"{piece_name} from {from_sq} captures {captured_name} on {to_sq}"
            
            piece_values = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 100}
            our_val = piece_values.get(piece_name, 0)
            their_val = piece_values.get(captured_name, 0)
            
            if their_val > our_val:
                move_captured += f" (winning trade: +{their_val - our_val:.0f})"
            elif their_val == our_val:
                move_captured += " (equal trade)"
            else:
                move_captured += f" (losing trade: {their_val - our_val:.0f})"
            
            reasons.append(f"⚔️ {move_captured}")
    
    if move.promotion:
        reasons.append(f"👑 Pawn promotes to {PIECE_SYMBOLS.get(move.promotion, 'queen')} on {to_sq}")
    
    if board.is_en_passant(move):
        ep_square = square_to_name(move.to_square + (-8 if is_white else 8))
        reasons.append(f"⚔️ En passant: captures pawn on {ep_square}")
    
    attacking_squares = []
    for sq in [s for s in chess.SQUARES if temp_board.is_attacked_by(is_white, s)]:
        if temp_board.piece_at(sq) and temp_board.piece_at(sq).color == is_white:
            sq_name = square_to_name(sq)
            attacked = temp_board.attackers(not is_white, sq)
            for attacker in attacked:
                if temp_board.piece_at(attacker):
                    attacker_piece = temp_board.piece_at(attacker)
                    if attacker_piece.color == is_white:
                        attacking_squares.append((sq_name, PIECE_SYMBOLS.get(attacker_piece.piece_type, "piece")))
    
    if len(attacking_squares) > 1:
        squares = [s[0] for s in attacking_squares[:3]]
        pieces = [s[1] for s in attacking_squares[:3]]
        reasons.append(f"⚠️ Double attack on {', '.join(squares)} - opponent must respond to one threat")
    
    features = extract_features(temp_board)
    if is_white:
        opp_king_safety = features[11]
        player_king_safety = features[10]
    else:
        opp_king_safety = features[10]
        player_king_safety = features[11]
    
    opp_king_sq = square_to_name(temp_board.king(not is_white))
    if opp_king_safety < 50:
        reasons.append(f"👑 {opponent}'s king at {opp_king_sq} is exposed and vulnerable (safety: {opp_king_safety:.0f})")
    elif opp_king_safety < 70:
        reasons.append(f"🛡️ {opponent}'s king at {opp_king_sq} has developing weaknesses")
    
    controlled_squares = []
    for sq in CENTER_SQUARES + EXTENDED_CENTER:
        if temp_board.is_attacked_by(is_white, sq):
            controlled_squares.append(square_to_name(sq))
    
    if len(controlled_squares) >= 4:
        reasons.append(f"🎯 Strong center control: occupies {', '.join(controlled_squares[:4])}")
    elif controlled_squares:
        reasons.append(f"🎯 Controls {len(controlled_squares)} central squares: {', '.join(controlled_squares[:3])}")
    
    if is_white:
        player_center = features[12]
        opp_center = features[13]
        player_passed = features[18]
        opp_passed = features[19]
        player_mobility = features[14]
        opp_mobility = features[15]
    else:
        player_center = features[13]
        opp_center = features[12]
        player_passed = features[19]
        opp_passed = features[18]
        player_mobility = features[15]
        opp_mobility = features[14]
    
    if player_passed > opp_passed + 0.5:
        reasons.append(f"♟️ Advanced passed pawn threatens promotion on {['h','g','f','e','d','c','b','a'][7 if is_white else 0]}-file")
    elif opp_passed > player_passed + 0.5:
        reasons.append(f"⚠️ Opponent's passed pawn on {['h','g','f','e','d','c','b','a'][7 if not is_white else 0]}-file requires immediate attention")
    
    if player_mobility > opp_mobility + 5:
        reasons.append(f"🐴 Superior piece activity: {player_mobility:.0f} vs {opp_mobility:.0f} legal moves")
    elif opp_mobility > player_mobility + 5:
        reasons.append(f"⚠️ Opponent has more active pieces: {opp_mobility:.0f} vs {player_mobility:.0f} moves")
    
    legal_moves = list(temp_board.legal_moves)
    capture_moves = [m for m in legal_moves if temp_board.is_capture(m)]
    if len(capture_moves) >= 3 and len(legal_moves) > 0:
        capture_ratio = len(capture_moves) / len(legal_moves) * 100
        reasons.append(f"💥 Tactic opportunity: {len(capture_moves)} of {len(legal_moves)} available moves are captures")
    
    if temp_board.has_kingside_castling_rights(is_white):
        reasons.append(f"🏰 Kingside castling still available")
    if temp_board.has_queenside_castling_rights(is_white):
        reasons.append(f"🏰 Queenside castling still available")
    
    return reasons

def explain_best_move(fen):
    """Provide detailed chess move explanations with specific square names and tactical patterns."""
    board = chess.Board(fen)
    
    current_player = "White" if board.turn else "Black"
    opponent = "Black" if board.turn else "White"
    is_white = board.turn
    
    eval_before = get_stockfish_evaluation(board, depth=12)
    
    result = engine.play(board, chess.engine.Limit(depth=15))
    best_move = result.move
    best_move_san = board.san(best_move)
    from_sq = square_to_name(best_move.from_square)
    to_sq = square_to_name(best_move.to_square)
    moving_piece = board.piece_at(best_move.from_square)
    piece_name = PIECE_SYMBOLS.get(moving_piece.piece_type, "piece") if moving_piece else "piece"
    
    move_reasons = analyze_move_reasoning(board, best_move)
    
    board.push(best_move)
    eval_after = get_stockfish_evaluation(board, depth=12)
    
    features = extract_features(board)
    white_king_sq = square_to_name(board.king(True))
    black_king_sq = square_to_name(board.king(False))
    
    def format_eval(eval_score):
        if abs(eval_score) > 4:
            strength = "winning advantage"
        elif abs(eval_score) > 2:
            strength = "strong advantage"
        elif abs(eval_score) > 1:
            strength = "moderate advantage"
        elif abs(eval_score) > 0.5:
            strength = "slight advantage"
        else:
            strength = "roughly equal"
        
        if eval_score > 0:
            return f"+{eval_score:.2f} ({strength})"
        elif eval_score < 0:
            return f"{eval_score:.2f} ({strength})"
        return f"{eval_score:.2f} (equal)"
    
    eval_change = eval_after - eval_before
    
    move_explanation = f"""🎯 Best Move: {piece_name.capitalize()} {from_sq}-{to_sq} ({best_move_san})

📊 Position Evaluation:
   Before: {format_eval(eval_before)}
   After:  {format_eval(eval_after)}
   Change: {'+' if eval_change > 0 else ''}{eval_change:.2f} pawns

⭐ Why {best_move_san}?
"""
    
    if move_reasons:
        for reason in move_reasons:
            move_explanation += f"   {reason}\n"
    else:
        move_explanation += f"   • Develops {piece_name} to active square {to_sq}\n"
        move_explanation += f"   • Maintains tactical flexibility\n"
    
    move_explanation += f"""
🔍 Position Details:
   White King: {white_king_sq}
   Black King: {black_king_sq}
   White Material: {features[0]:.0f}p + {features[1]:.0f}N + {features[2]:.0f}B + {features[3]:.0f}R + {features[4]:.0f}Q
   Black Material: {features[5]:.0f}p + {features[6]:.0f}N + {features[7]:.0f}B + {features[8]:.0f}R + {features[9]:.0f}Q
   Center Control: White {features[12]:.0f} vs Black {features[13]:.0f}
   Piece Activity: White {features[14]:.0f} vs Black {features[15]:.0f} legal moves
   
💡 Key Insights:"""
    
    if features[12] > features[13] + 2:
        move_explanation += f"\n   • White controls key central squares (d4, e4, d5, e5)"
    elif features[13] > features[12] + 2:
        move_explanation += f"\n   • Black dominates the center"
    
    white_has_passed = features[18] > 0
    black_has_passed = features[19] > 0
    if white_has_passed and not black_has_passed:
        move_explanation += f"\n   • White has passed pawn(s) that can advance toward promotion"
    elif black_has_passed and not white_has_passed:
        move_explanation += f"\n   • Black has dangerous passed pawn threatening promotion"
    
    if is_white:
        if features[10] < 50:
            move_explanation += f"\n   • Your king at {white_king_sq} is exposed - consider castling or evacuating"
        if features[11] < 50:
            move_explanation += f"\n   • {opponent}'s king at {black_king_sq} is vulnerable to attack"
    else:
        if features[11] < 50:
            move_explanation += f"\n   • Your king at {black_king_sq} is exposed - consider castling or evacuating"
        if features[10] < 50:
            move_explanation += f"\n   • {opponent}'s king at {white_king_sq} is vulnerable to attack"
    
    move_explanation += f"""
   
💡 Technical: Stockfish depth 15 | 1 pawn = 1.0 evaluation"""
    
    return move_explanation