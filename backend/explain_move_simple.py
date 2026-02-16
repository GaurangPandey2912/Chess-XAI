import chess
import chess.engine
from features import extract_features

ENGINE_PATH = "/Users/gaurangpandey/stockfish/stockfish"
engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

def get_stockfish_evaluation(board, depth=15):
    """Get accurate Stockfish evaluation for the current position."""
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=10000)
    # Convert to centipawns and normalize
    if score is None:
        return 0.0
    return score / 100.0  # Convert to pawns

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
    """Analyze why a move is good - checks, tactics, threats against opponent."""
    reasons = []
    
    is_white = board.turn
    opponent = "Black" if is_white else "White"
    player = "White" if is_white else "Black"
    
    temp_board = board.copy()
    temp_board.push(move)
    
    # Check if move gives check
    if temp_board.is_check():
        reasons.append(f"🎯 Gives {opponent} king a check!")
    
    # Check if move captures something
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            piece_name = {1: "pawn", 2: "knight", 3: "bishop", 4: "rook", 5: "queen", 6: "king"}
            reasons.append(f"⚔️ Captures opponent's {piece_name.get(captured.piece_type, 'piece')}")
    
    # Check opponent's king safety in resulting position
    features = extract_features(temp_board)
    if is_white:
        opp_king_safety = features[11]  # black_king_safety
        player_king_safety = features[10]  # white_king_safety
    else:
        opp_king_safety = features[10]  # white_king_safety
        player_king_safety = features[11]  # black_king_safety
    
    if opp_king_safety < 50:
        reasons.append(f"👑 {opponent}'s king is vulnerable - safety score: {opp_king_safety:.0f}")
    elif opp_king_safety < 70:
        reasons.append(f"🛡️ {opponent}'s king has some weaknesses")
    
    # Check for passed pawns
    if is_white:
        player_passed = features[18]
        opp_passed = features[19]
    else:
        player_passed = features[19]
        opp_passed = features[18]
    
    if player_passed > opp_passed:
        reasons.append(f"♟️ You have passed pawns advancing")
    elif opp_passed > player_passed:
        reasons.append(f"⚠️ Opponent has dangerous passed pawns to defend")
    
    # Center control
    if is_white:
        player_center = features[12]
        opp_center = features[13]
    else:
        player_center = features[13]
        opp_center = features[12]
    
    if player_center > opp_center + 2:
        reasons.append(f"🎯 You dominate the center")
    elif opp_center > player_center + 2:
        reasons.append(f"⚠️ {opponent} controls the center")
    
    # Mobility advantage
    if is_white:
        player_mobility = features[14]
        opp_mobility = features[15]
    else:
        player_mobility = features[15]
        opp_mobility = features[14]
    
    if player_mobility > opp_mobility + 5:
        reasons.append(f"🐴 You have significantly more piece activity")
    elif opp_mobility > player_mobility + 5:
        reasons.append(f"⚠️ {opponent} has more active pieces")
    
    # Material
    if is_white:
        player_mat = features[0] + 3*features[1] + 3*features[2] + 5*features[3] + 9*features[4]
        opp_mat = features[5] + 3*features[6] + 3*features[7] + 5*features[8] + 9*features[9]
    else:
        player_mat = features[5] + 3*features[6] + 3*features[7] + 5*features[8] + 9*features[9]
        opp_mat = features[0] + 3*features[1] + 3*features[2] + 5*features[3] + 9*features[4]
    
    if player_mat > opp_mat + 3:
        reasons.append(f"💰 You have a material advantage")
    elif opp_mat > player_mat + 3:
        reasons.append(f"⚠️ {opponent} has material advantage")
    
    return reasons

def explain_best_move(fen):
    """Provide chess move explanations using Stockfish evaluation and feature analysis."""
    board = chess.Board(fen)
    
    # Determine who is playing based on turn BEFORE move
    current_player = "White" if board.turn else "Black"
    opponent = "Black" if board.turn else "White"
    
    # Get evaluation before the move
    eval_before = get_stockfish_evaluation(board, depth=12)
    
    # Get best move
    result = engine.play(board, chess.engine.Limit(depth=15))
    best_move = result.move
    best_move_san = board.san(best_move)
    
    # Analyze why the move is good (before pushing)
    move_reasons = analyze_move_reasoning(board, best_move)
    
    # Push the move and get evaluation after
    board.push(best_move)
    eval_after = get_stockfish_evaluation(board, depth=12)
    
    # Analyze the position after the move
    position_insights = analyze_position_features(board)
    
    # Format evaluation using accurate Stockfish evaluation
    def format_stockfish_eval(eval_score):
        if eval_score > 4:
            return f"+{eval_score:.2f} (White has winning advantage)"
        elif eval_score > 2:
            return f"+{eval_score:.2f} (White has strong advantage)"
        elif eval_score > 1:
            return f"+{eval_score:.2f} (White has moderate advantage)"
        elif eval_score > 0.5:
            return f"+{eval_score:.2f} (White has slight advantage)"
        elif eval_score > -0.5:
            return f"{eval_score:.2f} (Position is roughly equal)"
        elif eval_score > -1:
            return f"{eval_score:.2f} (Black has slight advantage)"
        elif eval_score > -2:
            return f"{eval_score:.2f} (Black has moderate advantage)"
        elif eval_score > -4:
            return f"{eval_score:.2f} (Black has strong advantage)"
        else:
            return f"{eval_score:.2f} (Black has winning advantage)"
    
    # Calculate evaluation change
    eval_change = eval_after - eval_before
    if abs(eval_change) > 0.3:
        if eval_change > 0:
            change_desc = f"📈 Move improves evaluation by +{eval_change:.2f} pawns"
        else:
            change_desc = f"📉 Move worsens evaluation by {eval_change:.2f} pawns"
    else:
        change_desc = "➡️ Move maintains positional balance"
    
    move_explanation = f"""🎯 Best Move: {best_move_san} ({current_player})

📊 Current Position: {format_stockfish_eval(eval_before)}
📊 After Best Move: {format_stockfish_eval(eval_after)}
{change_desc}

⭐ Why this move is good:
"""
    
    if move_reasons:
        move_explanation += "\n".join(f"• {reason}" for reason in move_reasons)
    else:
        move_explanation += "• Solid positional move"
    
    move_explanation += f"""

🔍 Position Analysis:
""" + "\n".join(f"• {insight}" for insight in position_insights[:5])

    move_explanation += f"""

💡 Analysis Details:
• Current Position: Evaluation before best move is played
• After Best Move: Evaluation after the best move is played
• Evaluation based on Stockfish depth 15
• Each pawn = 1.0 points advantage
• Mate scores converted to 100.0 points
• Positive scores favor White, negative favor Black"""
    
    return move_explanation