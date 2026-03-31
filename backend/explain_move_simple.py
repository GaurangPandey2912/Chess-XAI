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


def _interpret_shap_features(shap_vals, features, feature_names, top_k=8):
    pairs = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)
    
    explanations = []
    
    material_idx = {
        "white_pawns": 0, "white_knights": 1, "white_bishops": 2, "white_rooks": 3, "white_queens": 4,
        "black_pawns": 5, "black_knights": 6, "black_bishops": 7, "black_rooks": 8, "black_queens": 9
    }
    
    white_material = sum(features[i] * [1, 3, 3, 5, 9][i] for i in range(5))
    black_material = sum(features[i] * [1, 3, 3, 5, 9][i - 5 if i >= 5 else 0] for i in range(5, 10))
    white_material = sum(features[i] * v for i, v in enumerate([1, 3, 3, 5, 9]))
    black_material = sum(features[5 + i] * v for i, v in enumerate([1, 3, 3, 5, 9]))
    material_diff = white_material - black_material
    
    for name, val in pairs[:top_k]:
        if abs(val) < 0.1:
            continue
            
        if name == "white_king_safety":
            if val > 0.2:
                explanations.append(f"  - White's king position is safe (safety: {features[10]:.0f})")
            elif val < -0.2:
                explanations.append(f"  - White's king safety is compromised (safety: {features[10]:.0f})")
                
        elif name == "black_king_safety":
            if val > 0.2:
                explanations.append(f"  - Black's king position is safe (safety: {features[11]:.0f})")
            elif val < -0.2:
                explanations.append(f"  - Black's king safety is compromised (safety: {features[11]:.0f})")
                
        elif name == "white_center_control":
            if val > 0.2:
                explanations.append(f"  - White controls the center ({features[12]:.0f} squares)")
            elif val < -0.2:
                explanations.append(f"  - Black dominates center control ({features[13]:.0f} squares)")
                
        elif name == "black_center_control":
            if val > 0.2:
                explanations.append(f"  - Black controls the center ({features[13]:.0f} squares)")
            elif val < -0.2:
                explanations.append(f"  - White dominates center control ({features[12]:.0f} squares)")
                
        elif name == "white_mobility":
            if val > 0.2:
                explanations.append(f"  - White has active pieces ({features[14]:.0f} legal moves)")
            elif val < -0.2:
                explanations.append(f"  - Black has more active pieces ({features[15]:.0f} moves)")
                
        elif name == "black_mobility":
            if val > 0.2:
                explanations.append(f"  - Black has active pieces ({features[15]:.0f} legal moves)")
            elif val < -0.2:
                explanations.append(f"  - White has more active pieces ({features[14]:.0f} moves)")
                
        elif name == "white_passed_pawns":
            if val > 0.15 and features[18] > 0:
                explanations.append(f"  - White has passed pawns threatening promotion")
                
        elif name == "black_passed_pawns":
            if val > 0.15 and features[19] > 0:
                explanations.append(f"  - Black has passed pawns threatening promotion")
                
        elif name == "white_space":
            if val > 0.15:
                explanations.append(f"  - White controls more board space ({features[24]:.0f} squares)")
            elif val < -0.15:
                explanations.append(f"  - Black controls more board space ({features[25]:.0f} squares)")
                
        elif name == "minor_piece_diff":
            if val > 0.2:
                explanations.append(f"  - White has better minor piece coordination")
            elif val < -0.2:
                explanations.append(f"  - Black has better minor piece coordination")
                
        elif name == "rook_diff":
            if val > 0.2:
                explanations.append(f"  - White has rook activity advantage")
            elif val < -0.2:
                explanations.append(f"  - Black has rook activity advantage")
                
        elif name == "queen_diff":
            if val > 0.2:
                explanations.append(f"  - White's queen is better positioned")
            elif val < -0.2:
                explanations.append(f"  - Black's queen is better positioned")
                
        elif name in material_idx:
            idx = material_idx[name]
            if "white" in name:
                piece = name.replace("white_", "")
                if val > 0.3 and features[idx] > 0:
                    explanations.append(f"  - White has {features[idx]:.0f} {piece} on the board")
            elif "black" in name:
                piece = name.replace("black_", "")
                if val > 0.3 and features[idx] > 0:
                    explanations.append(f"  - Black has {features[idx]:.0f} {piece} on the board")
    
    if material_diff > 2:
        explanations.insert(0, f"  - Material: White leads by {material_diff:.1f} points")
    elif material_diff < -2:
        explanations.insert(0, f"  - Material: Black leads by {-material_diff:.1f} points")
    elif abs(material_diff) < 0.5:
        explanations.insert(0, f"  - Material is equal")
    
    return explanations[:top_k]


def square_to_name(sq):
    return SQUARE_NAMES[sq]


def get_stockfish_evaluation(board, depth=15):
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white().score(mate_score=10000)
    if score is None:
        return 0.0
    return score / 100.0


def analyze_move_reasoning(board, move):
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
            reasons.append(f"Checkmate! Forces {opponent} to resign")
        elif temp_board.is_stalemate():
            reasons.append(f"Stalemate trap - {opponent} has no legal moves")
        else:
            reasons.append(f"Check on {opponent}'s king at {square_to_name(temp_board.king(not is_white))}")
    
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured:
            captured_name = PIECE_SYMBOLS.get(captured.piece_type, "piece")
            piece_values = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 100}
            our_val = piece_values.get(piece_name, 0)
            their_val = piece_values.get(captured_name, 0)
            
            trade_desc = f"{piece_name} from {from_sq} captures {captured_name} on {to_sq}"
            if their_val > our_val:
                trade_desc += f" (winning trade: +{their_val - our_val:.0f})"
            elif their_val == our_val:
                trade_desc += " (equal trade)"
            else:
                trade_desc += f" (losing trade: {their_val - our_val:.0f})"
            
            reasons.append(trade_desc)
    
    if move.promotion:
        reasons.append(f"Pawn promotes to {PIECE_SYMBOLS.get(move.promotion, 'queen')} on {to_sq}")
    
    if board.is_en_passant(move):
        ep_square = square_to_name(move.to_square + (-8 if is_white else 8))
        reasons.append(f"En passant: captures pawn on {ep_square}")
    
    controlled_squares = []
    for sq in CENTER_SQUARES + EXTENDED_CENTER:
        if temp_board.is_attacked_by(is_white, sq):
            controlled_squares.append(square_to_name(sq))
    
    if len(controlled_squares) >= 4:
        reasons.append(f"Strong center control: occupies {', '.join(controlled_squares[:4])}")
    
    if temp_board.has_kingside_castling_rights(is_white):
        reasons.append("Kingside castling still available")
    if temp_board.has_queenside_castling_rights(is_white):
        reasons.append("Queenside castling still available")
    
    return reasons


def explain_best_move(fen):
    """Provide SHAP-based explainable AI chess move explanations."""
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
    
    shap_vals, nn_pred, features = _get_shap_explanation(board)
    if shap_vals.ndim > 1:
        shap_vals = shap_vals.flatten()
    shap_explanations = _interpret_shap_features(shap_vals, features, FEATURE_NAMES)
    
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
    
    move_explanation = f"""Best Move: {piece_name.capitalize()} {from_sq}-{to_sq} ({best_move_san})

Position Evaluation (Stockfish depth 15):
   Before: {format_eval(eval_before)}
   After:  {format_eval(eval_after)}
   Change: {'+' if eval_change > 0 else ''}{eval_change:.2f} pawns

Neural Network Prediction: {nn_pred:+.2f} (based on learned position patterns)

Why {best_move_san}?
"""
    
    if move_reasons:
        for reason in move_reasons:
            move_explanation += f"   - {reason}\n"
    else:
        move_explanation += f"   - Develops {piece_name} to active square {to_sq}\n"
        move_explanation += f"   - Maintains tactical flexibility\n"
    
    move_explanation += f"""
SHAP Feature Attribution Analysis:
   The neural network's evaluation is explained by:
"""
    
    for exp in shap_explanations:
        move_explanation += f"   {exp}\n"
    
    move_explanation += f"""
Position Details:
   White King: {white_king_sq}
   Black King: {black_king_sq}
   Material: White {features[0]:.0f}p+{features[1]:.0f}N+{features[2]:.0f}B+{features[3]:.0f}R+{features[4]:.0f}Q
              Black {features[5]:.0f}p+{features[6]:.0f}N+{features[7]:.0f}B+{features[8]:.0f}R+{features[9]:.0f}Q
   Center Control: White {features[12]:.0f} vs Black {features[13]:.0f}
   Piece Activity: White {features[14]:.0f} vs Black {features[15]:.0f} legal moves

Technical Details:
   - Stockfish evaluation: depth 15 search
   - SHAP DeepExplainer: neural network feature attribution
   - 1 pawn = 1.0 evaluation points
   - Positive SHAP values favor White, negative favor Black
"""
    
    return move_explanation
