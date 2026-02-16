import chess
import chess.engine
from features import extract_features  # Remove 'backend.'
from explain import load_model_and_explainer, explain_position  # Remove 'backend.'
from text_explain import shap_to_text  # Remove 'backend.'

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

def explain_best_move(fen):
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
    
    # Push the move and get evaluation after
    board.push(best_move)
    eval_after = get_stockfish_evaluation(board, depth=12)
    
    # Get features for position analysis (keep for SHAP explanations)
    model, explainer, mean, std = load_model_and_explainer()
    shap_after, pred_after, names = explain_position(board, model, explainer, mean, std)
    
    # Generate explanation for the new position
    text_after = shap_to_text(names, shap_after)
    
    # Format evaluation using accurate Stockfish evaluation
    def format_stockfish_eval(eval_score, is_white_to_move=False):
        if eval_score is None:
            return "0.00 (Equal position)"
        
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
    change_desc = ""
    if abs(eval_change) > 0.3:
        if eval_change > 0:
            change_desc = f"📈 Improvement: +{eval_change:.2f} pawns"
        else:
            change_desc = f"📉 Decline: {eval_change:.2f} pawns"
    else:
        change_desc = "➡️ Position quality maintained"
    
    move_explanation = f"""
🎯 Best Move: {best_move_san} ({current_player})

📊 Stockfish Evaluation: {format_stockfish_eval(eval_after, board.turn)}
{change_desc}

🔍 Position Analysis:
{text_after}

💡 Strategic Insights:
• Evaluation from Stockfish depth 15 analysis
• Positive scores favor White, negative favor Black
• Each pawn = 1.0 points advantage
• Mate scores converted to 100.0 points
"""
    return move_explanation