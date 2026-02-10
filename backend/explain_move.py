import chess
import chess.engine
from backend.features import extract_features
from backend.explain import load_model_and_explainer, explain_position
from backend.text_explain import shap_to_text


ENGINE_PATH = "/Users/gaurangpandey/stockfish/stockfish"

engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)

def explain_best_move(fen):
    board = chess.Board(fen)

    model, explainer = load_model_and_explainer()

    # Position BEFORE
    shap_before, pred_before, names = explain_position(board, model, explainer)

    # Get best move
    result = engine.play(board, chess.engine.Limit(depth=15))
    best_move = result.move

    # Store SAN BEFORE pushing
    best_move_san = board.san(best_move)

    # Position AFTER
    board.push(best_move)
    shap_after, pred_after, _ = explain_position(board, model, explainer)

    # Generate explanations
    text_before = shap_to_text(names, shap_before)
    text_after = shap_to_text(names, shap_after)

    move_explanation = f"""
Best Move: {best_move_san}

Evaluation before: {pred_before:.2f}
Evaluation after: {pred_after:.2f}

Before explanation:
{text_before}

After explanation:
{text_after}
"""

    return move_explanation
