import torch
import shap
import numpy as np
from backend.model import ChessEvalNet
from backend.features import extract_features


MODEL_PATH = "backend/chess_eval_nn.pt"

FEATURE_NAMES = [
    "white_pawns","white_knights","white_bishops","white_rooks","white_queens",
    "black_pawns","black_knights","black_bishops","black_rooks","black_queens",

    "white_king_safety","black_king_safety",

    "white_center_control","black_center_control",

    "white_mobility","black_mobility",

    "white_isolated_pawns","black_isolated_pawns",
    "white_passed_pawns","black_passed_pawns",
    "white_doubled_pawns","black_doubled_pawns",

    "white_space","black_space",

    "minor_piece_diff",
    "rook_diff",
    "queen_diff",

    "white_turn"
]

def load_model_and_explainer():
    model = ChessEvalNet()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # background must match feature size (28)
    background = np.zeros((1, len(FEATURE_NAMES)))
    explainer = shap.DeepExplainer(model, torch.tensor(background).float())

    return model, explainer


def explain_position(board, model, explainer):
    features = extract_features(board)
    x = torch.tensor(features).float().unsqueeze(0)

    shap_values = explainer.shap_values(x)
    prediction = model(x).item()

    return shap_values, prediction, FEATURE_NAMES
