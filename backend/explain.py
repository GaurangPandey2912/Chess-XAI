import torch
import shap
import numpy as np
from model import ChessEvalNet  # Remove 'backend.'
from features import extract_features  


MODEL_PATH = "chess_eval_nn.pt"

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
    # Load training statistics for normalization
    import pandas as pd
    df = pd.read_csv("../data/positions.csv")
    X = df.iloc[:, :-1].values.astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    
    model = ChessEvalNet()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # background must match feature size (28) and be normalized
    background = np.zeros((1, len(FEATURE_NAMES)))
    background = (background - mean) / std
    explainer = shap.DeepExplainer(model, torch.tensor(background).float())

    return model, explainer, mean, std


def explain_position(board, model, explainer, mean, std):
    features = extract_features(board)
    # Apply same normalization as training
    features_norm = (np.array(features) - mean) / std
    x = torch.tensor(features_norm).float().unsqueeze(0)

    shap_values = explainer.shap_values(x)
    prediction = model(x).item()

    return shap_values, prediction, FEATURE_NAMES
