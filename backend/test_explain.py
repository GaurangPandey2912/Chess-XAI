import chess
from explain import explain_position
from load_background import load_background
from text_explain import shap_to_text

board = chess.Board()
background = load_background(100)

pred, shap_values, feature_names = explain_position(board, background)
explanation = shap_to_text(feature_names, shap_values)
print("\nExplanation:")
print(explanation)

print("NN evaluation:", pred)
print("Top contributing features:")

vals = shap_values.values[0]
top = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:8]

for f, v in top:
    print(f"{f}: {v:+.3f}")
