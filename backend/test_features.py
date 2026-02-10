
import chess
from features import extract_features

board = chess.Board()
features = extract_features(board)

print("Feature vector length:", len(features))
print(features)
