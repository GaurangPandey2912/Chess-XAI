from explain_move import explain_best_move

# Test with starting position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
explanation = explain_best_move(fen)
print("Test explanation from neural network API:")
print(explanation)