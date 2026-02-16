from explain_move_simple import explain_best_move

# Test with a Black to move position
fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2"
explanation = explain_best_move(fen)
print("Test explanation for Black to move:")
print(explanation)