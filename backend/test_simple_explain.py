from explain_move_simple import explain_best_move

# Test with a simple starting position
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
explanation = explain_best_move(fen)
print("Test explanation for starting position:")
print(explanation)

print("\n" + "="*50 + "\n")

# Test with a middle game position
fen = "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 3"
explanation = explain_best_move(fen)
print("Test explanation for middle game position:")
print(explanation)