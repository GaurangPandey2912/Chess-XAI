import chess
import chess.engine
import random
import csv
from features import extract_features

STOCKFISH_PATH = "/Users/gaurangpandey/stockfish/stockfish"  # adjust if needed
ENGINE_DEPTH = 12
NUM_SAMPLES = 5000   # start small, we can increase later

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

def random_position(max_moves=20):
    board = chess.Board()
    moves = random.randint(4, max_moves)
    for _ in range(moves):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

with open("data/positions.csv", "w", newline="") as f:
    writer = csv.writer(f)

    # Header
    writer.writerow([f"f{i}" for i in range(28)] + ["eval"])

    for i in range(NUM_SAMPLES):
        board = random_position()
        features = extract_features(board)

        info = engine.analyse(board, chess.engine.Limit(depth=ENGINE_DEPTH))
        score = info["score"].white().score(mate_score=10000)

        writer.writerow(list(features) + [score])

        if i % 500 == 0:
            print(f"Generated {i} positions")

engine.quit()
print("Dataset generation complete.")
