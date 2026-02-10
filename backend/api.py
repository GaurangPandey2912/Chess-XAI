from fastapi import FastAPI
from backend.explain_move import explain_best_move

app = FastAPI()

@app.get("/explain_move")
def explain_move_api(fen: str):
    explanation = explain_best_move(fen)
    return {"result": explanation}
