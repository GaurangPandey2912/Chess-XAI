from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from explain_move_simple import explain_best_move, explain_best_move_dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Chess AI Backend is running"}


@app.get("/explain_move")
def explain_move_api(fen: str = None):
    try:
        if not fen:
            return {"error": "No FEN provided"}
        from urllib.parse import unquote
        fen = unquote(fen)
        if fen == "start":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        explanation = explain_best_move(fen)
        return {"result": explanation}
    except Exception as e:
        return {"error": f"AI Analysis Error: {str(e)}"}


@app.get("/api/analyze")
def analyze_position(fen: str = None, beforeFen: str = None, lastPlayed: str = None):
    try:
        if not fen:
            return {"success": False, "error": "No FEN provided"}
        from urllib.parse import unquote
        fen = unquote(fen)
        if fen == "start":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        if beforeFen:
            beforeFen = unquote(beforeFen)
        data = explain_best_move_dict(fen, beforeFen=beforeFen, lastPlayed=lastPlayed)
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": f"AI Analysis Error: {str(e)}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)