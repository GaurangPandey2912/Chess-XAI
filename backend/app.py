from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from explain_move_simple import explain_best_move

app = FastAPI()

# Add CORS middleware with port 5174
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5174", 
        "http://localhost:5173", 
        "http://localhost:3000", 
        "http://127.0.0.1:5174",
        "http://127.0.0.1:5173", 
        "http://127.0.0.1:3000"
    ],
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
        
        # URL decode the FEN string
        from urllib.parse import unquote
        fen = unquote(fen)
        
        # Handle chess.js 'start' FEN
        if fen == "start":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        print(f"Debug: FEN received: '{fen}'")
        print(f"Debug: FEN length: {len(fen)}")
        
        explanation = explain_best_move(fen)
        return {"result": explanation}
    except Exception as e:
        return {"error": f"AI Analysis Error: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)