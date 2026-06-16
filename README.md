# Chess-XAI

An explainable AI chess analysis tool that combines Stockfish engine evaluation with a neural network and SHAP feature attribution to provide natural-language explanations of chess positions and moves.

## Architecture

```
ChessAI/
├── backend/          # Python backend (FastAPI + ML)
│   ├── app.py        # Main FastAPI server (port 8000)
│   ├── model.py      # 3-layer MLP neural network (PyTorch)
│   ├── features.py   # 28 hand-crafted chess feature extractors
│   ├── explain_move_simple.py  # Full analysis pipeline
│   ├── generate_data.py  # Training data generator
│   ├── train.py      # Model training script
│   └── text_explain.py  # SHAP values → human-readable text
├── frontend/chess-ui/ # React + Vite SPA
│   └── src/
│       ├── App.jsx   # Main chess UI with board + analysis panel
│       └── main.jsx  # React entry point
└── data/
    └── positions.csv # 5000 training positions
```

## Prerequisites

- **Backend:** Python 3.10+ with [Stockfish](https://stockfishchess.org/) installed
- **Frontend:** Node.js 18+
- **Stockfish path:** Default is `/usr/local/bin/stockfish`; update in `backend/explain_move_simple.py`

## Setup & Run

### Backend

```bash
cd backend
pip install fastapi uvicorn torch shap python-chess numpy scikit-learn
python app.py
```

The API starts at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend/chess-ui
npm install
npm run dev
```

The UI starts at `http://127.0.0.1:5173`.

## API

### `POST /explain_move`

Analyzes a chess position and returns an AI-generated explanation.

**Request body:**
```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
}
```

**Response:** JSON with move reasoning, SHAP feature attribution, material evaluation, and strategic insights.

## ML Pipeline

1. **Feature extraction:** 28 chess features (material, king safety, center control, mobility, pawn structure, etc.)
2. **Neural network:** 3-layer MLP (28 → 64 → 32 → 1) trained on Stockfish evaluations
3. **Explainability:** SHAP DeepExplainer attributes each feature's contribution to the evaluation
4. **Text generation:** Converts SHAP values into natural-language strategic commentary

## Model Training

```bash
cd backend
python generate_data.py   # Generate 5000 training positions
python train.py            # Train the neural network
```
