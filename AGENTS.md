# ChessAI - Agent Development Guide

This guide helps agentic coding agents work effectively with this Chess AI project, which consists of a Python backend with FastAPI and a React frontend built with Vite.

## Project Structure

```
ChessAI/
├── backend/                    # Python FastAPI server
│   ├── *.py                   # Core Python modules
│   ├── test_*.py              # Test files
│   └── chess_eval_nn.pt       # Trained model file
├── frontend/chess-ui/          # React frontend
│   ├── src/                   # React source code
│   ├── public/                # Static assets
│   ├── package.json           # Frontend dependencies
│   ├── vite.config.js         # Vite configuration
│   └── eslint.config.js       # ESLint configuration
└── data/                       # Training data files
```

## Build, Test, and Development Commands

### Frontend (React + Vite)
```bash
cd frontend/chess-ui

# Development server (runs on localhost:5173/5174)
npm run dev

# Build for production  
npm run build

# Lint code
npm run lint

# Preview production build
npm run preview
```

### Backend (Python 3.x + FastAPI)
```bash
# Activate conda environment first
conda activate chessxai

cd backend

# Run development server (runs on localhost:8000)
python app.py                 # Main server using Stockfish evaluations
# OR (legacy)
python api.py                 # Legacy server with neural network

# Run individual test files (simple scripts that print results)
# Tests are in backend/ directory - run from project root:
python backend/test_features.py       # Test feature extraction
python backend/test_move_explain.py   # Test move explanation system
python backend/test_explain.py        # Test SHAP explanation system
python backend/test_simple_explain.py # Test Stockfish-only explanation system
python backend/test_nn_explain.py     # Test neural network explanation
python backend/test_black_perspective.py # Test black player perspective

# Train the neural network model (legacy)
python train.py

# Generate training data with Stockfish evaluations
python generate_data.py

# Stockfish evaluation utility
python stockfish_eval.py
```

**IMPORTANT**: The main application now uses `app.py` which provides consistent Stockfish-based evaluations instead of the unpredictable neural network. Use `test_simple_explain.py` for testing.

### Running Full Application
1. Start backend: `conda activate chessxai && cd backend && python app.py` (runs on localhost:8000)
2. Start frontend: `cd frontend/chess-ui && npm run dev` (runs on localhost:5173/5174)
3. CORS is configured for both localhost:5173 and localhost:5174

## Code Style Guidelines

### Python Backend
- **Language**: Python 3.x
- **Framework**: FastAPI for API server
- **ML Libraries**: PyTorch, pandas, numpy
- **Chess Libraries**: chess
- **File Naming**: snake_case for files and functions (e.g., `extract_features`, `explain_move`)
- **Import Style**: 
  - Standard library imports first
  - Third-party imports second  
  - Local imports last
  - Use `import chess` not `from chess import *`
  - Group imports with blank lines between groups
- **Type Hints**: Use type hints for function parameters and return values where beneficial
- **Error Handling**: 
  - Try-catch blocks for API calls and file operations
  - Catch specific exceptions when possible
  - Use descriptive error messages
  - Return appropriate HTTP status codes in FastAPI endpoints
- **Documentation**: Minimal inline comments, function-level docstrings for complex functions
- **Testing**: Simple test files that call functions and print results, not pytest-style assertions

### Frontend (React + Vite)
- **Language**: JavaScript (not TypeScript)
- **Framework**: React 18 with functional components and hooks
- **State Management**: useState for local state
- **Styling**: Inline styles with style objects
- **Import Style**:
  ```javascript
  import { useState } from "react";
  import { Chess } from "chess.js";
  import { Chessboard } from "react-chessboard";
  import axios from "axios";
  ```
- **Component Naming**: PascalCase for components (e.g., `App`, `Chessboard`)
- **Variable Naming**: camelCase for variables and functions
- **Error Handling**: Try-catch with console.error for API failures
- **ESLint Rules**: 
  - No unused variables (except uppercase/underscore patterns)
  - React hooks and refresh plugins enabled
  - Module source type: 'module'

### General Conventions
- **Logging**: Use console.log in frontend for debugging, print statements in backend
- **API Integration**: Frontend calls backend via axios to localhost:8000 endpoints
- **Code Organization**: Keep related functionality in same file, separate concerns across modules
- **No TypeScript**: Project uses JavaScript/Python, avoid adding .ts/.tsx files
- **Minimal Comments**: Code should be self-documenting, avoid excessive inline comments
- **Simple Testing**: Tests are simple script files that print results, not formal test suites

## Key Dependencies

### Backend
- `fastapi` - Web framework
- `torch` - Neural network models
- `chess` - Chess logic and FEN handling
- `pandas` - Data manipulation
- `numpy` - Numerical operations

### Frontend  
- `react` ^18.3.1 - UI framework
- `axios` ^1.13.5 - HTTP client
- `chess.js` ^1.4.0 - Chess logic
- `react-chessboard` ^5.8.6 - Chess board component
- `vite` ^7.3.1 - Build tool

## Development Workflow

1. **Backend Changes**: Modify Python files, test with `python test_*.py`, restart server
2. **Frontend Changes**: Modify React components, Vite hot reloads automatically
3. **Linting**: Run `npm run lint` in frontend before commits
4. **Testing**: Run individual test files, no formal test runner required
5. **API Integration**: Backend runs on port 8000, frontend on 5173/5174, CORS configured

## Common Tasks

- **Add new API endpoint**: Create function in backend, add route to FastAPI app
- **Add new chess feature**: Update features.py with extraction logic
- **Add frontend component**: Create JSX file, import in App.jsx, use inline styles
- **Debug model**: Use explain.py and test_explain.py for SHAP explanations
- **Retrain model**: Run generate_data.py then train.py with new data