import { useState, useCallback, useRef } from "react";
import { Chess } from "chess.js";

export default function useChessGame() {
  const [game, setGame] = useState(new Chess());
  const [fen, setFen] = useState(game.fen());
  const [history, setHistory] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [orientation, setOrientation] = useState("white");
  const [gameOver, setGameOver] = useState(false);
  const [showBestMove, setShowBestMove] = useState(true);
  const gameRef = useRef(game);

  const syncGame = useCallback((updatedGame) => {
    gameRef.current = updatedGame;
    setGame(updatedGame);
    setFen(updatedGame.fen());
    setHistory(updatedGame.history({ verbose: true }));
    setGameOver(
      updatedGame.isGameOver() ||
      updatedGame.isCheckmate() ||
      updatedGame.isDraw() ||
      updatedGame.isStalemate() ||
      updatedGame.isThreefoldRepetition()
    );
  }, []);

  const makeMove = useCallback((from, to, promotion = "q") => {
    const updated = new Chess();
    const prevMoves = gameRef.current.history();
    try {
      for (const san of prevMoves) {
        const result = updated.move(san);
        if (!result) throw new Error(`Cannot replay ${san}`);
      }
      const move = updated.move({ from, to, promotion });
      if (!move) return null;
      syncGame(updated);
      return move;
    } catch {
      return null;
    }
  }, [syncGame]);

  const undoMove = useCallback(() => {
    const prevMoves = gameRef.current.history();
    if (prevMoves.length === 0) return null;
    try {
      const updated = new Chess();
      for (let i = 0; i < prevMoves.length - 1; i++) {
        const result = updated.move(prevMoves[i]);
        if (!result) throw new Error(`Cannot replay ${prevMoves[i]}`);
      }
      syncGame(updated);
      setAnalysis(null);
      setError(null);
      return prevMoves[prevMoves.length - 1];
    } catch {
      return null;
    }
  }, [syncGame]);

  const resetGame = useCallback(() => {
    const fresh = new Chess();
    syncGame(fresh);
    setAnalysis(null);
    setError(null);
  }, [syncGame]);

  const flipBoard = useCallback(() => {
    setOrientation((o) => (o === "white" ? "black" : "white"));
  }, []);

  const analyzePosition = useCallback(async () => {
    setIsAnalyzing(true);
    setError(null);
    try {
      const currentFen = gameRef.current.fen();
      const moves = gameRef.current.history({ verbose: true });
      let beforeFen = currentFen;
      let lastPlayed = "";
      if (moves.length > 0) {
        const last = moves[moves.length - 1];
        beforeFen = last.before;
        lastPlayed = last.san;
      }
      const params = new URLSearchParams({ fen: currentFen, beforeFen, lastPlayed });
      const res = await fetch(`/api/analyze?${params}`);
      const json = await res.json();
      if (json.success && json.data) {
        setAnalysis(json.data);
      } else {
        setError(json.error || "Analysis failed");
      }
    } catch (e) {
      setError("Cannot connect to backend. Make sure the server is running.");
    }
    setIsAnalyzing(false);
  }, []);

  const loadFen = useCallback((fenStr) => {
    try {
      const updated = new Chess(fenStr);
      syncGame(updated);
      setAnalysis(null);
      setError(null);
    } catch {
      setError("Invalid FEN string");
    }
  }, [syncGame]);

  const toggleBestMove = useCallback(() => {
    setShowBestMove((prev) => !prev);
  }, []);

  return {
    game,
    fen,
    history,
    analysis,
    isAnalyzing,
    error,
    orientation,
    gameOver,
    showBestMove,
    makeMove,
    undoMove,
    resetGame,
    flipBoard,
    analyzePosition,
    loadFen,
    toggleBestMove,
  };
}
