import useChessGame from "./hooks/useChessGame";
import ChessBoardSection from "./components/ChessBoard";
import MoveHistory from "./components/MoveHistory";
import AnalysisPanel from "./components/AnalysisPanel";
import "./App.css";

export default function App() {
  const {
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
    toggleBestMove,
  } = useChessGame();

  const onPieceDrop = (source, target) => {
    if (gameOver) return false;
    const move = makeMove(source, target);
    if (!move) return false;
    analyzePosition();
    return true;
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a10 10 0 1 0 10 10h-10V2z" />
              <path d="M12 12 8 8" />
              <path d="M16 16 12 12" />
            </svg>
          </div>
          <h1>Chess AI Coach</h1>
        </div>
        <div className="header-right">
          <span className="status-dot" />
          <span className="status-text">AI-Powered Analysis</span>
        </div>
      </header>

      <main className="app-main">
        <div className="board-column">
          <ChessBoardSection
            fen={fen}
            orientation={orientation}
            onPieceDrop={onPieceDrop}
            analysis={analysis}
            isAnalyzing={isAnalyzing}
            error={error}
            gameOver={gameOver}
            onNewGame={resetGame}
            onUndo={undoMove}
            onFlip={flipBoard}
            onAnalyze={analyzePosition}
            history={history}
            showBestMove={showBestMove}
            onToggleBestMove={toggleBestMove}
          />
        </div>

        <div className="sidebar-column">
          <MoveHistory history={history} />
          <AnalysisPanel analysis={analysis} isAnalyzing={isAnalyzing} error={error} />
        </div>
      </main>
    </div>
  );
}
