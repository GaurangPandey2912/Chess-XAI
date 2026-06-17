import { useMemo } from "react";
import { Chessboard } from "react-chessboard";

function EvalBar({ evaluation }) {
  const score = evaluation ?? 0;
  const whitePct = Math.min(100, Math.max(0, 50 + score * 8));
  const barColor = score >= 0 ? "var(--eval-positive)" : "var(--eval-negative)";

  return (
    <div className="eval-bar">
      <div className="eval-bar-label top">
        <span className="eval-icon">♔</span>
        <span className="eval-score">{score > 0 ? `+${score.toFixed(1)}` : score.toFixed(1)}</span>
      </div>
      <div className="eval-bar-track">
        <div
          className="eval-bar-fill"
          style={{ height: `${whitePct}%`, backgroundColor: barColor }}
        />
      </div>
      <div className="eval-bar-label bottom">
        <span className="eval-icon">♚</span>
        <span className="eval-score">{score < 0 ? `+${Math.abs(score).toFixed(1)}` : (score * -1).toFixed(1)}</span>
      </div>
    </div>
  );
}

function GameControls({ onNewGame, onUndo, onFlip, onAnalyze, canUndo, isAnalyzing, gameOver, showBestMove, onToggleBestMove }) {
  return (
    <div className="game-controls">
      <button className="btn btn-icon" onClick={onNewGame} title="New game">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M3 3l7.5 7.5M3 11.5l7.5-7.5M9 3v8h8"></path>
        </svg>
        New Game
      </button>
      <button className="btn btn-icon" onClick={onUndo} disabled={!canUndo} title="Undo move">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="1 4 1 10 7 10"></polyline>
          <path d="M3.51 15a9 9 0 102.13-9.36L1 10"></path>
        </svg>
        Undo
      </button>
      <button className="btn btn-icon" onClick={onFlip} title="Flip board">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
          <circle cx="12" cy="12" r="3"></circle>
        </svg>
        Flip
      </button>
      <button
        className={`btn btn-icon ${showBestMove ? "active" : ""}`}
        onClick={onToggleBestMove}
        title={showBestMove ? "Hide best move highlight" : "Show best move highlight"}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"></path>
        </svg>
        Best Move
      </button>
      <button
        className="btn btn-primary"
        onClick={onAnalyze}
        disabled={isAnalyzing || gameOver}
        title="Analyze with AI"
      >
        {isAnalyzing ? (
          <>
            <span className="spinner" />
            Analyzing...
          </>
        ) : (
          <>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <path d="M12 16v-4"></path>
              <path d="M12 8h.01"></path>
            </svg>
            Analyze Position
          </>
        )}
      </button>
    </div>
  );
}

export default function ChessBoardSection({
  fen,
  orientation,
  onPieceDrop,
  analysis,
  isAnalyzing,
  error,
  gameOver,
  onNewGame,
  onUndo,
  onFlip,
  onAnalyze,
  history,
  showBestMove,
  onToggleBestMove,
}) {
  const customStyles = useMemo(() => {
    const styles = {};
    if (showBestMove && analysis?.bestMove) {
      styles[analysis.bestMove.from] = {
        background: "radial-gradient(circle, rgba(50, 154, 240, 0.4) 0%, rgba(50, 154, 240, 0.15) 60%, transparent 70%)",
        borderRadius: "50%",
      };
      styles[analysis.bestMove.to] = {
        background: "radial-gradient(circle, rgba(50, 154, 240, 0.5) 0%, rgba(50, 154, 240, 0.2) 60%, transparent 70%)",
        borderRadius: "50%",
      };
    }
    return styles;
  }, [analysis, showBestMove]);

  return (
    <div className="board-section">
      <div className="board-container">
        <EvalBar evaluation={analysis?.evaluationAfter ?? null} />
        <div className="board-wrapper">
          <Chessboard
            id="main-board"
            position={fen}
            onPieceDrop={onPieceDrop}
            boardOrientation={orientation}
            boardWidth={480}
            customSquareStyles={customStyles}
            animationDuration={200}
            areArrowsAllowed={false}
          />
          {isAnalyzing && (
            <div className="board-overlay">
              <span className="spinner large" />
              <span>Analyzing position...</span>
            </div>
          )}
        </div>
      </div>

      <GameControls
        onNewGame={onNewGame}
        onUndo={onUndo}
        onFlip={onFlip}
        onAnalyze={onAnalyze}
        canUndo={history.length > 0}
        isAnalyzing={isAnalyzing}
        gameOver={gameOver}
        showBestMove={showBestMove}
        onToggleBestMove={onToggleBestMove}
      />

      {error && (
        <div className="error-banner">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          {error}
        </div>
      )}
    </div>
  );
}
