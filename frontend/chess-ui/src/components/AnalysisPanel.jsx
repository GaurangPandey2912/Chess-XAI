function EvalChange({ before, after, change }) {
  if (change == null) return null;
  const improved = change > 0;
  const worsened = change < 0;
  const diffStr = `${change >= 0 ? "+" : ""}${change.toFixed(2)}`;
  return (
    <div className="eval-change">
      <div className="eval-badge before">{before ?? "—"}</div>
      <div className="eval-arrow">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
          {improved ? <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" /> :
           worsened ? <polyline points="23 18 13.5 8.5 8.5 13.5 1 6" /> :
           <line x1="23" y1="12" x2="1" y2="12" />}
        </svg>
      </div>
      <div className="eval-badge after">{after ?? "—"}</div>
      <div className={`eval-diff ${improved ? "positive" : worsened ? "negative" : ""}`}>
        {diffStr}
      </div>
    </div>
  );
}

function PlayedMoveCard({ playedMove, beforeTopSuggestions }) {
  if (!playedMove) return null;
  let qualityLabel = null;
  let qualityClass = "";
  let missedMove = null;
  let playedSwing = null;
  if (playedMove.evalAfter != null && playedMove.evalBefore != null && playedMove.bestEvalAfter != null) {
    const sign = playedMove.playedBy === "White" ? 1 : -1;
    const qualityLoss = (playedMove.bestEvalAfter - playedMove.evalAfter) * sign;
    const absDiff = Math.abs(playedMove.bestEvalAfter - playedMove.evalAfter);
    if (qualityLoss <= 0 && absDiff < 0.15) {
      qualityLabel = "Brilliant";
      qualityClass = "quality-brilliant";
    } else if (qualityLoss <= 0) {
      qualityLabel = "Excellent";
      qualityClass = "quality-excellent";
    } else if (absDiff < 0.3) {
      qualityLabel = "Good";
      qualityClass = "quality-good";
    } else if (absDiff < 0.7) {
      qualityLabel = "Inaccuracy";
      qualityClass = "quality-inaccuracy";
    } else if (absDiff < 1.5) {
      qualityLabel = "Mistake";
      qualityClass = "quality-mistake";
    } else {
      qualityLabel = "Blunder";
      qualityClass = "quality-blunder";
    }
    playedSwing = playedMove.evalAfter - playedMove.evalBefore;
    if (qualityLoss > 0 && beforeTopSuggestions && beforeTopSuggestions.length > 0) {
      missedMove = beforeTopSuggestions[0].san;
    }
    if (qualityLoss > 0 && !missedMove) {
      missedMove = playedMove.bestEvalAfter > playedMove.evalAfter ? "a better alternative" : null;
    }
  }
  return (
    <div className="played-move-section">
      <div className="section-header">
        <h3 className="section-title">Your Move</h3>
        <span className="section-turn">{playedMove.playedBy}</span>
      </div>
      <div className="played-move-card">
        <div className="played-move-header">
          <span className="played-move-san">{playedMove.san}</span>
          <div className="played-move-right">
            {playedMove.evalAfter != null && (
              <span className={`top-move-eval ${playedMove.evalAfter >= 0 ? "positive" : "negative"}`}>
                {playedMove.evalAfter >= 0 ? "+" : ""}{playedMove.evalAfter.toFixed(2)}
              </span>
            )}
          </div>
        </div>
        <div className="played-move-swing">
          Eval: <strong>{playedMove.evalBefore != null ? `${playedMove.evalBefore >= 0 ? "+" : ""}${playedMove.evalBefore.toFixed(2)}` : "?"}</strong>
          <span className="swing-arrow">→</span>
          <strong>{playedMove.evalAfter != null ? `${playedMove.evalAfter >= 0 ? "+" : ""}${playedMove.evalAfter.toFixed(2)}` : "?"}</strong>
          {playedSwing != null && (
            <span className={`swing-delta ${playedSwing > 0 ? "positive" : "negative"}`}>
              ({playedSwing >= 0 ? "+" : ""}{playedSwing.toFixed(2)})
            </span>
          )}
        </div>
        <div className="played-move-quality">
          <span className={`quality-label ${qualityClass}`}>{qualityLabel}</span>
          {missedMove && (
            <span className="missed-move">Best was <strong>{missedMove}</strong></span>
          )}
        </div>
      </div>
      {beforeTopSuggestions && beforeTopSuggestions.length > 0 && (
        <div className="before-alternatives">
          <span className="alt-label">What you could have played instead:</span>
          <div className="alt-moves">
            {beforeTopSuggestions.map((m, i) => (
              <span key={i} className="alt-move-tag">
                {m.san} <span className="alt-eval">({m.eval >= 0 ? "+" : ""}{m.eval.toFixed(2)})</span>
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TopMoves({ moves, currentPlayer }) {
  if (!moves || moves.length === 0) return null;
  return (
    <div className="analysis-section">
      <div className="section-header">
        <h3 className="section-title">Best Responses</h3>
        <span className="section-turn">{currentPlayer} to move</span>
      </div>
      <div className="top-moves-list">
        {moves.map((m, i) => (
          <div key={i} className={`top-move-row ${i === 0 ? "best-line" : ""}`}>
            <span className="top-move-rank">{i + 1}.</span>
            <span className="top-move-san">{m.san}</span>
            <span className={`top-move-eval ${m.eval >= 0 ? "positive" : "negative"}`}>
              {m.eval >= 0 ? "+" : ""}{m.eval.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function moveIcon(reason) {
  if (!reason) return "•";
  const r = reason.toLowerCase();
  if (r.includes("checkmate")) return "👑";
  if (r.includes("check")) return "⚡";
  if (r.includes("capture") || r.includes("takes")) return "✖";
  if (r.includes("promot")) return "⬆";
  if (r.includes("castl")) return "🏰";
  if (r.includes("center")) return "🎯";
  if (r.includes("develop")) return "📈";
  if (r.includes("trade")) return "🔄";
  if (r.includes("en passant")) return "♟";
  if (r.includes("king safety") || r.includes("king") || r.includes("shield")) return "🛡️";
  if (r.includes("alternatives") || r.includes("considered")) return "📋";
  if (r.includes("space") || r.includes("spatial")) return "🗺️";
  if (r.includes("open")) return "🔓";
  return "•";
}

function MoveReasoning({ reasons }) {
  if (!reasons || reasons.length === 0) return null;
  return (
    <div className="analysis-section">
      <h3 className="section-title">Move Analysis</h3>
      <ul className="reasoning-list">
        {reasons.map((r, i) => (
          <li key={i}>
            <span className="reason-icon">{moveIcon(r)}</span>
            <span>{r}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function ShapAnalysis({ shapExplanations }) {
  if (!shapExplanations || shapExplanations.length === 0) return null;
  const cleaned = shapExplanations.map((s) => s.replace(/^\s+-\s*/, ""));
  return (
    <div className="analysis-section">
      <h3 className="section-title">Position Factors</h3>
      <ul className="shap-list">
        {cleaned.map((exp, i) => (
          <li key={i}>{exp}</li>
        ))}
      </ul>
      <div className="shap-note">Neural network analysis (SHAP)</div>
    </div>
  );
}

function KingSafetyCard({ kingSafety }) {
  if (!kingSafety) return null;
  const { white, black } = kingSafety;
  const statusColor = (status) => {
    switch (status) {
      case "fortress": return "var(--success)";
      case "safe": return "var(--success)";
      case "adequate": return "var(--warning)";
      case "pressured": return "#f59f00";
      case "exposed": return "var(--danger)";
      case "vulnerable": return "var(--danger)";
      default: return "var(--text-muted)";
    }
  };
  return (
    <div className="analysis-section">
      <h3 className="section-title">King Safety</h3>
      <div className="king-safety-grid">
        <div className="king-safety-item">
          <div className="king-safety-header">
            <span className="king-label">♔ White</span>
            <span className="king-badge" style={{ color: statusColor(white.status), background: `${statusColor(white.status)}18` }}>
              {white.status}
            </span>
          </div>
          <p className="king-desc">{white.description}</p>
          <div className="king-stats">
            <span>🏠 {white.shelterPawns} shield pawns</span>
            <span>⚔️ {white.attackersNear} nearby attackers</span>
            <span>🔓 {white.openFilesNear} open files</span>
          </div>
        </div>
        <div className="king-safety-item">
          <div className="king-safety-header">
            <span className="king-label">♚ Black</span>
            <span className="king-badge" style={{ color: statusColor(black.status), background: `${statusColor(black.status)}18` }}>
              {black.status}
            </span>
          </div>
          <p className="king-desc">{black.description}</p>
          <div className="king-stats">
            <span>🏠 {black.shelterPawns} shield pawns</span>
            <span>⚔️ {black.attackersNear} nearby attackers</span>
            <span>🔓 {black.openFilesNear} open files</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function PositionDetails({ details }) {
  if (!details) return null;
  return (
    <div className="analysis-section">
      <h3 className="section-title">Position Details</h3>
      <div className="details-grid">
        <div className="detail-item">
          <span className="detail-label">King</span>
          <span className="detail-value">♔ {details.whiteKing} / ♚ {details.blackKing}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Material</span>
          <span className="detail-value mono">{details.whiteMaterial}</span>
          <span className="detail-value mono">{details.blackMaterial}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Center</span>
          <span className="detail-value">♔ {details.centerControlWhite} / ♚ {details.centerControlBlack}</span>
        </div>
        <div className="detail-item">
          <span className="detail-label">Mobility</span>
          <span className="detail-value">♔ {details.mobilityWhite} / ♚ {details.mobilityBlack}</span>
        </div>
      </div>
    </div>
  );
}

export default function AnalysisPanel({ analysis, isAnalyzing, error }) {
  if (error) {
    return (
      <div className="analysis-panel">
        <div className="analysis-header"><h2>Analysis</h2></div>
        <div className="analysis-error">
          <h3>Analysis Failed</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (isAnalyzing && !analysis) {
    return (
      <div className="analysis-panel">
        <div className="analysis-empty">
          <span className="spinner large" />
          <p>Analyzing position with AI...</p>
        </div>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="analysis-panel">
        <div className="analysis-empty">
          <h3>AI Coach Ready</h3>
          <p>Make a move or click "Analyze Position" to get an AI-powered explanation of the best move.</p>
        </div>
      </div>
    );
  }

  const {
    bestMove, topMoves, playedMove, beforeTopSuggestions,
    evaluationBefore, evaluationAfter, evalChange,
    formattedEvalBefore, formattedEvalAfter,
    nnPrediction, moveReasoning, shapExplanations,
    kingSafety, positionDetails, currentPlayer,
  } = analysis;

  return (
    <div className="analysis-panel">
      <div className="analysis-header"><h2>Analysis</h2></div>

      <PlayedMoveCard playedMove={playedMove} beforeTopSuggestions={beforeTopSuggestions} />

      {(evaluationBefore != null || evaluationAfter != null) && (
        <div className="analysis-section">
          <div className="section-header">
            <h3 className="section-title">Position Evaluation</h3>
            <span className="section-turn">{currentPlayer} to move</span>
          </div>
          <EvalChange before={formattedEvalBefore} after={formattedEvalAfter} change={evalChange} />
          {nnPrediction != null && (
            <div className="nn-prediction">
              Neural Net: <strong>{nnPrediction >= 0 ? "+" : ""}{nnPrediction.toFixed(2)}</strong>
              <span className="nn-desc">learned positional value</span>
            </div>
          )}
        </div>
      )}

      <TopMoves moves={topMoves} currentPlayer={currentPlayer} />

      <MoveReasoning reasons={moveReasoning} />
      <KingSafetyCard kingSafety={kingSafety} />
      <ShapAnalysis shapExplanations={shapExplanations} />
      <PositionDetails details={positionDetails} />
    </div>
  );
}