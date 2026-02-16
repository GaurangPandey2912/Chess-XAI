import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";

function App() {
  const [game, setGame] = useState(new Chess());
  const [explanation, setExplanation] = useState("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const makeAMove = (move) => {
    const gameCopy = new Chess(game.fen());
    const result = gameCopy.move(move);
    setGame(gameCopy);
    return result;
  };

  const analyzePosition = async () => {
    setIsAnalyzing(true);
    try {
      const fen = game.fen();
      console.log("Analyzing position with FEN:", fen);
      const res = await axios.get("http://127.0.0.1:8000/explain_move", {
        params: { fen: fen },
      });
      console.log("API response:", res.data);
      if (res.data.error) {
        setExplanation(`Error: ${res.data.error}`);
      } else {
        setExplanation(res.data.result);
      }
    } catch (e) {
      console.error("API Error:", e);
      setExplanation("Error getting AI explanation. Check if backend is running.");
    }
    setIsAnalyzing(false);
  };

  const onPieceDrop = (sourceSquare, targetSquare) => {
    const move = makeAMove({
      from: sourceSquare,
      to: targetSquare,
      promotion: "q",
    });

    if (move === null) return false;

    analyzePosition();

    return true;
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>Chess AI Coach</h1>
      <div style={{ display: "flex", gap: "20px", alignItems: "flex-start" }}>
        <div>
          <Chessboard 
            position={game.fen()} 
            onPieceDrop={onPieceDrop} 
            boardWidth={400}
          />
          <button 
            onClick={analyzePosition}
            disabled={isAnalyzing}
            style={{ marginTop: "10px", padding: "10px 20px", fontSize: "14px" }}
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Current Position"}
          </button>
        </div>
        <div style={{ width: "400px", whiteSpace: "pre-wrap" }}>
          <h2>AI Coach Explanation</h2>
          {explanation || "Make a move or click 'Analyze' to get an explanation..."}
        </div>
      </div>
    </div>
  );
}

export default App;