import { useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import axios from "axios";

function App() {
  const [game, setGame] = useState(new Chess());
  const [explanation, setExplanation] = useState("");

  const onPieceDrop = async (sourceSquare, targetSquare) => {
    const gameCopy = new Chess(game.fen());
  
    const move = gameCopy.move({
      from: sourceSquare,
      to: targetSquare,
      promotion: "q",
    });
  
    if (!move) return false;
  
    setGame(gameCopy);
  
    try {
      const res = await axios.get("http://127.0.0.1:8000/explain_move", {
        params: { fen: gameCopy.fen() },
      });
      setExplanation(res.data.result);
    } catch (e) {
      console.error(e);
    }
  
    return true;
  };
  

  return (
    <div style={{ display: "flex", gap: "20px", padding: "20px" }}>
      <Chessboard
        position={game.fen()}
        onPieceDrop={onPieceDrop}
      />
      <div style={{ width: "420px", whiteSpace: "pre-wrap" }}>
        <h2>AI Coach Explanation</h2>
        {explanation}
      </div>
    </div>
  );
}

export default App;
