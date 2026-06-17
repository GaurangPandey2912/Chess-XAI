import { useRef, useEffect } from "react";

export default function MoveHistory({ history }) {
  const listRef = useRef(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [history.length]);

  const pairs = [];
  for (let i = 0; i < history.length; i += 2) {
    pairs.push({
      number: i / 2 + 1,
      white: history[i]?.san || "",
      black: history[i + 1]?.san || "",
    });
  }

  return (
    <div className="move-history">
      <div className="panel-header">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
        </svg>
        Move History
      </div>
      <div className="move-list" ref={listRef}>
        {history.length === 0 ? (
          <div className="move-list-empty">No moves yet</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th className="move-num-col">#</th>
                <th className="move-white-col">White</th>
                <th className="move-black-col">Black</th>
              </tr>
            </thead>
            <tbody>
              {pairs.map((pair) => (
                <tr key={pair.number}>
                  <td className="move-num">{pair.number}.</td>
                  <td className="move-cell">{pair.white}</td>
                  <td className="move-cell">{pair.black}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
