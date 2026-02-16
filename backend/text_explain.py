def shap_to_text(feature_names, shap_values, top_k=6):
    """
    Convert SHAP values into human readable explanation with more detailed insights.
    Works for both numpy-array and SHAP Explanation outputs.
    """

    # Handle both SHAP formats
    if hasattr(shap_values, "values"):
        vals = shap_values.values[0]
    else:
        vals = shap_values[0]

    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

    insights = []

    def interpret_feature(name, val):
        name = name.replace("_", " ")
        
        # Material advantages
        if "white" in name and ("pawns" in name or "knights" in name or "bishops" in name or "rooks" in name or "queens" in name):
            piece = name.replace("white ", "")
            if val > 0.5:
                return f"White has significant {piece} advantage (+{val:.1f})"
            elif val > 0.2:
                return f"White has slight {piece} edge"
        elif "black" in name and ("pawns" in name or "knights" in name or "bishops" in name or "rooks" in name or "queens" in name):
            piece = name.replace("black ", "")
            if val > 0.5:
                return f"Black has significant {piece} advantage (+{val:.1f})"
            elif val > 0.2:
                return f"Black has slight {piece} edge"
                
        # Piece advantages  
        elif "knights" in name or "bishops" in name or "rooks" in name or "queens" in name:
            if val > 0.3:
                piece_type = name.split()[1]
                if "white" in name:
                    return f"White has superior {piece_type} positioning"
                elif "black" in name:
                    return f"Black has superior {piece_type} positioning"
                
        # King safety
        elif "king safety" in name:
            if "white" in name and val > 0.3:
                return "White's king position is very safe"
            elif "black" in name and val > 0.3:
                return "Black's king position is very safe"
            elif "white" in name and val < -0.3:
                return "White's king safety is compromised"
            elif "black" in name and val < -0.3:
                return "Black's king safety is compromised"
                
        # Center control
        elif "center control" in name:
            if "white" in name and val > 0.3:
                return "White controls key center squares"
            elif "black" in name and val > 0.3:
                return "Black controls key center squares"
                
        # Mobility
        elif "mobility" in name:
            if "white" in name and val > 0.3:
                return "White has more active piece development"
            elif "black" in name and val > 0.3:
                return "Black has more active piece development"
                
        # Pawn structure
        elif "passed pawns" in name:
            if "white" in name and val > 0.2:
                return "White has dangerous passed pawns"
            elif "black" in name and val > 0.2:
                return "Black has dangerous passed pawns"
        elif "isolated pawns" in name:
            if val > 0.3:
                return "Weak pawn structure with isolated pawns"
                
        # Space advantage
        elif "space" in name:
            if "white" in name and val > 0.2:
                return "White controls more board space"
            elif "black" in name and val > 0.2:
                return "Black controls more board space"
                
        # Piece differences
        elif "minor_piece_diff" in name:
            if val > 0.3:
                return "Better minor piece coordination"
        elif "rook_diff" in name:
            if val > 0.3:
                return "Rook activity advantage"
        elif "queen_diff" in name:
            if val > 0.3:
                return "Queen positioning advantage"
        
        return None  # Skip less significant features

    for name, val in pairs[:top_k]:
        insight = interpret_feature(name, val)
        if insight and abs(val) > 0.15:  # Only include meaningful contributions
            insights.append(insight)

    if not insights:
        return "📊 Positional factors are relatively balanced."
    
    return "📊 Key Positional Factors:\n" + "\n".join(f"• {insight}" for insight in insights[:4])
