def shap_to_text(feature_names, shap_values, top_k=5):
    """
    Convert SHAP values into human readable explanation.
    Works for both numpy-array and SHAP Explanation outputs.
    """

    # Handle both SHAP formats
    if hasattr(shap_values, "values"):
        vals = shap_values.values[0]
    else:
        vals = shap_values[0]

    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

    positive = []
    negative = []

    for name, val in pairs[:top_k]:
        if val > 0:
            positive.append(name.replace("_", " "))
        else:
            negative.append(name.replace("_", " "))

    text = ""

    if negative:
        text += "The position is worse mainly because of "
        text += ", ".join(negative) + ". "

    if positive:
        text += "The position is improved by "
        text += ", ".join(positive) + ". "

    return text
