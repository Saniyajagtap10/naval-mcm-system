
import pandas as pd


def rsst_table(tracks):
    """Convert sonar tracker results into a Streamlit-friendly table."""
    if tracks is None:
        return pd.DataFrame()

    if isinstance(tracks, pd.DataFrame):
        return tracks.copy()

    if not isinstance(tracks, (list, tuple)):
        return pd.DataFrame({"Result": [str(tracks)]})

    if not tracks:
        return pd.DataFrame()

    rows = []

    for item in tracks:
        if isinstance(item, dict):
            rows.append({
                "Hazard Type": item.get("kind", "Unknown"),
                "Observations": item.get("n", 0),
                "History": str(item.get("his", "")),
                "Confidence": item.get("cv", None),
                "Points": str(item.get("points", "")),
            })
        else:
            rows.append({"Result": str(item)})

    return pd.DataFrame(rows)
