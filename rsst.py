
import numpy as np
import pandas as pd


def rsst_table(tracks):
    """Build an RSST summary table from invariance experiment results."""
    columns = [
        "Object Type",
        "Observations",
        "Height Invariance Score",
        "Height Variation (CV)",
        "Range Change (m)",
        "RSST Decision",
    ]

    if tracks is None:
        return pd.DataFrame(columns=columns)

    if isinstance(tracks, pd.DataFrame):
        return tracks.copy()

    if not isinstance(tracks, (list, tuple)) or len(tracks) == 0:
        return pd.DataFrame(columns=columns)

    rows = []

    for item in tracks:
        if not isinstance(item, dict):
            continue

        points = item.get("points") or []
        ranges = []
        shadows = []

        for point in points:
            try:
                if len(point) >= 2:
                    ranges.append(float(point[0]))
                    shadows.append(float(point[1]))
            except (TypeError, ValueError):
                continue

        range_change = (
            max(ranges) - min(ranges) if len(ranges) >= 2 else 0.0
        )

        his = item.get("his")
        cv = item.get("cv")
        count = item.get("n", len(points))
        kind = str(item.get("kind", "unknown"))

        if len(ranges) < 3 or range_change < 1e-6:
            decision = "UNDECIDED"
        elif len(shadows) < 3 or np.std(shadows) < 1e-9:
            decision = "UNDECIDED"
        else:
            correlation = np.corrcoef(ranges, shadows)[0, 1]
            if not np.isfinite(correlation):
                decision = "UNDECIDED"
            elif correlation > 0.5:
                decision = "RANGE-SCALING"
            else:
                decision = "FIXED-SHAPE"

        rows.append({
            "Object Type": kind,
            "Observations": count,
            "Height Invariance Score": his,
            "Height Variation (CV)": cv,
            "Range Change (m)": round(range_change, 3),
            "RSST Decision": decision,
        })

    return pd.DataFrame(rows, columns=columns)
