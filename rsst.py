"""PRJ-44 Range-Scaling Shadow Test utilities."""
import pandas as pd


def rsst_table(trk_inv):
    """Convert range-scaling experiment results into a display table."""
    if isinstance(trk_inv, pd.DataFrame):
        return trk_inv.copy()

    if isinstance(trk_inv, dict):
        # Support either a single result dictionary or named result groups.
        if not trk_inv:
            return pd.DataFrame()

        if all(not isinstance(v, (dict, list, tuple)) for v in trk_inv.values()):
            return pd.DataFrame([trk_inv])

        rows = []
        for key, value in trk_inv.items():
            if isinstance(value, dict):
                rows.append({"Test": key, **value})
            elif isinstance(value, (list, tuple)):
                rows.append({"Test": key, "Results": str(value)})
            else:
                rows.append({"Test": key, "Results": value})
        return pd.DataFrame(rows)

    if isinstance(trk_inv, (list, tuple)):
        return pd.DataFrame(trk_inv)

    return pd.DataFrame({"Results": [str(trk_inv)]})
