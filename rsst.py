"""
RSST - Range-Scaling Shadow Test (add-on for saug_core.py)

Idea (flat seabed):  h = L*alt / (R + L)   =>   L = k * R,  k = h / (alt - h)
  * REAL object      : shadow length L grows in proportion to range R (line through origin).
  * FIXED artifact   : shadow length L stays constant whatever R is (flat line).
Two one-parameter models are compared with a log-likelihood ratio (LLR) and decided with
Wald's sequential thresholds. No altitude needed, no hand-tuned cv threshold, and if the
range did not change enough the answer is UNDECIDED (instead of a wrong guess).
"""
import numpy as np
import pandas as pd

ALPHA = 0.05                                  # target error rate
WALD_A = float(np.log((1 - ALPHA) / ALPHA))   # ~2.94


def range_scaling_llr(points, min_spread_m=1.0, min_n=3):
    """points: [(range_m, shadow_m, height_m), ...] for ONE tracked object.
    Returns dict(llr, k, status) with status in PHYSICAL / ARTIFACT / UNDECIDED."""
    if points is None or len(points) < min_n:
        return dict(llr=None, k=None, status="UNDECIDED")
    R = np.array([p[0] for p in points], dtype=float)
    L = np.array([p[1] for p in points], dtype=float)
    if np.ptp(R) < min_spread_m:              # range barely changed -> models indistinguishable
        return dict(llr=0.0, k=None, status="UNDECIDED")
    k = float(R @ L / (R @ R))                # least-squares slope of L = k*R
    rss_phys = float(np.sum((L - k * R) ** 2))
    rss_art = float(np.sum((L - L.mean()) ** 2))
    llr = 0.5 * len(R) * float(np.log(max(rss_art, 1e-9) / max(rss_phys, 1e-9)))
    status = "PHYSICAL" if llr > WALD_A else "ARTIFACT" if llr < -WALD_A else "UNDECIDED"
    return dict(llr=llr, k=k, status=status)


def rsst_table(tracks, his_thr=0.5, min_n=4):
    """Compare old HIS gate vs new RSST on output of saug_core.invariance_experiment()."""
    rows = []
    for kind, label in (("mine", "Real mines (physical)"), ("artifact", "Non-physical artifacts")):
        T = [t for t in tracks if t["kind"] == kind and t["n"] >= min_n]
        res = [range_scaling_llr(t["points"]) for t in T]
        rows.append({
            "Object type": label,
            "Objects tracked": len(T),
            "HIS accepts (old)": sum(1 for t in T if t["his"] is not None and t["his"] >= his_thr),
            "RSST says PHYSICAL": sum(r["status"] == "PHYSICAL" for r in res),
            "RSST says ARTIFACT": sum(r["status"] == "ARTIFACT" for r in res),
            "RSST UNDECIDED": sum(r["status"] == "UNDECIDED" for r in res),
        })
    return pd.DataFrame(rows)
