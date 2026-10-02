"""
PRJ-44: SAUG-HPI AUV Sonar Hazard Engine  --  single-file Streamlit app.
Run:  streamlit run app.py
"""
import time
import sqlite3
from collections import deque

import numpy as np
import pandas as pd
import cv2
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ======================================================================
# CORE (saug_core)
# ======================================================================
W = 200          # range bins (columns)  -> acoustic range direction
H = 160          # along-track rows per ping window
STEP = 10        # rows the AUV advances per ping


# ---------------------------------------------------------------- scene model
def make_scene(seed, n_mines=3, n_clutter=5, length=900):
    """Long seabed strip (waterfall image) with mines (highlight+shadow)
    and clutter (bright rocks WITHOUT shadow, dark patches WITHOUT highlight)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)[None, :]
    rows = np.arange(length)[:, None]
    clean = 0.45 + 0.04 * np.sin(2 * np.pi * (x * 6 + rows / 90.0))
    clean = clean * (1.0 - 0.12 * x)                      # range falloff
    yy, xx = np.ogrid[:length, :W]
    mines, clutter = [], []

    def pick_row(existing):
        for _ in range(100):
            r = int(rng.integers(H, length - H))
            if all(abs(r - e[0]) > H for e in existing):
                return r
        return int(rng.integers(H, length - H))

    for _ in range(n_mines):
        r, c, rad = pick_row(mines + clutter), int(rng.integers(30, W - 80)), int(rng.integers(5, 8))
        clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
        sl = int(rad * rng.uniform(2.5, 3.5))
        clean[(np.abs(yy - r) <= rad * 0.9) & (xx > c + rad) & (xx <= c + rad + sl)] *= 0.2
        mines.append((r, c))
    for i in range(n_clutter):
        r, c, rad = pick_row(mines + clutter), int(rng.integers(30, W - 60)), int(rng.integers(4, 8))
        if i % 2 == 0:   # bright rock, no shadow
            clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
        else:            # dark patch, no highlight
            clean[(np.abs(yy - r) <= rad) & (xx > c) & (xx <= c + 2 * rad)] *= 0.3
        clutter.append((r, c))
    return np.clip(clean, 0, 1), mines, clutter


def render_ping(clean_strip, start, seed, noise=0.25):
    clean = clean_strip[start:start + H]
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(4.0, 1 / 4.0, clean.shape)         # multiplicative speckle
    noisy = np.clip(clean * speckle + rng.normal(0, noise * 0.4, clean.shape), 0, 1)
    return clean.astype(np.float32), noisy.astype(np.float32)


# ------------------------------------------------------------- shadow pair map
def pair_map(img):
    """Highlight-then-shadow strength per pixel (intensity units)."""
    s = cv2.GaussianBlur(img, (5, 5), 0)
    bg = cv2.medianBlur((s * 255).astype(np.uint8), 31).astype(np.float32) / 255.0
    h = np.clip(s - bg, 0, None)
    d = np.clip(bg - s, 0, None)
    h_box = cv2.blur(h, (9, 11))
    d_box = cv2.blur(d, (15, 11))
    d_shift = np.roll(d_box, -13, axis=1)                  # shadow lies downrange
    d_shift[:, -13:] = 0
    pm = np.sqrt(h_box * d_shift)
    pm[:8], pm[-8:], pm[:, :8] = 0, 0, 0
    return pm, h_box


def score_to_prob(strength, t=0.12, w=0.02):
    return float(1.0 / (1.0 + np.exp(-(strength - t) / w)))


def detect(img):
    pm, hb = pair_map(img)
    idx = np.unravel_index(np.argmax(pm), pm.shape)
    return float(pm[idx]), float(hb.max()), idx


# --------------------------------------------------------- shadow-aware denoise
def shadow_aware_denoise(noisy):
    """Blend light filtering (near highlight/shadow structure) with heavy
    filtering (background), weighted by the shadow-pair prior."""
    u8 = (noisy * 255).astype(np.uint8)
    heavy = cv2.GaussianBlur(u8, (0, 0), 2.2)
    light = cv2.GaussianBlur(u8, (0, 0), 0.9)
    pm, _ = pair_map(noisy)
    wmap = cv2.GaussianBlur(np.clip(pm / 0.12, 0, 1), (0, 0), 8)
    wmap = np.clip(wmap * 3.0, 0, 1)
    out = wmap * light + (1 - wmap) * heavy
    return (out / 255.0).astype(np.float32), wmap


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def snr_db(clean, test):
    return 10 * np.log10(np.sum(clean ** 2) / max(np.sum((clean - test) ** 2), 1e-10))


# ------------------------------------------------------------- uncertainty
def perturbed_scores(img, k=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        g = rng.uniform(0.9, 1.1)
        p = np.clip(img * g + rng.normal(0, 0.03, img.shape), 0, 1).astype(np.float32)
        out.append(detect(p)[0])
    return np.array(out)


def analyse_ping(noisy, seed=0):
    den, wmap = shadow_aware_denoise(noisy)
    strength, bright, (r, c) = detect(den)
    scores = perturbed_scores(den, seed=seed)
    p = score_to_prob(strength)
    probs = np.array([score_to_prob(s) for s in scores])
    unc = float(np.clip(np.std(probs) * 2.0, 0, 1))        # spread of probability
    return dict(p=p, unc=unc, strength=strength, brightness=bright, r=int(r), c=int(c),
                denoised=den, wmap=wmap)


# ------------------------------------------------------------- persistence index
class HazardTracker:
    """Hazard Persistence Index over the last N pings."""
    def __init__(self, n=4, cand_p=0.5, unc_max=0.30, tol=14, hpi_thr=0.45, min_hits=3):
        self.hist = deque(maxlen=n)
        self.n, self.cand_p, self.unc_max = n, cand_p, unc_max
        self.tol, self.hpi_thr, self.min_hits = tol, hpi_thr, min_hits

    def update(self, ping_idx, a):
        world_row = ping_idx * STEP + a["r"]
        self.hist.append(dict(p=a["p"], unc=a["unc"], row=world_row, col=a["c"]))
        cur = self.hist[-1]
        if cur["p"] < self.cand_p:
            return dict(state="SAFE", hpi=0.0, hits=0)
        match = [h for h in self.hist if h["p"] >= self.cand_p
                 and abs(h["row"] - cur["row"]) <= self.tol and abs(h["col"] - cur["col"]) <= self.tol]
        persistence = len(match) / self.n
        mp = float(np.mean([h["p"] for h in match]))
        mu = float(np.mean([h["unc"] for h in match]))
        hpi = mp * (1 - mu) * persistence
        if len(match) >= self.min_hits and hpi >= self.hpi_thr and mu <= self.unc_max:
            state = "CRITICAL"
        elif cur["unc"] > self.unc_max or len(match) >= 2:
            state = "REVIEW"
        else:
            state = "SAFE"
        return dict(state=state, hpi=round(hpi, 3), hits=len(match))


# ------------------------------------------------------------- experiments
def hazard_in_view(mines, start, margin=25):
    """True = clearly in view, False = absent, None = partially in view (ignored)."""
    if any(start + margin <= r <= start + H - margin for r, _ in mines):
        return True
    if any(start - 10 <= r <= start + H + 10 for r, _ in mines):
        return None
    return False


def run_experiment(seed, n_scenes=3, length=900, noise=0.5):
    """Compare: A) brightness threshold, B) single-ping shadow-pair threshold,
    C) full SAUG-HPI.  Returns per-ping alarm confusion counts."""
    res = {k: dict(tp=0, fp=0, fn=0, tn=0) for k in ("A", "B", "C")}

    def tally(key, alarm, truth):
        if truth is None:
            return
        res[key][("tp" if alarm else "fn") if truth else ("fp" if alarm else "tn")] += 1

    for s in range(n_scenes):
        clean, mines, _ = make_scene(seed + s, length=length)
        trk = HazardTracker()
        for i, start in enumerate(range(0, length - H, STEP)):
            _, noisy = render_ping(clean, start, seed * 1000 + s * 100 + i, noise)
            a = analyse_ping(noisy, seed=i)
            truth = hazard_in_view(mines, start)
            cstate = trk.update(i, a)["state"]          # tracker must see EVERY ping
            tally("A", a["brightness"] > 0.14, truth)
            tally("B", a["p"] >= 0.5, truth)
            tally("C", cstate == "CRITICAL", truth)
    return res


def denoise_benchmark(seed, n=12, noise=0.25):
    rows = {"Noisy input": [], "Gaussian 5x5": [], "Bilateral": [], "Median 5x5": [], "Shadow-aware (proposed)": []}
    for i in range(n):
        clean, mines, _ = make_scene(seed + i, n_mines=2, n_clutter=3, length=400)
        start = max(0, min(mines[0][0] - H // 2, 400 - H))
        c, noisy = render_ping(clean, start, seed + i, noise)
        u8 = (noisy * 255).astype(np.uint8)
        outs = {
            "Noisy input": noisy,
            "Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
            "Bilateral": cv2.bilateralFilter(u8, 9, 60, 5).astype(np.float32) / 255,
            "Median 5x5": cv2.medianBlur(u8, 5).astype(np.float32) / 255,
            "Shadow-aware (proposed)": shadow_aware_denoise(noisy)[0],
        }
        for k, v in outs.items():
            rows[k].append((psnr(c, v), snr_db(c, v), float(np.mean((c - v) ** 2))))
    return {k: tuple(np.mean(v, axis=0)) for k, v in rows.items()}


# ------------------------------------------------------------- active re-look
def is_ambiguous(a, unc_max=0.30, lo=0.25, hi=0.75):
    """A detection is 'unsure' if uncertainty is high OR probability sits in the grey zone."""
    return a["unc"] > unc_max or lo < a["p"] < hi


def relook_analysis(noisy, clean_strip, start, seed, noise, extra=2):
    """Take extra independent looks at the same spot, multi-look average, re-analyse."""
    frames = [noisy] + [render_ping(clean_strip, start, seed + 1000 * (k + 1), noise)[1]
                        for k in range(extra)]
    avg = np.mean(frames, axis=0).astype(np.float32)
    a = analyse_ping(avg, seed=seed)
    a["noisy_fused"] = avg
    return a


def analyse_adaptive(noisy, clean_strip, start, seed, noise, unc_max=0.30):
    """Normal analysis; only if unsure, spend extra looks (uncertainty-triggered sensing)."""
    a = analyse_ping(noisy, seed=seed)
    if is_ambiguous(a, unc_max):
        b = relook_analysis(noisy, clean_strip, start, seed, noise)
        b["relooked"] = True
        return b
    a["relooked"] = False
    return a


# ------------------------------------------------------------- standoff re-routing
M_PER_ROW = 0.5     # along-track metres per row
M_PER_COL = 0.5     # across-track metres per range bin


def standoff_route(hazards, radius, n0, n1, step=2.0):
    """hazards: list of (north_m, east_m). Straight transit line at east=0 is bent
    westward (smooth bump) so that it stays >= radius from every hazard."""
    n = np.arange(n0, n1, step)
    e = np.zeros_like(n)
    for hn, he in hazards:
        need = he - 1.1 * radius                 # target lateral position (<0 means move west)
        if need < 0:
            e = np.minimum(e, need * np.exp(-((n - hn) / (1.3 * radius)) ** 2))
    return n, e


def min_clearance(n, e, hazards):
    return min(float(np.min(np.hypot(n - hn, e - he))) for hn, he in hazards) if hazards else float("inf")


def relook_study(seed, noise, n_scenes=4):
    """On 'unsure' pings only: is the fused re-look more often correct than the single look?"""
    n_amb = ok_single = ok_fused = total = 0
    for sc in range(n_scenes):
        clean, mines, _ = make_scene(seed + 50 + sc)
        for i, start in enumerate(range(0, 900 - H, STEP)):
            _, nz = render_ping(clean, start, seed * 977 + sc * 100 + i, noise)
            a = analyse_ping(nz, seed=i)
            truth = hazard_in_view(mines, start)
            if truth is None:
                continue
            total += 1
            if not is_ambiguous(a):
                continue
            b = relook_analysis(nz, clean, start, i, noise)
            n_amb += 1
            ok_single += int((a["p"] >= 0.5) == truth)
            ok_fused += int((b["p"] >= 0.5) == truth)
    return dict(pings=total, unsure=n_amb, single_ok=ok_single, fused_ok=ok_fused)


# ------------------------------------------------------------- shape-from-shadow
PIX_M = 0.1        # metres per pixel at object scale (fine acoustic resolution)
RANGE0 = 10.0      # near-range offset (m) before column 0
ALT = 8.0          # AUV altitude above seabed (m)


def height_from_shadow(shadow_m, ground_range_m, alt=ALT):
    """Flat-seabed geometry: h = L * alt / (R + L)."""
    return float(shadow_m * alt / (ground_range_m + shadow_m))


def shadow_length_for_height(h, ground_range_m, alt=ALT):
    """Inverse of the above (used to build physically consistent test objects)."""
    return float(h * ground_range_m / (alt - h))


def measure_object(den, r, c):
    """Measure highlight width and shadow length along the range direction."""
    r0, r1 = max(r - 3, 0), min(r + 4, den.shape[0])
    prof = den[r0:r1].mean(axis=0)
    bg = float(np.median(prof))
    lo, hi = max(c - 10, 0), min(c + 11, len(prof))
    pk = lo + int(np.argmax(prof[lo:hi]))
    half = bg + 0.5 * (prof[pk] - bg)
    a = pk
    while a > 0 and prof[a - 1] > half:
        a -= 1
    b = pk
    while b < len(prof) - 1 and prof[b + 1] > half:
        b += 1
    seg = prof[b + 1:min(b + 60, len(prof))]
    floor = float(seg.min()) if len(seg) else bg
    thr = 0.5 * (bg + floor)                       # half-depth (FWHM-style) edge
    s = b + 1
    while s < len(prof) and prof[s] >= thr and s < b + 4:
        s += 1
    e = s
    while e < len(prof) and prof[e] < thr:
        e += 1
    has_shadow = s < len(prof) and prof[s] < thr and floor < 0.8 * bg
    shadow_px = max(e - s, 0) if has_shadow else 0
    width_m = (b - a + 1) * PIX_M
    shadow_m = shadow_px * PIX_M
    rng_m = RANGE0 + pk * PIX_M
    h = min(height_from_shadow(shadow_m, rng_m), 3.0)
    return dict(width_m=width_m, shadow_m=shadow_m, range_m=rng_m, height_m=h, peak_col=pk)


def size_class(m):
    """Geometry-based class from estimated size."""
    if m["height_m"] < 0.25 or m["width_m"] < 0.5:
        return "Man-Made Debris (small)"
    if m["height_m"] > 1.6 or m["width_m"] > 3.5:
        return "Submerged Wreckage / Large Structure"
    return "Mine-Like Object (MLO-sized)"


def reconstruct_3d(den, r, c, h, half=30):
    """Pseudo-3D relief: object dome scaled by shadow-derived height + gentle seabed texture.
    Returns x, y (metres), z (metres), and the backscatter patch to drape as colour."""
    r0, r1 = max(r - half, 0), min(r + half, den.shape[0])
    c0, c1 = max(c - half, 0), min(c + half + 15, den.shape[1])
    patch = den[r0:r1, c0:c1].astype(np.float32)
    sm = cv2.GaussianBlur(patch, (0, 0), 1.5)
    bg = float(np.median(sm))
    peak = float(max(sm.max() - bg, 1e-3))
    dome = np.clip((sm - bg) / peak, 0, 1) ** 1.3
    texture = cv2.GaussianBlur(patch, (0, 0), 3.0)
    texture = (texture - texture.mean()) * 0.15
    z = texture + dome * h
    x = (np.arange(patch.shape[1]) + c0) * PIX_M
    y = (np.arange(patch.shape[0]) + r0) * PIX_M
    return x, y, z, patch


def height_validation(seed, noise, heights=(0.3, 0.5, 0.7, 0.9, 1.1, 1.3), trials=5):
    """Objects of KNOWN height with physically consistent shadows -> measured estimation error."""
    rows = []
    for hh in heights:
        ests, det = [], 0
        for t in range(trials):
            rng = np.random.default_rng(seed * 100 + t + int(hh * 10))
            c = int(rng.integers(40, 110))
            r = H // 2
            R = RANGE0 + c * PIX_M
            L_px = int(round(shadow_length_for_height(hh, R) / PIX_M))
            rad = 6
            x = np.linspace(0, 1, W)[None, :]
            clean = (0.45 + 0.04 * np.sin(2 * np.pi * (x * 6 + np.arange(H)[:, None] / 90.0))) * (1 - 0.12 * x)
            clean = np.repeat(clean[:1], H, axis=0) if clean.shape[0] != H else clean
            yy, xx = np.ogrid[:H, :W]
            clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
            clean[(np.abs(yy - r) <= rad * 0.9) & (xx > c + rad) & (xx <= c + rad + L_px)] *= 0.2
            _, noisy = render_ping(np.clip(clean, 0, 1), 0, seed * 31 + t, noise)
            den, _ = shadow_aware_denoise(noisy)
            strength, _, (dr, dc) = detect(den)
            if abs(dr - r) > 12 or abs(dc - c) > 14:
                continue
            det += 1
            m = measure_object(den, r, dc)
            ests.append(m["height_m"])
        if ests:
            rows.append(dict(true_h=hh, est_h=float(np.mean(ests)), mae=float(np.mean(np.abs(np.array(ests) - hh))),
                             detected=det, trials=trials))
        else:
            rows.append(dict(true_h=hh, est_h=float("nan"), mae=float("nan"), detected=0, trials=trials))
    return rows


# ------------------------------------------------------------- Height Invariance Test
LAT_AMP = 45   # AUV lateral wobble (pixels) -> object range changes between pings


def lateral_shift(ping_idx, amp=LAT_AMP):
    """Known navigation offset: AUV moves sideways, so the same object appears at different range."""
    return int(round(amp * np.sin(ping_idx / 2.5)))


def make_phys_scene(seed, n_mines=2, n_artifacts=2, length=1100):
    """Scene with PHYSICALLY CONSISTENT mines (shadow length follows geometry for the current range)
    and NON-PHYSICAL artifacts (bright+dark pair with a fixed pixel shadow, e.g. fixed-geometry
    sonar artifact / coincidence) that look like mines in any single ping."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)[None, :]
    rows = np.arange(length)[:, None]
    base = (0.45 + 0.04 * np.sin(2 * np.pi * (x * 6 + rows / 90.0))) * (1 - 0.12 * x)
    objs = []
    for kind, n in (("mine", n_mines), ("artifact", n_artifacts)):
        for _ in range(n):
            for _try in range(100):
                r = int(rng.integers(H, length - H))
                if all(abs(r - o["row"]) > H for o in objs):
                    break
            objs.append(dict(kind=kind, row=r, col=int(rng.integers(65, 95)), rad=int(rng.integers(5, 8)),
                             h=float(rng.uniform(0.6, 1.2)), fixed_L=int(rng.integers(18, 32))))
    return np.clip(base, 0, 1).astype(np.float32), objs


def render_phys_ping(scene, start, ping_idx, seed, noise=0.25, amp=LAT_AMP):
    base, objs = scene
    clean = base[start:start + H].copy()
    sh = lateral_shift(ping_idx, amp)
    yy, xx = np.ogrid[:H, :W]
    for o in objs:
        r = o["row"] - start
        if not (-10 <= r < H + 10):
            continue
        c = o["col"] + sh
        rad = o["rad"]
        clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
        if o["kind"] == "mine":
            L = int(round(shadow_length_for_height(o["h"], RANGE0 + c * PIX_M) / PIX_M))
        else:
            L = o["fixed_L"]
        clean[(np.abs(yy - r) <= rad * 0.9) & (xx > c + rad) & (xx <= c + rad + L)] *= 0.2
    clean = np.clip(clean, 0, 1)
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(4.0, 1 / 4.0, clean.shape)
    return np.clip(clean * speckle + rng.normal(0, noise * 0.4, clean.shape), 0, 1).astype(np.float32)


def height_invariance(heights, cv_scale=0.12, min_n=4):
    """HIS in [0,1]: 1 = height perfectly consistent across pings (physical object)."""
    h = np.asarray(heights, dtype=float)
    if len(h) < min_n:
        return None
    cv = float(np.std(h) / max(np.mean(h), 1e-6))
    return float(np.clip(1.0 - cv / cv_scale, 0.0, 1.0))


def invariance_experiment(seed, noise=0.4, n_scenes=4, length=1100, amp=LAT_AMP):
    """Collect per-object (range, shadow, height) across pings, compute HIS."""
    tracks = []   # one dict per object
    for sc in range(n_scenes):
        scene = make_phys_scene(seed + sc, length=length)
        objs = scene[1]
        data = [[] for _ in objs]
        for i, start in enumerate(range(0, length - H, STEP)):
            noisy = render_phys_ping(scene, start, i, seed * 1000 + sc * 100 + i, noise, amp)
            den, _ = shadow_aware_denoise(noisy)
            strength, _, (r, c) = detect(den)
            if score_to_prob(strength) < 0.5:
                continue
            sh = lateral_shift(i, amp)
            for k, o in enumerate(objs):
                if abs(start + r - o["row"]) <= 12 and abs((c - sh) - o["col"]) <= 14 and 25 <= r <= H - 25:
                    m = measure_object(den, r, c)
                    if m["shadow_m"] > 0:            # failed measurements are excluded, not counted as height 0
                        data[k].append((m["range_m"], m["shadow_m"], m["height_m"]))
        for o, d in zip(objs, data):
            hs = [x[2] for x in d]
            tracks.append(dict(kind=o["kind"], n=len(d), his=height_invariance(hs),
                               cv=(float(np.std(hs) / max(np.mean(hs), 1e-6)) if len(hs) >= 2 else None),
                               points=d))
    return tracks


# ======================================================================
# RSST
# ======================================================================
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
    """Compare old HIS gate vs new RSST on output of invariance_experiment()."""
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


# ======================================================================
# ESCC
# ======================================================================
def analyze_escc(
    image,
    target_height_m=0.5,
    grazing_angle_deg=25.0,
    pixel_size_m=0.05,
    shadow_threshold=0.30,
):
    """
    ESCC prototype:
    compare an observed dark-region estimate with a
    geometry-based expected shadow length.
    """

    arr = np.asarray(
        image.convert("L")
        if isinstance(image, Image.Image)
        else image
    )

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    arr = arr.astype(float)

    if arr.size == 0:
        raise ValueError("Image is empty.")

    arr = (arr - arr.min()) / (np.ptp(arr) + 1e-9)

    profile = np.mean(arr, axis=0)

    threshold = float(np.quantile(profile, shadow_threshold))

    dark = profile <= threshold

    runs = []
    start = None

    for i, value in enumerate(dark):
        if value and start is None:
            start = i

        elif not value and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(dark) - 1))

    observed_px = max(
        (end - begin + 1 for begin, end in runs),
        default=0,
    )

    observed_m = observed_px * pixel_size_m

    angle = np.deg2rad(np.clip(grazing_angle_deg, 1, 89))

    expected_m = target_height_m / np.tan(angle)

    score = float(
        np.exp(
            -abs(observed_m - expected_m)
            / max(expected_m, pixel_size_m, 1e-6)
        )
    )

    return {
        "observed_shadow_m": observed_m,
        "expected_shadow_m": float(expected_m),
        "consistency_score": score,
        "status": "CONSISTENT" if score >= 0.65 else "REVIEW REQUIRED",
        "profile": profile,
        "dark_mask": dark,
    }


def escc_render_page():
    st.subheader("🔬 ESCC — Echo–Shadow Counterfactual Consistency")

    st.markdown("""
    ESCC compares an observed dark-region estimate with a
    geometry-based expected shadow length.

    This is an additional research module. It does not
    replace the existing SAUG-HPI or RSST pipeline.
    """)

    st.warning(
        "Prototype only: this heuristic has not been validated "
        "for real-world underwater hazard detection."
    )

    upload = st.file_uploader(
        "Upload a side-scan sonar image",
        type=["png", "jpg", "jpeg", "bmp"],
        key="escc_upload",
    )

    c1, c2, c3 = st.columns(3)

    height = c1.slider(
        "Assumed target height (m)", 0.1, 3.0, 0.5, 0.1, key="escc_height"
    )

    angle = c2.slider(
        "Grazing angle (degrees)", 5.0, 70.0, 25.0, 1.0, key="escc_angle"
    )

    pixel = c3.number_input(
        "Pixel scale (m/pixel)",
        min_value=0.001,
        max_value=2.0,
        value=0.05,
        step=0.01,
        key="escc_pixel",
    )

    live = st.checkbox(
        "🔴 Analyse the live sonar ping automatically (every 2 s)",
        value=True,
        key="escc_live",
    )

    @st.fragment(run_every=2 if live else None)
    def result_view():
        image = None
        source = ""
        if live and st.session_state.get("last_analysis") is not None:
            noisy = st.session_state.last_analysis["noisy"]
            image = Image.fromarray(
                (np.clip(noisy, 0, 1) * 255).astype(np.uint8)
            ).convert("RGB")
            source = "Live sonar ping"
        elif upload is not None:
            image = Image.open(upload).convert("RGB")
            source = "Input sonar image"

        if image is None:
            st.info(
                "Upload a sonar image, or keep the live stream running, "
                "to calculate the ESCC score."
            )
            return

        try:
            result = analyze_escc(image, height, angle, pixel)

            left, right = st.columns(2)

            with left:
                st.image(image, caption=source, width="stretch")

            with right:
                st.metric(
                    "ESCC Consistency Score",
                    f"{result['consistency_score']:.3f}",
                )
                st.metric(
                    "Observed Dark-Region Estimate",
                    f"{result['observed_shadow_m']:.2f} m",
                )
                st.metric(
                    "Expected Shadow Length",
                    f"{result['expected_shadow_m']:.2f} m",
                )

                if result["status"] == "CONSISTENT":
                    st.success("Consistent under the selected assumptions.")
                else:
                    st.warning("Mismatch detected — human review recommended.")

            fig, ax = plt.subplots(figsize=(9, 2.5))
            ax.plot(result["profile"])
            ax.set_title("Mean Acoustic Intensity Profile")
            ax.set_xlabel("Range Pixel")
            ax.set_ylabel("Normalized Intensity")
            ax.grid(alpha=0.25)

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "This score measures agreement with user-selected "
                "assumptions. It is not the probability that an "
                "object is a mine or hazard."
            )

        except Exception as exc:
            st.error(f"Could not analyze this image: {exc}")

    result_view()


# ======================================================================
# STREAMLIT APP
# ======================================================================
st.set_page_config(
    page_title="PRJ-44: SAUG-HPI AUV Sonar Hazard Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def init_db():
    conn = sqlite3.connect("sonar_missions.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mission_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, lat REAL, lon REAL, depth REAL,
            classification TEXT, confidence REAL, snr REAL,
            uncertainty REAL, hpi REAL
        )
    """)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(mission_logs)")]
    for column in ("uncertainty", "hpi"):
        if column not in cols:
            conn.execute(f"ALTER TABLE mission_logs ADD COLUMN {column} REAL")
    conn.commit()
    return conn


db_conn = init_db()

st.markdown("""
<style>
.stApp, .main { background-color: #03071e; color: #f8f9fa; }
[data-testid="stSidebar"] { background-color: #050b2a; }
.stMetric { background-color: #101c38; padding: 15px; border-radius: 8px;
            border: 1px solid #1d3557; }
h1, h2, h3 { color: #00b4d8; font-family: 'Courier New', monospace; }
.stTabs [data-baseweb="tab"] { background-color: #101c38; border-radius: 5px;
    color: white; padding: 10px 15px; font-weight: bold; }
.stTabs [aria-selected="true"] { background-color: #00b4d8 !important;
    color: #03071e !important; }
.flash-alert { background-color: #720026; color: #ff4d6d; padding: 15px;
    border-radius: 8px; border: 2px solid #ff0054; font-weight: bold;
    text-align: center; }
.review-alert { background-color: #4a3b00; color: #ffd166; padding: 15px;
    border-radius: 8px; border: 2px solid #ffb703; text-align: center; }
.novelty-box { background: #101c38; border-left: 4px solid #00b4d8;
    padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

DARK = {"plot_bgcolor": "#03071e", "paper_bgcolor": "#03071e", "font_color": "white"}

# ---------------------------------------------------------------- SIDEBAR
st.sidebar.title("🚢 AUV Command Center")
st.sidebar.info(
    "PRJ-44: SAUG-HPI — Shadow-Aware, "
    "Uncertainty-Gated Hazard Persistence Engine"
)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Detector Configuration")

live_stream_toggle = st.sidebar.checkbox("🔴 Active AUV Live Telemetry Stream", value=True)
sea_noise = st.sidebar.slider("Sea / Speckle Noise Level", 0.1, 0.9, 0.45, 0.05)
unc_max = st.sidebar.slider("Uncertainty Gate (max)", 0.05, 0.6, 0.30, 0.05)
hpi_thr = st.sidebar.slider("HPI Alert Threshold", 0.2, 0.9, 0.45, 0.05)
standoff_r = st.sidebar.slider("Standoff Radius (m)", 15, 60, 30, 5)
relook_on = st.sidebar.checkbox("🔁 Uncertainty-triggered re-look", value=True)

if st.sidebar.button("🔄 New Mission Scene"):
    for key in (
        "scene", "ping_idx", "tracker", "last_state", "alerts_raised",
        "track", "hazards", "relooks", "twin", "last_analysis", "den_hist",
    ):
        st.session_state.pop(key, None)

# ---------------------------------------------------------- INITIAL SCENE
if "scene" not in st.session_state:
    seed = int(time.time()) % 10000
    st.session_state.scene = make_scene(seed, length=1200)
    st.session_state.seed = seed

    clean_strip, scene_mines, _ = st.session_state.scene
    if scene_mines:
        first_mine_row = min(row for row, _ in scene_mines)
        max_ping = max(0, (clean_strip.shape[0] - H) // STEP)
        st.session_state.ping_idx = max(
            0, min((first_mine_row - H // 2) // STEP, max_ping)
        )
    else:
        st.session_state.ping_idx = 0

    st.session_state.tracker = HazardTracker()
    st.session_state.last_state = "SAFE"
    st.session_state.alerts_raised = 0
    st.session_state.track = []
    st.session_state.hazards = []
    st.session_state.relooks = 0
    st.session_state.auv_lat0 = 15.4989
    st.session_state.auv_lon0 = 73.8278

tracker = st.session_state.tracker
tracker.unc_max = unc_max
tracker.hpi_thr = hpi_thr

DEG_PER_M = 1 / 111000
REFRESH = 2 if live_stream_toggle else None


def auv_position(ping_idx):
    north = ping_idx * STEP * M_PER_ROW
    return (
        st.session_state.auv_lat0 + north * DEG_PER_M,
        st.session_state.auv_lon0 + 0.0002 * np.sin(ping_idx / 6.0),
    )


auv_lat, auv_lon = auv_position(st.session_state.ping_idx)
auv_depth = round(-45.0 + 1.5 * np.sin(st.session_state.ping_idx / 5.0), 1)

# ----------------------------------------------------------------- HEADER
st.title("⚡ PRJ-44: AUV Side-Scan Sonar Hazard Classification Engine")
st.markdown(
    "### SAUG-HPI: Shadow-Aware · Uncertainty-Gated · "
    "Persistence-Confirmed Mine Detection"
)
st.caption(
    "Interactive research demonstration using generated sonar scenes. "
    "It is not a real-world AUV feed or a validated operational detector."
)
st.markdown("---")

tab1, tab2, tab_twin, tab3, tab4, tab5, tab6, tab_escc = st.tabs([
    "🔴 Live Telemetry & Stream",
    "🧪 SAUG-HPI Novelty Study",
    "🧊 3D Hazard Digital Twin",
    "📊 Denoiser Benchmarks",
    "🧠 Acoustic FFT Spectrum",
    "🗺️ GIS Mission Track",
    "💾 SQLite Mission Database",
    "🔬 ESCC Novelty Lab",
])

# ================================================== TAB 1 — LIVE DETECTION
with tab1:

    @st.fragment(run_every=REFRESH)
    def live_stream():
        ss = st.session_state
        clean_strip, mines, clutter = ss.scene
        max_idx = (clean_strip.shape[0] - H) // STEP

        if live_stream_toggle and ss.ping_idx < max_idx:
            ss.ping_idx += 1

        i = min(ss.ping_idx, max_idx)
        start = i * STEP

        clean, noisy = render_ping(clean_strip, start, ss.seed * 7919 + i, sea_noise)

        if relook_on:
            analysis = analyse_adaptive(noisy, clean_strip, start, i, sea_noise, unc_max)
        else:
            analysis = analyse_ping(noisy, seed=i)
            analysis["relooked"] = False

        if analysis["relooked"] and live_stream_toggle:
            ss.relooks += 1

        geo = measure_object(analysis["denoised"], analysis["r"], analysis["c"])
        geo["cls"] = size_class(geo)

        if live_stream_toggle:
            result = tracker.update(i, analysis)
        else:
            result = {"state": ss.last_state, "hpi": 0.0, "hits": 0}

        lat, lon = auv_position(i)
        if live_stream_toggle:
            ss.track.append((lat, lon, result["state"]))

        st.subheader("🔴 Live Side-Scan Ping & Hazard Detection")
        st.caption(
            "The initial scene is generated for demonstration. "
            "Hazard decisions are calculated by the SAUG-HPI pipeline."
        )

        c1, c2, c3 = st.columns(3)
        c1.image(noisy, clamp=True, width="stretch", caption=f"Raw sonar ping #{i}")
        c2.image(analysis["denoised"], clamp=True, width="stretch",
                 caption="Denoised sonar output")
        c3.image(analysis["wmap"], clamp=True, width="stretch",
                 caption="Highlight-to-shadow protection map")

        snr_in = snr_db(clean, noisy)
        snr_out = snr_db(clean, analysis["denoised"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Measured SNR Gain", f"{snr_out - snr_in:+.1f} dB",
                  f"{snr_in:.1f} → {snr_out:.1f} dB")
        m2.metric("Shadow-Pair Score", f"{analysis['strength']:.3f}",
                  f"P(MLO) = {analysis['p']:.2f}")
        m3.metric("Model Uncertainty", f"{analysis['unc']:.2f}",
                  "Gate OK" if analysis["unc"] <= unc_max else "Above gate")
        m4.metric("Hazard Persistence Index", f"{result['hpi']:.2f}",
                  f"{result['hits']}/{tracker.n} pings agree")

        left, right = st.columns(2)

        with left:
            st.subheader("🎯 Hazard Classification")
            state = result["state"]

            if state == "CRITICAL":
                hn = (start + analysis["r"]) * M_PER_ROW
                he = analysis["c"] * M_PER_COL

                if all(np.hypot(hn - h["n"], he - h["e"]) > 20 for h in ss.hazards):
                    ss.hazards.append({"n": hn, "e": he})

                hlat = ss.auv_lat0 + hn * DEG_PER_M
                hlon = ss.auv_lon0 + he / (111000 * np.cos(np.radians(ss.auv_lat0)))

                old_twin = ss.get("twin")
                if live_stream_toggle and (
                    old_twin is None
                    or old_twin["p"] <= analysis["p"]
                    or not old_twin["critical"]
                ):
                    ss.twin = {
                        "den": analysis["denoised"], "r": analysis["r"],
                        "c": analysis["c"], "geo": geo, "p": analysis["p"],
                        "ping": i, "critical": True,
                    }

                st.markdown(
                    f"""<div class="flash-alert">
                    🚨 CONFIRMED {geo['cls'].upper()}<br>
                    Estimated height: {geo['height_m']:.2f} m<br>
                    Width: {geo['width_m']:.1f} m<br>
                    Persistence: {result['hits']} pings<br>
                    HPI: {result['hpi']:.2f}<br>
                    Estimated position: {hlat:.5f}, {hlon:.5f}
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif state == "REVIEW":
                st.markdown(
                    f"""<div class="review-alert">
                    ⚠️ HUMAN REVIEW REQUIRED<br>
                    Candidate is uncertain or not yet persistent.<br>
                    Uncertainty: {analysis['unc']:.2f}
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.success("✅ No persistent hazard confirmed in this ping.")

            if state != ss.last_state and live_stream_toggle:
                if state == "CRITICAL":
                    ss.alerts_raised += 1
                db_conn.execute(
                    """INSERT INTO mission_logs
                    (timestamp, lat, lon, depth, classification,
                     confidence, snr, uncertainty, hpi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, auv_depth,
                        {"CRITICAL": "Mine-Like Object",
                         "REVIEW": "Needs Human Review",
                         "SAFE": "Safe Seabed"}[state],
                        analysis["p"], float(snr_out),
                        analysis["unc"], result["hpi"],
                    ),
                )
                db_conn.commit()

            ss.last_state = state

            st.caption(
                f"Alerts: {ss.alerts_raised} · Hazards mapped: {len(ss.hazards)} · "
                f"Re-looks: {ss.relooks} · Ping: {i}/{max_idx}"
            )

        with right:
            st.subheader("📈 Detector Score Map")
            pm, _ = pair_map(analysis["denoised"])
            fig = px.imshow(pm, color_continuous_scale="Magma", aspect="auto",
                            labels={"color": "Pair strength"})
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), **DARK)
            st.plotly_chart(fig, width="stretch", key=f"pairmap_{i}")

        if live_stream_toggle and analysis["p"] >= 0.5 and ss.get("twin") is None:
            ss.twin = {
                "den": analysis["denoised"], "r": analysis["r"],
                "c": analysis["c"], "geo": geo, "p": analysis["p"],
                "ping": i, "critical": False,
            }

        ss.last_analysis = {"noisy": noisy, "lat": lat, "lon": lon, "depth": auv_depth}

    live_stream()


# ============================================= TAB 2 — NOVELTY STUDY
@st.cache_data(show_spinner=False)
def cached_experiment(seed, noise):
    return run_experiment(seed, n_scenes=3, noise=noise)


@st.cache_data(show_spinner=False)
def cached_invariance(seed, noise, amp):
    return invariance_experiment(seed, noise, amp=amp)


@st.cache_data(show_spinner=False)
def cached_relook(seed, noise):
    return relook_study(seed, noise, n_scenes=4)


@st.cache_data(show_spinner=False)
def cached_height_validation(seed, noise):
    return height_validation(seed, noise, trials=8)


@st.cache_data(show_spinner=False)
def cached_denoiser_benchmark(seed, noise):
    return denoise_benchmark(seed, n=10, noise=noise)


with tab2:
    st.subheader("🧪 SAUG-HPI vs Baseline — Measured Experiment")
    st.markdown(
        """<div class="novelty-box">
        SAUG-HPI combines highlight-to-shadow geometry, uncertainty gating,
        and persistence across consecutive pings. The experiments use
        generated scenes with simulated objects and clutter.
        </div>""",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([1, 3])
    exp_noise = col_a.slider("Experiment noise", 0.1, 0.9, float(sea_noise), 0.1,
                             key="exp_noise")
    exp_seed = col_a.number_input("Random seed", 1, 9999, 1)

    auto_seed = col_a.checkbox("🔁 Auto-run new random seed (every 30 s)", value=False,
                               key="auto_seed")
    run_now = col_a.button("▶ Run experiments", key="run_novelty")
    if run_now:
        st.session_state.novelty_ready = True
    if "seed_counter" not in st.session_state:
        st.session_state.seed_counter = 0

    @st.fragment(run_every=30 if (auto_seed and st.session_state.get("novelty_ready")) else None)
    def novelty_study():
        if not st.session_state.get("novelty_ready"):
            st.info("Click **▶ Run experiments** to run the SAUG-HPI study "
                    "(takes about 10-30 seconds).")
            return
        if auto_seed:
            st.session_state.seed_counter += 1
        seed_eff = int(exp_seed) + (st.session_state.seed_counter if auto_seed else 0)
        st.caption(f"Scenes generated with seed **{seed_eff}**")
        with st.spinner("Running synthetic-scene experiment..."):
            exp_result = cached_experiment(seed_eff, float(exp_noise))

        method_names = {
            "A": "Brightness threshold baseline",
            "B": "Single-ping shadow-pair",
            "C": "SAUG-HPI proposed",
        }

        rows = []
        for key, v in exp_result.items():
            positives = v["tp"] + v["fn"]
            negatives = v["fp"] + v["tn"]
            precision = v["tp"] / max(v["tp"] + v["fp"], 1)
            rows.append({
                "Method": method_names.get(key, key),
                "Detection rate (%)": round(100 * v["tp"] / max(positives, 1), 1),
                "False-alarm rate (%)": round(100 * v["fp"] / max(negatives, 1), 1),
                "Precision (%)": round(100 * precision, 1),
                "False alarms": v["fp"],
            })

        df_exp = pd.DataFrame(rows)
        st.dataframe(df_exp, width="stretch", hide_index=True)

        chart1, chart2 = st.columns(2)
        with chart1:
            fig = px.bar(df_exp, x="Method", y="False-alarm rate (%)",
                         color="False-alarm rate (%)", color_continuous_scale="Reds",
                         title="False-alarm rate")
            fig.update_layout(xaxis_tickangle=-20, **DARK)
            st.plotly_chart(fig, width="stretch")
        with chart2:
            fig = px.bar(df_exp, x="Method", y="Detection rate (%)",
                         color="Detection rate (%)", color_continuous_scale="Tealgrn",
                         title="Detection rate")
            fig.update_layout(xaxis_tickangle=-20, **DARK)
            st.plotly_chart(fig, width="stretch")

        st.caption("Results are simulation measurements, not validated performance "
                   "on operational sonar data.")

        st.markdown("---")
        st.subheader("🔁 Uncertainty-Triggered Re-Look")
        rl_noise = st.slider("Re-look study noise", 0.3, 1.2, 1.0, 0.1, key="rl_noise")

        with st.spinner("Running re-look experiment..."):
            rl = cached_relook(seed_eff, float(rl_noise))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Pings evaluated", rl["pings"])
        r2.metric("Uncertain pings", rl["unsure"])
        r3.metric("Single-look correct", f"{rl['single_ok']}/{max(rl['unsure'], 1)}")
        r4.metric("After re-look correct", f"{rl['fused_ok']}/{max(rl['unsure'], 1)}")

        st.markdown("---")
        st.subheader("⚖️ Height Invariance Test")
        swing = st.slider("AUV sideways swing (m)", 0.5, 4.5, 4.5, 0.5, key="swing")

        with st.spinner("Running height-invariance experiment..."):
            track_invariance = cached_invariance(
                seed_eff, float(exp_noise), int(round(swing / PIX_M))
            )

        rows_his, scatter_data = [], []
        for kind, label in (("mine", "Simulated real mines"),
                            ("artifact", "Simulated artifacts")):
            tracks = [t for t in track_invariance if t["kind"] == kind and t["n"] >= 4]
            accepted = [t for t in tracks if t["his"] is not None and t["his"] >= 0.5]
            rows_his.append({
                "Object type": label,
                "Objects tracked": len(tracks),
                "Accepted after HIS gate": len(accepted),
                "Mean height variation (CV)": round(
                    float(np.mean([t["cv"] for t in tracks])), 3
                ) if tracks else None,
            })
            for t in tracks:
                for range_m, shadow_m, height_m in t["points"]:
                    scatter_data.append({"Range (m)": range_m,
                                         "Shadow length (m)": shadow_m, "Type": label})

        st.dataframe(pd.DataFrame(rows_his), width="stretch", hide_index=True)

        st.subheader("RSST — Range-Scaling Shadow Test")
        st.markdown("RSST compares how shadow length changes with range. "
                    "It is a research test using simulated tracks.")
        st.dataframe(rsst_table(track_invariance), width="stretch", hide_index=True)

        if scatter_data:
            fig = px.scatter(pd.DataFrame(scatter_data), x="Range (m)",
                             y="Shadow length (m)", color="Type", opacity=0.7,
                             title="Shadow length versus range")
            fig.update_layout(**DARK)
            st.plotly_chart(fig, width="stretch")

    novelty_study()


# ============================================ TAB 3 — 3D DIGITAL TWIN
with tab_twin:

    @st.fragment(run_every=REFRESH)
    def twin_view():
        st.subheader("🧊 3D Hazard Digital Twin")
        twin = st.session_state.get("twin")

        if twin is None:
            st.info("Waiting for a hazard candidate. Keep the live stream running.")
            return

        g = twin["geo"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Estimated height", f"{g['height_m']:.2f} m")
        k2.metric("Width", f"{g['width_m']:.1f} m")
        k3.metric("Shadow length", f"{g['shadow_m']:.1f} m")
        k4.metric("Range", f"{g['range_m']:.1f} m")
        k5.metric("Status", "CONFIRMED" if twin["critical"] else "Candidate")

        x, y, z, patch = reconstruct_3d(twin["den"], twin["r"], twin["c"],
                                        g["height_m"])
        fig3d = go.Figure(go.Surface(
            x=x, y=y, z=z, surfacecolor=patch, colorscale="Cividis",
            showscale=False,
            lighting={"ambient": 0.55, "diffuse": 0.8, "specular": 0.3},
        ))
        fig3d.update_layout(
            height=520, margin=dict(t=10, b=10, l=0, r=0),
            scene={"xaxis_title": "Range (m)", "yaxis_title": "Along-track (m)",
                   "zaxis_title": "Height (m)",
                   "aspectratio": {"x": 1.3, "y": 1.0, "z": 0.45}},
            **DARK,
        )
        st.plotly_chart(fig3d, width="stretch", key=f"twin3d_{twin['ping']}")
        st.caption("Pseudo-3D reconstruction from simulated acoustic geometry; "
                   "not measured bathymetry.")

    twin_view()

    with st.expander("Height validation experiment"):
        height_results = cached_height_validation(int(exp_seed), float(exp_noise))
        df_height = pd.DataFrame([
            {
                "True height (m)": r["true_h"],
                "Estimated height (m)": round(r["est_h"], 2),
                "Mean absolute error (m)": round(r["mae"], 2),
                "Detected": f"{r['detected']}/{r['trials']}",
            }
            for r in height_results
        ])
        st.dataframe(df_height, width="stretch", hide_index=True)


# ======================================== TAB 4 — DENOISER BENCHMARKS
with tab3:

    @st.fragment(run_every=REFRESH)
    def denoiser_view():
        ss = st.session_state
        st.subheader("📊 Live Denoiser Comparison (current sonar ping)")
        la = ss.get("last_analysis")
        if la is None:
            st.info("Waiting for the first ping...")
            return

        clean_strip, _, _ = ss.scene
        max_idx = (clean_strip.shape[0] - H) // STEP
        i = min(ss.ping_idx, max_idx)
        clean, noisy = render_ping(clean_strip, i * STEP,
                                   ss.seed * 7919 + i, sea_noise)
        u8 = (noisy * 255).astype(np.uint8)
        outs = {
            "Noisy input": noisy,
            "Gaussian 5x5": cv2.GaussianBlur(noisy, (5, 5), 0),
            "Bilateral": cv2.bilateralFilter(u8, 9, 60, 5).astype(np.float32) / 255,
            "Median 5x5": cv2.medianBlur(u8, 5).astype(np.float32) / 255,
            "Shadow-aware (proposed)": shadow_aware_denoise(noisy)[0],
        }

        scores = {k: (psnr(clean, v), snr_db(clean, v),
                      float(np.mean((clean - v) ** 2))) for k, v in outs.items()}

        # running average over all pings seen in this mission
        hist = ss.setdefault("den_hist", {"n": 0, "last": -1, "sum": {}})
        if hist["last"] != i:
            hist["last"] = i
            hist["n"] += 1
            for k, v in scores.items():
                prev = hist["sum"].get(k, np.zeros(3))
                hist["sum"][k] = prev + np.array(v)

        cols = st.columns(5)
        for col, (k, v) in zip(cols, outs.items()):
            col.image(v, clamp=True, width="stretch", caption=k)

        df_now = pd.DataFrame([
            {"Method": k,
             "PSNR now (dB)": round(v[0], 2),
             "SNR now (dB)": round(v[1], 2),
             "MSE now": round(v[2], 4),
             f"PSNR avg ({hist['n']} pings)": round(hist["sum"][k][0] / hist["n"], 2)}
            for k, v in scores.items()
        ])
        st.dataframe(df_now, width="stretch", hide_index=True)

        fig = px.bar(df_now, x="Method", y="PSNR now (dB)", color="PSNR now (dB)",
                     color_continuous_scale="Viridis")
        fig.update_layout(xaxis_tickangle=-20, **DARK)
        st.plotly_chart(fig, width="stretch", key=f"den_{i}_{time.time()}")
        st.caption(f"Ping #{i}. Computed live on the current simulated ping; "
                   "the average column accumulates over the mission.")

    denoiser_view()

    with st.expander("Fixed multi-scene benchmark (10 scenes)"):
        benchmark = cached_denoiser_benchmark(int(exp_seed), float(exp_noise))
        st.dataframe(pd.DataFrame([
            {"Method": n, "PSNR (dB)": round(v[0], 2),
             "SNR (dB)": round(v[1], 2), "MSE": round(v[2], 4)}
            for n, v in benchmark.items()
        ]), width="stretch", hide_index=True)


# ============================================== TAB 5 — FFT SPECTRUM
with tab4:

    @st.fragment(run_every=REFRESH)
    def fft_view():
        st.subheader("🧠 FFT Spectrum of the Live Sonar Ping")
        la = st.session_state.get("last_analysis")
        if la is None:
            st.info("Waiting for the first ping...")
            return
        profile = la["noisy"].mean(axis=0)
        profile = profile - profile.mean()
        spectrum = np.abs(np.fft.rfft(profile))
        freqs = np.fft.rfftfreq(len(profile), d=1.0)
        fig = px.line(x=freqs, y=spectrum,
                      labels={"x": "Spatial frequency (cycles / range bin)",
                              "y": "Magnitude"})
        fig.update_layout(**DARK)
        st.plotly_chart(fig, width="stretch", key=f"fft_{time.time()}")

    fft_view()


# ============================================ TAB 6 — GIS MISSION TRACK
with tab5:

    @st.fragment(run_every=REFRESH)
    def gis_view():
        st.subheader("🗺️ AUV Mission Track and Hazard Zones")
        track = st.session_state.track
        hazards = st.session_state.hazards

        if not track:
            st.info("Start the live stream to build the simulated mission track.")
            return

        df_track = pd.DataFrame(track, columns=["lat", "lon", "state"])
        state_colors = {"SAFE": [0, 212, 255, 200], "REVIEW": [255, 183, 3, 230],
                        "CRITICAL": [255, 0, 84, 255]}
        df_track["color"] = df_track["state"].map(state_colors)

        lat0 = st.session_state.auv_lat0
        lon0 = st.session_state.auv_lon0
        m_per_deg_lon = 111000 * np.cos(np.radians(lat0))

        layers = [
            pdk.Layer("PathLayer",
                      data=pd.DataFrame({"path": [df_track[["lon", "lat"]].values.tolist()]}),
                      get_path="path", get_color=[0, 180, 216], width_min_pixels=3),
            pdk.Layer("ScatterplotLayer", data=df_track,
                      get_position="[lon, lat]", get_fill_color="color",
                      get_radius=3, pickable=True),
        ]

        end_north = max(df_track.lat.iloc[-1] - lat0, 0) / DEG_PER_M + 150
        route_info = None

        if hazards:
            hazard_list = [(h["n"], h["e"]) for h in hazards]
            route_n, route_e = standoff_route(hazard_list, standoff_r, 0, end_north)
            clearance = min_clearance(route_n, route_e, hazard_list)
            route_info = (route_n, route_e, clearance)

            df_hazards = pd.DataFrame({
                "lat": [lat0 + h["n"] * DEG_PER_M for h in hazards],
                "lon": [lon0 + h["e"] / m_per_deg_lon for h in hazards],
            })
            straight = [[lon0, lat0 + n * DEG_PER_M] for n in (0, end_north)]
            detour = [[lon0 + e / m_per_deg_lon, lat0 + n * DEG_PER_M]
                      for n, e in zip(route_n, route_e)]

            layers.extend([
                pdk.Layer("ScatterplotLayer", data=df_hazards,
                          get_position="[lon, lat]", get_radius=standoff_r,
                          get_fill_color=[255, 0, 84, 70],
                          get_line_color=[255, 0, 84, 255], stroked=True,
                          line_width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", data=df_hazards,
                          get_position="[lon, lat]", get_radius=4,
                          get_fill_color=[255, 0, 84, 255]),
                pdk.Layer("PathLayer", data=pd.DataFrame({"path": [straight]}),
                          get_path="path", get_color=[160, 160, 160],
                          width_min_pixels=2),
                pdk.Layer("PathLayer", data=pd.DataFrame({"path": [detour]}),
                          get_path="path", get_color=[80, 255, 140],
                          width_min_pixels=4),
            ])

        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(
                latitude=df_track.lat.iloc[-1], longitude=df_track.lon.iloc[-1],
                zoom=15.5, pitch=30),
            layers=layers,
        ))
        st.caption("Cyan: AUV track · Red: hazard zone · Grey: original route · "
                   "Green: computed detour")

        if route_info:
            route_n, route_e, clearance = route_info
            c1, c2, c3 = st.columns(3)
            c1.metric("Mapped hazards", len(hazards))
            c2.metric("Minimum route clearance", f"{clearance:.1f} m")
            extra = float(np.sum(np.hypot(np.diff(route_n), np.diff(route_e)))
                          - (route_n[-1] - route_n[0]))
            c3.metric("Extra detour distance", f"{extra:.1f} m")

    gis_view()


# ============================================ TAB 7 — SQLITE DATABASE
with tab6:

    @st.fragment(run_every=REFRESH)
    def db_view():
        st.subheader("💾 SQLite Mission History")
        rows = db_conn.execute(
            "SELECT * FROM mission_logs ORDER BY id DESC LIMIT 50"
        ).fetchall()
        if rows:
            df_logs = pd.DataFrame(rows, columns=[
                "ID", "Timestamp", "Latitude", "Longitude", "Depth (m)",
                "Classification", "Confidence", "SNR (dB)", "Uncertainty", "HPI",
            ])
            st.dataframe(df_logs, width="stretch", hide_index=True)
            st.download_button(
                "📥 Export Mission History CSV",
                df_logs.to_csv(index=False).encode("utf-8"),
                file_name="SQLite_Mission_Logs.csv", mime="text/csv",
                key=f"dl_{len(rows)}_{rows[0][0]}",
            )
        else:
            st.info("No events logged yet. Keep the live stream running.")

    db_view()


# ============================================== TAB 8 — ESCC NOVELTY LAB
with tab_escc:
    escc_render_page()
