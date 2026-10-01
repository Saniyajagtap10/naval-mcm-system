"""
SAUG-HPI core: Shadow-Aware, Uncertainty-Gated Hazard Persistence Index
for side-scan sonar Mine Countermeasures (MCM).

Novelty (combination, not a single known trick):
  1. Shadow-aware denoising  - the highlight->shadow pair of a mine-like object
     is used as a prior that decides WHERE to smooth lightly vs. heavily.
  2. Uncertainty gating      - detection is re-scored under input perturbations;
     unstable detections are routed to HUMAN REVIEW instead of alarming.
  3. Persistence index (HPI) - alarm only when the same object persists over
     consecutive pings at a consistent world position.
"""
import numpy as np
import cv2
from collections import deque

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
            st = trk.update(i, a)["state"]          # tracker must see EVERY ping
            tally("A", a["brightness"] > 0.14, truth)
            tally("B", a["p"] >= 0.5, truth)
            tally("C", st == "CRITICAL", truth)
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


# ------------------------------------------------------------- NEW: active re-look
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


# ------------------------------------------------------------- NEW: standoff re-routing
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


# ------------------------------------------------------------- NEW: shape-from-shadow
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


# ------------------------------------------------------------- NEW: Height Invariance Test
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
