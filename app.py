"""
Complete SAUG-HPI, RSST, and ESCC MCM Dashboard (app.py)
"""
import streamlit as st
import numpy as np
import pandas as pd
import cv2
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque

# --- Page Config (Fixed: layout instead of page_layout) ---
st.set_page_config(
    page_title="SAUG-HPI Underwater MCM Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. SAUG-HPI CORE ENGINE
# ==========================================
W = 200          # range bins (columns)
H = 160          # along-track rows per ping window
STEP = 10        # rows the AUV advances per ping

def make_scene(seed, n_mines=3, n_clutter=5, length=900):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)[None, :]
    rows = np.arange(length)[:, None]
    clean = 0.45 + 0.04 * np.sin(2 * np.pi * (x * 6 + rows / 90.0))
    clean = clean * (1.0 - 0.12 * x)
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
        if i % 2 == 0:
            clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
        else:
            clean[(np.abs(yy - r) <= rad) & (xx > c) & (xx <= c + 2 * rad)] *= 0.3
        clutter.append((r, c))
    return np.clip(clean, 0, 1), mines, clutter

def render_ping(clean_strip, start, seed, noise=0.25):
    clean = clean_strip[start:start + H]
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(4.0, 1 / 4.0, clean.shape)
    noisy = np.clip(clean * speckle + rng.normal(0, noise * 0.4, clean.shape), 0, 1)
    return clean.astype(np.float32), noisy.astype(np.float32)

def pair_map(img):
    s = cv2.GaussianBlur(img, (5, 5), 0)
    bg = cv2.medianBlur((s * 255).astype(np.uint8), 31).astype(np.float32) / 255.0
    h = np.clip(s - bg, 0, None)
    d = np.clip(bg - s, 0, None)
    h_box = cv2.blur(h, (9, 11))
    d_box = cv2.blur(d, (15, 11))
    d_shift = np.roll(d_box, -13, axis=1)
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

def shadow_aware_denoise(noisy):
    u8 = (noisy * 255).astype(np.uint8)
    heavy = cv2.GaussianBlur(u8, (0, 0), 2.2)
    light = cv2.GaussianBlur(u8, (0, 0), 0.9)
    pm, _ = pair_map(noisy)
    wmap = cv2.GaussianBlur(np.clip(pm / 0.12, 0, 1), (0, 0), 8)
    wmap = np.clip(wmap * 3.0, 0, 1)
    out = wmap * light + (1 - wmap) * heavy
    return (out / 255.0).astype(np.float32), wmap

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
    unc = float(np.clip(np.std(probs) * 2.0, 0, 1))
    return dict(p=p, unc=unc, strength=strength, brightness=bright, r=int(r), c=int(c), denoised=den, wmap=wmap)

class HazardTracker:
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
        match = [h for h in self.hist if h["p"] >= self.cand_p and abs(h["row"] - cur["row"]) <= self.tol and abs(h["col"] - cur["col"]) <= self.tol]
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

def hazard_in_view(mines, start, margin=25):
    if any(start + margin <= r <= start + H - margin for r, _ in mines):
        return True
    if any(start - 10 <= r <= start + H + 10 for r, _ in mines):
        return None
    return False

def run_experiment(seed, n_scenes=3, length=900, noise=0.5):
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
            st_val = trk.update(i, a)["state"]
            tally("A", a["brightness"] > 0.14, truth)
            tally("B", a["p"] >= 0.5, truth)
            tally("C", st_val == "CRITICAL", truth)
    return res


# ==========================================
# 2. RSST MODULE
# ==========================================
ALPHA = 0.05
WALD_A = float(np.log((1 - ALPHA) / ALPHA))

def range_scaling_llr(points, min_spread_m=1.0, min_n=3):
    if points is None or len(points) < min_n:
        return dict(llr=None, k=None, status="UNDECIDED")
    R = np.array([p[0] for p in points], dtype=float)
    L = np.array([p[1] for p in points], dtype=float)
    if np.ptp(R) < min_spread_m:
        return dict(llr=0.0, k=None, status="UNDECIDED")
    k = float(R @ L / (R @ R))
    rss_phys = float(np.sum((L - k * R) ** 2))
    rss_art = float(np.sum((L - L.mean()) ** 2))
    llr = 0.5 * len(R) * float(np.log(max(rss_art, 1e-9) / max(rss_phys, 1e-9)))
    status = "PHYSICAL" if llr > WALD_A else "ARTIFACT" if llr < -WALD_A else "UNDECIDED"
    return dict(llr=llr, k=k, status=status)

def rsst_table(tracks, his_thr=0.5, min_n=4):
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

LAT_AMP = 45
PIX_M = 0.1
RANGE0 = 10.0
ALT = 8.0

def lateral_shift(ping_idx, amp=LAT_AMP):
    return int(round(amp * np.sin(ping_idx / 2.5)))

def height_from_shadow(shadow_m, ground_range_m, alt=ALT):
    return float(shadow_m * alt / (ground_range_m + shadow_m))

def shadow_length_for_height(h, ground_range_m, alt=ALT):
    return float(h * ground_range_m / (alt - h))

def make_phys_scene(seed, n_mines=2, n_artifacts=2, length=1100):
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

def measure_object(den, r, c):
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
    thr = 0.5 * (bg + floor)
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

def height_invariance(heights, cv_scale=0.12, min_n=4):
    h = np.asarray(heights, dtype=float)
    if len(h) < min_n:
        return None
    cv = float(np.std(h) / max(np.mean(h), 1e-6))
    return float(np.clip(1.0 - cv / cv_scale, 0.0, 1.0))

def invariance_experiment(seed, noise=0.4, n_scenes=4, length=1100, amp=LAT_AMP):
    tracks = []
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
                    if m["shadow_m"] > 0:
                        data[k].append((m["range_m"], m["shadow_m"], m["height_m"]))
        for o, d in zip(objs, data):
            hs = [x[2] for x in d]
            tracks.append(dict(kind=o["kind"], n=len(d), his=height_invariance(hs),
                               cv=(float(np.std(hs) / max(np.mean(hs), 1e-6)) if len(hs) >= 2 else None),
                               points=d))
    return tracks


# ==========================================
# 3. ESCC MODULE
# ==========================================
def analyze_escc(image, target_height_m=0.5, grazing_angle_deg=25.0, pixel_size_m=0.05, shadow_threshold=0.30):
    arr = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
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
    observed_px = max((end - begin + 1 for begin, end in runs), default=0)
    observed_m = observed_px * pixel_size_m
    angle = np.deg2rad(np.clip(grazing_angle_deg, 1, 89))
    expected_m = target_height_m / np.tan(angle)
    score = float(np.exp(-abs(observed_m - expected_m) / max(expected_m, pixel_size_m, 1e-6)))
    return {
        "observed_shadow_m": observed_m,
        "expected_shadow_m": float(expected_m),
        "consistency_score": score,
        "status": "CONSISTENT" if score >= 0.65 else "REVIEW REQUIRED",
        "profile": profile,
        "dark_mask": dark,
    }


# ==========================================
# 4. STREAMLIT UI & SESSION STATE MANAGEMENT
# ==========================================
st.sidebar.header("🎛️ Mission Controls")
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
noise_level = st.sidebar.slider("Acoustic Noise Level", 0.1, 1.0, 0.4, 0.05)
hpi_thr = st.sidebar.slider("HPI Critical Threshold", 0.1, 0.9, 0.45, 0.05)
unc_max = st.sidebar.slider("Max Uncertainty Gate", 0.1, 0.6, 0.30, 0.05)

if "ping_idx" not in st.session_state:
    st.session_state.ping_idx = 0

if "tracker" not in st.session_state:
    st.session_state.tracker = HazardTracker(
        n=4, cand_p=0.5, unc_max=unc_max, hpi_thr=hpi_thr
    )

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

st.title("🛡️ SAUG-HPI: Underwater Mine Countermeasures")

tab_live, tab_fa, tab_rsst, tab_escc = st.tabs([
    "🔴 Live Sonar Stream", 
    "📊 False Alarm Benchmark", 
    "📐 RSST & Invariance",
    "🔬 ESCC Counterfactual"
])

# --- Tab 1: Live Sonar Stream ---
with tab_live:
    st.subheader("Live AUV Telemetry & Shadow-Aware Detection")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("Simulate Next Ping Step"):
            clean_strip, _, _ = make_scene(int(seed), length=900)
            st.session_state.ping_idx = (st.session_state.ping_idx + 1) % 70
            start = st.session_state.ping_idx * STEP
            _, noisy = render_ping(clean_strip, start, int(seed) + st.session_state.ping_idx, noise_level)
            analysis = analyse_ping(noisy, seed=st.session_state.ping_idx)
            analysis["noisy"] = noisy
            st.session_state.last_analysis = analysis
            
        if st.session_state.last_analysis is not None:
            res_a = st.session_state.last_analysis
            st.image(res_a["noisy"], caption=f"Live Sonar Ping (Step {st.session_state.ping_idx})", use_container_width=True)
        else:
            st.info("Click 'Simulate Next Ping Step' to generate telemetry frames.")
            
    with col2:
        if st.session_state.last_analysis is not None:
            res_a = st.session_state.last_analysis
            tracking_res = st.session_state.tracker.update(st.session_state.ping_idx, res_a)
            st.metric("Detection Probability (p)", f"{res_a['p']:.3f}")
            st.metric("Uncertainty (unc)", f"{res_a['unc']:.3f}")
            st.metric("HPI State", tracking_res["state"])
            st.metric("Persistence Score", f"{tracking_res['hpi']:.3f}")
        else:
            st.write("Awaiting live telemetry data...")

# --- Tab 2: False Alarm Benchmark ---
with tab_fa:
    st.subheader("📊 Detector False Alarm & Confusion Matrix Evaluation")
    st.markdown("Compares **Method A** (Brightness), **Method B** (Single-Ping), and **Method C** (SAUG-HPI).")
    
    if st.button("Run Performance Experiment", key="btn_run_exp"):
        with st.spinner("Running simulation across multiple scenes..."):
            exp_res = run_experiment(seed=int(seed), noise=noise_level)
            
            methods = ["Method A (Brightness)", "Method B (Single-Ping)", "Method C (SAUG-HPI)"]
            keys = ["A", "B", "C"]
            
            df_metrics = pd.DataFrame({
                "Detector": methods,
                "True Positives (TP)": [exp_res[k]["tp"] for k in keys],
                "False Alarms (FP)": [exp_res[k]["fp"] for k in keys],
                "False Negatives (FN)": [exp_res[k]["fn"] for k in keys],
                "True Negatives (TN)": [exp_res[k]["tn"] for k in keys]
            })
            
            st.dataframe(df_metrics, use_container_width=True)
            
            fig = px.bar(
                df_metrics, 
                x="Detector", 
                y=["True Positives (TP)", "False Alarms (FP)"],
                barmode="group",
                title="Performance & False Alarm Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: RSST & Invariance ---
with tab_rsst:
    st.subheader("📐 Range-Scaling Shadow Test (RSST) Table")
    st.markdown("Evaluates whether tracked objects are real physical items or fixed artifacts.")
    
    if st.button("Run RSST Invariance Test", key="btn_run_rsst"):
        with st.spinner("Tracking targets across spatial range shifts..."):
            tracks = invariance_experiment(seed=int(seed), noise=noise_level)
            df_rsst = rsst_table(tracks, his_thr=hpi_thr)
            st.dataframe(df_rsst, use_container_width=True)

# --- Tab 4: ESCC Analysis ---
with tab_escc:
    st.subheader("🔬 ESCC — Echo–Shadow Counterfactual Consistency")
    upload = st.file_uploader("Upload a side-scan sonar image", type=["png", "jpg", "jpeg", "bmp"], key="escc_upload")
    
    c1, c2, c3 = st.columns(3)
    height = c1.slider("Assumed target height (m)", 0.1, 3.0, 0.5, 0.1, key="escc_height")
    angle = c2.slider("Grazing angle (degrees)", 5.0, 70.0, 25.0, 1.0, key="escc_angle")
    pixel = c3.number_input("Pixel scale (m/pixel)", min_value=0.001, max_value=2.0, value=0.05, step=0.01, key="escc_pixel")
    
    img_to_analyze = None
    source_label = ""
    if upload is not None:
        img_to_analyze = Image.open(upload).convert("RGB")
        source_label = "Uploaded Sonar Image"
    elif st.session_state.last_analysis is not None:
        noisy_arr = st.session_state.last_analysis["noisy"]
        img_to_analyze = Image.fromarray((np.clip(noisy_arr, 0, 1) * 255).astype(np.uint8)).convert("RGB")
        source_label = "Live Sonar Ping Frame"
        
    if img_to_analyze is not None:
        try:
            res_escc = analyze_escc(img_to_analyze, height, angle, pixel)
            l_col, r_col = st.columns(2)
            with l_col:
                st.image(img_to_analyze, caption=source_label, use_container_width=True)
            with r_col:
                st.metric("ESCC Consistency Score", f"{res_escc['consistency_score']:.3f}")
                st.metric("Observed Shadow Estimate", f"{res_escc['observed_shadow_m']:.2f} m")
                st.metric("Expected Shadow Length", f"{res_escc['expected_shadow_m']:.2f} m")
                if res_escc["status"] == "CONSISTENT":
                    st.success("Consistent under selected assumptions.")
                else:
                    st.warning("Mismatch detected — human review recommended.")
                    
            fig, ax = plt.subplots(figsize=(8, 2.5))
            ax.plot(res_escc["profile"])
            ax.set_title("Mean Acoustic Intensity Profile")
            ax.set_xlabel("Range Pixel")
            ax.set_ylabel("Normalized Intensity")
            ax.grid(alpha=0.25)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"Analysis failed: {e}")
    else:
        st.info("Upload an image or run a live ping simulation step to analyze ESCC.")
