import time
import sqlite3

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import streamlit as st

import saug_core as S
import rsst  # NEW: Range-Scaling Shadow Test

# --- Page Configuration ---
st.set_page_config(
    page_title="PRJ-44: SAUG-HPI AUV Sonar Hazard Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- SQLite (event-based logging: only on state change) ---
@st.cache_resource
def init_db():
    conn = sqlite3.connect("sonar_missions.db", check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mission_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, lat REAL, lon REAL, depth REAL,
            classification TEXT, confidence REAL, snr REAL,
            uncertainty REAL, hpi REAL
        )""")
    # migrate older DBs that lack the new columns
    cols = [r[1] for r in conn.execute("PRAGMA table_info(mission_logs)")]
    for c in ("uncertainty", "hpi"):
        if c not in cols:
            conn.execute(f"ALTER TABLE mission_logs ADD COLUMN {c} REAL")
    conn.commit()
    return conn

db_conn = init_db()

# --- Styling ---
st.markdown("""
<style>
.main { background-color: #03071e; color: #f8f9fa; }
.stMetric { background-color: #101c38; padding: 15px; border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,212,255,.15); border: 1px solid #1d3557; }
h1, h2, h3 { color: #00b4d8; font-family: 'Courier New', monospace; }
.stTabs [data-baseweb="tab"] { background-color: #101c38; border-radius: 5px; color: #fff;
                               padding: 10px 20px; font-weight: bold; }
.stTabs [aria-selected="true"] { background-color: #00b4d8 !important; color: #03071e !important; }
@keyframes pulse { 0%{opacity:1} 50%{opacity:.55} 100%{opacity:1} }
.flash-alert { background-color:#720026; color:#ff4d6d; padding:15px; border-radius:8px;
               border:2px solid #ff0054; font-weight:bold; text-align:center; animation:pulse 1.5s infinite; }
.review-alert { background-color:#4a3b00; color:#ffd166; padding:15px; border-radius:8px;
                border:2px solid #ffb703; font-weight:bold; text-align:center; }
.novelty-box { background:#101c38; border-left:4px solid #00b4d8; padding:12px 16px;
               border-radius:6px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

DARK = dict(plot_bgcolor="#03071e", paper_bgcolor="#03071e", font_color="white")

# --- Sidebar ---
st.sidebar.title("🚢 AUV Command Center")
st.sidebar.info("PRJ-44: SAUG-HPI — Shadow-Aware, Uncertainty-Gated Hazard Persistence Engine")
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Stream & Detector Config")

live_stream_toggle = st.sidebar.checkbox("🔴 Active AUV Live Telemetry Stream", value=True)
sea_noise = st.sidebar.slider("Sea / Speckle Noise Level", 0.1, 0.9, 0.45, 0.05)
unc_max = st.sidebar.slider("Uncertainty Gate (max)", 0.05, 0.6, 0.30, 0.05)
hpi_thr = st.sidebar.slider("HPI Alert Threshold", 0.2, 0.9, 0.45, 0.05)
standoff_r = st.sidebar.slider("Standoff Radius (m)", 15, 60, 30, 5)
relook_on = st.sidebar.checkbox("🔁 Uncertainty-triggered re-look", value=True)
if st.sidebar.button("🔄 New Mission Scene"):
    for k in ("scene", "ping_idx", "tracker", "last_state", "alerts_raised"):
        st.session_state.pop(k, None)

# --- Session state: mission scene, tracker, GPS ---
if "scene" not in st.session_state:
    seed = int(time.time()) % 10000
    st.session_state.scene = S.make_scene(seed, length=1200)
    st.session_state.seed = seed
    st.session_state.ping_idx = 0
    st.session_state.tracker = S.HazardTracker()
    st.session_state.last_state = "SAFE"
    st.session_state.alerts_raised = 0
    st.session_state.track = []
    st.session_state.hazards = []
    st.session_state.relooks = 0
    st.session_state.auv_lat0, st.session_state.auv_lon0 = 15.4989, 73.8278

tracker = st.session_state.tracker
tracker.unc_max, tracker.hpi_thr = unc_max, hpi_thr
M_PER_ROW = S.M_PER_ROW
DEG_PER_M = 1 / 111_000

def auv_position(ping_idx):
    north = ping_idx * S.STEP * M_PER_ROW
    return (st.session_state.auv_lat0 + north * DEG_PER_M,
            st.session_state.auv_lon0 + 0.0002 * np.sin(ping_idx / 6.0))

auv_lat, auv_lon = auv_position(st.session_state.ping_idx)
auv_depth = round(-45.0 + 1.5 * np.sin(st.session_state.ping_idx / 5.0), 1)

# --- Header ---
st.title("⚡ PRJ-44: AUV Side-Scan Sonar Hazard Classification Engine")
st.markdown("### SAUG-HPI: Shadow-Aware · Uncertainty-Gated · Persistence-Confirmed Mine Detection")
st.markdown("---")

tab1, tab2, tab_twin, tab3, tab4, tab5, tab6 = st.tabs([
    "🔴 Live Telemetry & Stream",
    "🧪 SAUG-HPI Novelty Study",
    "🧊 3D Hazard Digital Twin",
    "📊 Denoiser Benchmarks",
    "🧠 Acoustic FFT Spectrum",
    "🗺️ GIS Mission Track",
    "💾 SQLite Mission Database",
])

# ==========================================
# TAB 1: LIVE STREAM (SAUG-HPI pipeline)
# ==========================================
with tab1:
    @st.fragment(run_every=2 if live_stream_toggle else None)
    def live_stream():
        ss = st.session_state
        clean_strip, mines, clutter = ss.scene
        max_idx = (clean_strip.shape[0] - S.H) // S.STEP
        if live_stream_toggle and ss.ping_idx < max_idx:
            ss.ping_idx += 1
        i = min(ss.ping_idx, max_idx)
        start = i * S.STEP

        clean, noisy = S.render_ping(clean_strip, start, ss.seed * 7919 + i, sea_noise)
        if relook_on:
            a = S.analyse_adaptive(noisy, clean_strip, start, i, sea_noise, unc_max)
        else:
            a = S.analyse_ping(noisy, seed=i)
            a["relooked"] = False
        if a["relooked"] and live_stream_toggle:
            ss.relooks += 1
        geo = S.measure_object(a["denoised"], a["r"], a["c"])
        geo["cls"] = S.size_class(geo)
        res = tracker.update(i, a) if live_stream_toggle else dict(state=ss.last_state, hpi=0.0, hits=0)
        lat, lon = auv_position(i)
        if live_stream_toggle:
            ss.track.append((lat, lon, res["state"]))

        st.subheader("🔴 Live Side-Scan Ping & Shadow-Aware Inversion")
        c1, c2, c3 = st.columns(3)
        c1.image(noisy, clamp=True, use_container_width=True,
                 caption=f"Raw ping #{i} (range →, along-track ↓)")
        c2.image(a["denoised"], clamp=True, use_container_width=True,
                 caption="Shadow-aware denoised output")
        c3.image(a["wmap"], clamp=True, use_container_width=True,
                 caption="Structure-protection prior (bright = highlight→shadow zone)")

        snr_in = S.snr_db(clean, noisy)
        snr_out = S.snr_db(clean, a["denoised"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Measured SNR Gain", f"{snr_out - snr_in:+.1f} dB", f"{snr_in:.1f} → {snr_out:.1f} dB")
        m2.metric("Shadow-Pair Score", f"{a['strength']:.3f}", f"P(MLO) = {a['p']:.2f}")
        m3.metric("Model Uncertainty", f"{a['unc']:.2f}", "gate OK" if a["unc"] <= unc_max else "ABOVE GATE",
                  delta_color="normal" if a["unc"] <= unc_max else "inverse")
        m4.metric("Hazard Persistence Index", f"{res['hpi']:.2f}", f"{res['hits']}/{tracker.n} pings agree")

        st.markdown("---")
        left, right = st.columns([1, 1])
        with left:
            st.subheader("🎯 Alert Decision")
            state = res["state"]
            if state == "CRITICAL":
                hn = (start + a["r"]) * M_PER_ROW
                he = a["c"] * S.M_PER_COL
                if all(np.hypot(hn - h["n"], he - h["e"]) > 20 for h in ss.hazards):
                    ss.hazards.append(dict(n=hn, e=he))
                hlat = ss.auv_lat0 + hn * DEG_PER_M
                hlon = ss.auv_lon0 + he / (111_000 * np.cos(np.radians(ss.auv_lat0)))
                tw = ss.get("twin")
                if live_stream_toggle and (tw is None or tw["p"] <= a["p"] or not tw["critical"]):
                    ss.twin = dict(den=a["denoised"], r=a["r"], c=a["c"], geo=geo, p=a["p"], ping=i, critical=True)
                st.markdown(f"""<div class="flash-alert">🚨 CONFIRMED {geo['cls'].upper()}<br>
                    Est. height {geo['height_m']:.2f} m · width {geo['width_m']:.1f} m (from shadow geometry)<br>
                    Persistent over {res['hits']} pings · HPI {res['hpi']:.2f}<br>
                    Hazard at Lat {hlat:.5f}°N, Lon {hlon:.5f}°E · Replanning detour (see GIS tab)</div>""",
                            unsafe_allow_html=True)
            elif state == "REVIEW":
                st.markdown(f"""<div class="review-alert">⚠️ HUMAN REVIEW REQUESTED<br>
                    Candidate unstable or not yet persistent (uncertainty {a['unc']:.2f})</div>""",
                            unsafe_allow_html=True)
            else:
                st.success("✅ Safe navigation corridor — no persistent shadow-pair hazard.")

            # Event-based DB logging (only when state changes)
            if state != ss.last_state and live_stream_toggle:
                if state == "CRITICAL":
                    ss.alerts_raised += 1
                db_conn.execute(
                    "INSERT INTO mission_logs (timestamp, lat, lon, depth, classification, confidence, snr, uncertainty, hpi)"
                    " VALUES (?,?,?,?,?,?,?,?,?)",
                    (time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, auv_depth,
                     {"CRITICAL": "Mine-Like Object (MLO)", "REVIEW": "Needs Human Review",
                      "SAFE": "Safe Seabed"}[state],
                     a["p"], float(snr_out), a["unc"], res["hpi"]))
                db_conn.commit()
            ss.last_state = state
            st.caption(f"Alerts raised: {ss.alerts_raised} · Hazards mapped: {len(ss.hazards)} · "
                       f"Extra re-looks taken: {ss.relooks} · Pings: {i}/{max_idx}"
                       + (" · 🔁 re-look used on this ping" if a["relooked"] else ""))

        with right:
            st.subheader("📈 Detector Score Map (highlight→shadow)")
            pm, _ = S.pair_map(a["denoised"])
            fig = px.imshow(pm, color_continuous_scale="Magma", aspect="auto",
                            labels=dict(color="Pair strength"))
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), **DARK)
            st.plotly_chart(fig, use_container_width=True)

        tw = ss.get("twin")
        if live_stream_toggle and a["p"] >= 0.5 and tw is None:
            ss.twin = dict(den=a["denoised"], r=a["r"], c=a["c"], geo=geo, p=a["p"], ping=i, critical=False)
        ss.last_analysis = dict(noisy=noisy, lat=lat, lon=lon, depth=auv_depth)

    live_stream()

# ==========================================
# TAB 2: NOVELTY STUDY (measured, not hardcoded)
# ==========================================
@st.cache_data(show_spinner=False)
def cached_experiment(seed, noise):
    return S.run_experiment(seed, n_scenes=3, noise=noise)

@st.cache_data(show_spinner=False)
def cached_invariance(seed, noise, amp):
    return S.invariance_experiment(seed, noise, amp=amp)

@st.cache_data(show_spinner=False)
def cached_relook(seed, noise):
    return S.relook_study(seed, noise, n_scenes=4)

with tab2:
    st.subheader("🧪 SAUG-HPI vs. Baseline Alerting — Measured Experiment")
    st.markdown("""
    <div class="novelty-box">
    <b>Novel contribution.</b> Existing MCM pipelines denoise first, then threshold a classifier.
    SAUG-HPI closes the loop: (1) the <b>highlight→shadow geometry</b> of a mine is used as a prior inside the
    denoiser, (2) detection is re-scored under input perturbations to obtain <b>uncertainty</b> that gates alarms,
    and (3) a <b>Hazard Persistence Index</b> fuses probability, uncertainty and geographic consistency over
    consecutive pings. All numbers below are computed on synthetic scenes with known ground truth
    (mines + rock/dark-patch clutter), not hardcoded.
    </div>""", unsafe_allow_html=True)

    cA, cB = st.columns([1, 3])
    exp_noise = cA.slider("Experiment noise level", 0.1, 0.9, float(sea_noise), 0.1, key="exp_noise")
    exp_seed = cA.number_input("Random seed", 1, 9999, 1)
    with st.spinner("Running 3 scenes × ~75 pings..."):
        res = cached_experiment(int(exp_seed), float(exp_noise))

    names = {"A": "A) Brightness threshold (baseline)",
             "B": "B) Single-ping shadow-pair threshold",
             "C": "C) SAUG-HPI (proposed)"}
    rows = []
    for k, r in res.items():
        pos, neg = r["tp"] + r["fn"], r["fp"] + r["tn"]
        prec = r["tp"] / max(r["tp"] + r["fp"], 1)
        rows.append({"Method": names[k],
                     "Detection rate (%)": round(100 * r["tp"] / max(pos, 1), 1),
                     "False-alarm rate (%)": round(100 * r["fp"] / max(neg, 1), 1),
                     "Precision (%)": round(100 * prec, 1),
                     "False alarms (count)": r["fp"]})
    df_exp = pd.DataFrame(rows)
    st.dataframe(df_exp, use_container_width=True, hide_index=True)

    g1, g2 = st.columns(2)
    with g1:
        f = px.bar(df_exp, x="Method", y="False-alarm rate (%)", color="False-alarm rate (%)",
                   color_continuous_scale="Reds", title="False-alarm rate (lower is better)")
        f.update_layout(xaxis_tickangle=-20, **DARK)
        st.plotly_chart(f, use_container_width=True)
    with g2:
        f = px.bar(df_exp, x="Method", y="Detection rate (%)", color="Detection rate (%)",
                   color_continuous_scale="Tealgrn", title="Detection rate (higher is better)")
        f.update_layout(xaxis_tickangle=-20, **DARK)
        st.plotly_chart(f, use_container_width=True)
    st.caption("Trade-off to report honestly: persistence adds a few pings of latency, so SAUG-HPI "
               "can miss the first pings of a new object; it trades that for a large cut in false alarms.")

    st.markdown("---")
    st.subheader("🔁 Uncertainty-Triggered Re-Look (Active Sensing) — Measured")
    st.markdown("""<div class="novelty-box">When the detector is <b>unsure</b> (uncertainty above the gate, or probability in the
    grey zone 0.25–0.75), the AUV takes extra looks at the same spot and fuses them. Extra sensing time is spent
    <b>only where it is needed</b>. Below: on those unsure pings only, is the fused decision more often correct?</div>""",
                unsafe_allow_html=True)
    rl_noise = st.slider("Re-look study noise level", 0.3, 1.2, 1.0, 0.1, key="rl_noise")
    with st.spinner("Running re-look study..."):
        rl = cached_relook(int(exp_seed), float(rl_noise))
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Pings evaluated", rl["pings"])
    r2.metric("Unsure pings (re-look fired)", rl["unsure"], f"{100*rl['unsure']/max(rl['pings'],1):.1f}% of pings")
    r3.metric("Single-look correct", f"{rl['single_ok']}/{max(rl['unsure'],1)}")
    r4.metric("After re-look correct", f"{rl['fused_ok']}/{max(rl['unsure'],1)}",
              f"{rl['fused_ok']-rl['single_ok']:+d}")
    st.caption("Honest note: at low noise the detector is rarely unsure, so re-look seldom fires and changes little; "
               "its benefit appears at high noise.")

    st.markdown("---")
    st.subheader("⚖️ Height Invariance Test (Physics Lie-Detector) — Measured")
    st.markdown("""<div class="novelty-box"><b>In simple words:</b> a real object has one fixed height. As the AUV
    sees it from different ranges, its <b>shadow length changes</b> (nearer = shorter, farther = longer) but the
    <b>height worked out from the shadow stays the same</b>. A fake pattern (e.g. a sonar artifact with a fixed-shape
    shadow) does not follow this physics, so its calculated height keeps changing. The
    <b>Height Invariance Score (HIS)</b> measures this consistency. Persistence alone cannot reject such artifacts,
    because they appear in every ping.</div>""", unsafe_allow_html=True)
    swing = st.slider("AUV sideways swing (m) — how much the range to the object changes", 0.5, 4.5, 4.5, 0.5, key="swing")
    with st.spinner("Running invariance experiment..."):
        trk_inv = cached_invariance(int(exp_seed), float(exp_noise), int(round(swing / S.PIX_M)))
    rows_i, scat = [], []
    for kind, label in (("mine", "Real mines (physical)"), ("artifact", "Non-physical artifacts")):
        T = [x for x in trk_inv if x["kind"] == kind and x["n"] >= 4]
        acc = [x for x in T if x["his"] is not None and x["his"] >= 0.5]
        rows_i.append({"Object type": label, "Objects tracked": len(T),
                       "Accepted by persistence only": len(T),
                       "Accepted after HIS gate": len(acc),
                       "Mean height variation (cv)": round(float(np.mean([x["cv"] for x in T])), 3) if T else None})
        for x in T:
            for rng_m, sh_m, h_m in x["points"]:
                scat.append({"Range (m)": rng_m, "Shadow length (m)": sh_m, "Type": label})
    st.dataframe(pd.DataFrame(rows_i), use_container_width=True, hide_index=True)

    # NEW: Range-Scaling Shadow Test (RSST) vs old HIS
    st.markdown("""<div class="novelty-box"><b>NEW — Range-Scaling Shadow Test (RSST):</b> a real object's shadow
    grows <b>in proportion to range</b> (L = k·R), while a fixed-shape artifact keeps the <b>same shadow length</b>.
    RSST compares these two models with a log-likelihood ratio and a sequential decision. It needs no altitude,
    no hand-tuned threshold, and says <b>UNDECIDED</b> when the range did not change enough.</div>""",
                unsafe_allow_html=True)
    st.markdown("**RSST (new) vs HIS (old)**")
    st.dataframe(rsst.rsst_table(trk_inv), use_container_width=True, hide_index=True)

    if scat:
        fs = px.scatter(pd.DataFrame(scat), x="Range (m)", y="Shadow length (m)", color="Type", opacity=0.7,
                        color_discrete_map={"Real mines (physical)": "#00b4d8", "Non-physical artifacts": "#ff4d6d"},
                        title="Shadow length vs range: real objects follow geometry, artifacts do not")
        fs.update_layout(**DARK)
        st.plotly_chart(fs, use_container_width=True)
    st.caption("Honest notes: (1) the test only works when the range to the object actually changes — with a small "
               "swing, real and fake look alike (slide the swing down to see this). (2) The artifacts are my simulated "
               "decoys and the HIS threshold was chosen on this synthetic data, so treat the clean separation as "
               "optimistic until tested on real sonar.")

@st.cache_data(show_spinner=False)
def cached_height_val(seed, noise):
    return S.height_validation(seed, noise, trials=8)

with tab_twin:
    import plotly.graph_objects as go
    st.subheader("🧊 3D Hazard Digital Twin — Shape-from-Shadow Reconstruction")
    st.markdown("""<div class="novelty-box"><b>In simple words:</b> a tall object blocks the sonar and leaves a
    shadow behind it. A taller object makes a longer shadow. Using the shadow length, the AUV's altitude and the range,
    the code works out the object's <b>height</b> with the formula
    <code>height = shadow length × altitude ÷ (range + shadow length)</code>, then builds a rotatable
    <b>3D model</b> with the sonar image draped on top and labels the object by its real size.</div>""",
                unsafe_allow_html=True)
    tw = st.session_state.get("twin")
    if tw is None:
        st.info("No hazard candidate seen yet — let the live stream run until something is detected.")
    else:
        g = tw["geo"]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Estimated height", f"{g['height_m']:.2f} m")
        k2.metric("Width", f"{g['width_m']:.1f} m")
        k3.metric("Shadow length", f"{g['shadow_m']:.1f} m")
        k4.metric("Range from AUV", f"{g['range_m']:.1f} m")
        k5.metric("Status", "CONFIRMED" if tw["critical"] else "Candidate")
        st.markdown(f"**Geometry-based class:** {g['cls']}  ·  seen at ping #{tw['ping']}  ·  P(MLO) = {tw['p']:.2f}")
        x, y, z, patch = S.reconstruct_3d(tw["den"], tw["r"], tw["c"], g["height_m"])
        fig3d = go.Figure(go.Surface(x=x, y=y, z=z, surfacecolor=patch, colorscale="Cividis",
                                     showscale=False, lighting=dict(ambient=0.55, diffuse=0.8, specular=0.3)))
        fig3d.update_layout(
            height=520, margin=dict(t=10, b=10, l=0, r=0),
            scene=dict(xaxis_title="Range (m)", yaxis_title="Along-track (m)", zaxis_title="Height (m)",
                       aspectratio=dict(x=1.3, y=1.0, z=0.45),
                       camera=dict(eye=dict(x=1.5, y=-1.6, z=1.0))), **DARK)
        st.plotly_chart(fig3d, use_container_width=True)
        st.caption("Pseudo-3D from acoustic geometry (not measured bathymetry): the dome height is derived from the "
                   "shadow; the colour is the denoised sonar backscatter. Drag to rotate.")

    with st.expander("✅ How accurate is the height estimate? (measured on objects of known height)"):
        hv = cached_height_val(int(exp_seed), float(exp_noise))
        df_hv = pd.DataFrame([{"True height (m)": r["true_h"], "Estimated (m)": round(r["est_h"], 2),
                               "Mean abs. error (m)": round(r["mae"], 2),
                               "Detected": f"{r['detected']}/{r['trials']}"} for r in hv])
        st.dataframe(df_hv, use_container_width=True, hide_index=True)
        fh = px.line(df_hv, x="True height (m)", y="Estimated (m)", markers=True, title="Estimated vs true height")
        fh.add_shape(type="line", x0=0.3, y0=0.3, x1=1.3, y1=1.3, line=dict(dash="dash", color="#aaa"))
        fh.update_layout(**DARK)
        st.plotly_chart(fh, use_container_width=True)
        st.caption("Dashed line = perfect estimate. Honest note: estimates tend to read ~10–15% low for mid-height "
                   "objects and get worse at very high noise.")

# ==========================================
# TAB 3: DENOISER BENCHMARK (measured)
# ==========================================
@st.cache_data(show_spinner=False)
def cached_denoise(seed, noise):
    return S.denoise_benchmark(seed, n=10, noise=noise)

with tab3:
    st.subheader("📊 Denoiser Comparison — Measured vs. Ground-Truth Clean Scene")
    bench = cached_denoise(int(exp_seed), float(exp_noise))
    df_b = pd.DataFrame([{"Method": k, "PSNR (dB)": round(v[0], 2), "SNR (dB)": round(v[1], 2),
                          "MSE": round(v[2], 4)} for k, v in bench.items()])
    st.dataframe(df_b, use_container_width=True, hide_index=True)
    f = px.bar(df_b, x="Method", y="PSNR (dB)", color="PSNR (dB)", color_continuous_scale="Viridis")
    f.update_layout(xaxis_tickangle=-20, **DARK)
    st.plotly_chart(f, use_container_width=True)

# ==========================================
# TAB 4: FFT (computed from the live ping)
# ==========================================
with tab4:
    st.subheader("🧠 FFT Spectrum of the Live Range Profile")
    la = st.session_state.get("last_analysis")
    if la is None:
        st.info("Waiting for first ping...")
    else:
        profile = la["noisy"].mean(axis=0)
        profile = profile - profile.mean()
        spec = np.abs(np.fft.rfft(profile))
        freqs = np.fft.rfftfreq(len(profile), d=1.0)
        f = px.line(x=freqs, y=spec, labels={"x": "Spatial frequency (cycles / range bin)", "y": "Magnitude"})
        f.update_layout(**DARK)
        st.plotly_chart(f, use_container_width=True)

# ==========================================
# TAB 5: GIS
# ==========================================
with tab5:
    st.subheader("🗺️ AUV Mission Track · Hazard Zones · Auto Standoff Detour")
    trk = st.session_state.track
    hz = st.session_state.hazards
    if trk:
        df_t = pd.DataFrame(trk, columns=["lat", "lon", "state"])
        color = {"SAFE": [0, 212, 255, 200], "REVIEW": [255, 183, 3, 230], "CRITICAL": [255, 0, 84, 255]}
        df_t["color"] = df_t["state"].map(color)
        lat0, lon0 = st.session_state.auv_lat0, st.session_state.auv_lon0
        m_per_deg_lon = 111_000 * np.cos(np.radians(lat0))
        layers = [
            pdk.Layer("PathLayer", data=pd.DataFrame({"path": [df_t[["lon", "lat"]].values.tolist()]}),
                      get_path="path", get_color=[0, 180, 216], width_min_pixels=3),
            pdk.Layer("ScatterplotLayer", data=df_t, get_position="[lon, lat]",
                      get_fill_color="color", get_radius=3, pickable=True),
        ]
        n_end = max(df_t.lat.iloc[-1] - lat0, 0) / DEG_PER_M + 150
        route_info = None
        if hz:
            hlist = [(h["n"], h["e"]) for h in hz]
            rn, re_ = S.standoff_route(hlist, standoff_r, 0, n_end)
            route_info = (rn, re_, S.min_clearance(rn, re_, hlist))
            df_h = pd.DataFrame({"lat": [lat0 + h["n"] * DEG_PER_M for h in hz],
                                 "lon": [lon0 + h["e"] / m_per_deg_lon for h in hz]})
            straight = [[lon0, lat0 + n * DEG_PER_M] for n in (0, n_end)]
            detour = [[lon0 + e / m_per_deg_lon, lat0 + n * DEG_PER_M] for n, e in zip(rn, re_)]
            layers += [
                pdk.Layer("ScatterplotLayer", data=df_h, get_position="[lon, lat]",
                          get_radius=standoff_r, get_fill_color=[255, 0, 84, 70],
                          get_line_color=[255, 0, 84, 255], stroked=True, line_width_min_pixels=2),
                pdk.Layer("ScatterplotLayer", data=df_h, get_position="[lon, lat]",
                          get_radius=4, get_fill_color=[255, 0, 84, 255]),
                pdk.Layer("PathLayer", data=pd.DataFrame({"path": [straight]}), get_path="path",
                          get_color=[160, 160, 160], width_min_pixels=2),
                pdk.Layer("PathLayer", data=pd.DataFrame({"path": [detour]}), get_path="path",
                          get_color=[80, 255, 140], width_min_pixels=4),
            ]
        st.pydeck_chart(pdk.Deck(
            map_style="dark",
            initial_view_state=pdk.ViewState(latitude=df_t.lat.iloc[-1], longitude=df_t.lon.iloc[-1],
                                             zoom=15.5, pitch=30),
            layers=layers))
        st.caption("Cyan = surveyed track · Red zone = confirmed hazard + standoff radius · "
                   "Grey = original transit line · Green = auto-planned safe detour")
        if route_info:
            rn, re_, clr = route_info
            c1, c2, c3 = st.columns(3)
            c1.metric("Hazards mapped", len(hz))
            c2.metric("Min. clearance of detour", f"{clr:.1f} m", f"radius {standoff_r} m")
            extra = float(np.sum(np.hypot(np.diff(rn), np.diff(re_))) - (rn[-1] - rn[0]))
            c3.metric("Extra distance for safety", f"+{extra:.1f} m")
        else:
            st.info("No confirmed hazards yet — the original transit line is safe.")
    else:
        st.info("Start the live stream to build the mission track.")

# ==========================================
# TAB 6: DB
# ==========================================
with tab6:
    st.subheader("💾 Event-Based SQLite Mission Log")
    st.markdown("Rows are written only when the alert state **changes**, so the log is a clean event history.")
    rows = db_conn.execute("SELECT * FROM mission_logs ORDER BY id DESC LIMIT 50").fetchall()
    if rows:
        df_logs = pd.DataFrame(rows, columns=["ID", "Timestamp", "Latitude", "Longitude", "Depth (m)",
                                              "Classification", "Confidence", "SNR (dB)",
                                              "Uncertainty", "HPI"])
        st.dataframe(df_logs, use_container_width=True)
        st.download_button("📥 Export Mission History (CSV)", df_logs.to_csv(index=False).encode("utf-8"),
                           file_name="SQLite_Mission_Logs.csv", mime="text/csv")
    else:
        st.info("No events logged yet. Let the live stream run.")
