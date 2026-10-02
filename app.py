import time
import sqlite3

import numpy as np
import pandas as pd
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import saug_core as S
import rsst

try:
    import escc_novelty as E
except ImportError:
    E = None

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
        "track", "hazards", "relooks", "twin", "last_analysis",
    ):
        st.session_state.pop(key, None)

# ---------------------------------------------------------- INITIAL SCENE
if "scene" not in st.session_state:
    seed = int(time.time()) % 10000
    st.session_state.scene = S.make_scene(seed, length=1200)
    st.session_state.seed = seed

    clean_strip, scene_mines, _ = st.session_state.scene
    if scene_mines:
        first_mine_row = min(row for row, _ in scene_mines)
        max_ping = max(0, (clean_strip.shape[0] - S.H) // S.STEP)
        st.session_state.ping_idx = max(
            0, min((first_mine_row - S.H // 2) // S.STEP, max_ping)
        )
    else:
        st.session_state.ping_idx = 0

    st.session_state.tracker = S.HazardTracker()
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

M_PER_ROW = S.M_PER_ROW
DEG_PER_M = 1 / 111000
REFRESH = 2 if live_stream_toggle else None


def auv_position(ping_idx):
    north = ping_idx * S.STEP * M_PER_ROW
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
        max_idx = (clean_strip.shape[0] - S.H) // S.STEP

        if live_stream_toggle and ss.ping_idx < max_idx:
            ss.ping_idx += 1

        i = min(ss.ping_idx, max_idx)
        start = i * S.STEP

        clean, noisy = S.render_ping(clean_strip, start, ss.seed * 7919 + i, sea_noise)

        if relook_on:
            analysis = S.analyse_adaptive(noisy, clean_strip, start, i, sea_noise, unc_max)
        else:
            analysis = S.analyse_ping(noisy, seed=i)
            analysis["relooked"] = False

        if analysis["relooked"] and live_stream_toggle:
            ss.relooks += 1

        geo = S.measure_object(analysis["denoised"], analysis["r"], analysis["c"])
        geo["cls"] = S.size_class(geo)

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

        snr_in = S.snr_db(clean, noisy)
        snr_out = S.snr_db(clean, analysis["denoised"])

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
                he = analysis["c"] * S.M_PER_COL

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
            pm, _ = S.pair_map(analysis["denoised"])
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
    return S.run_experiment(seed, n_scenes=3, noise=noise)


@st.cache_data(show_spinner=False)
def cached_invariance(seed, noise, amp):
    return S.invariance_experiment(seed, noise, amp=amp)


@st.cache_data(show_spinner=False)
def cached_relook(seed, noise):
    return S.relook_study(seed, noise, n_scenes=4)


@st.cache_data(show_spinner=False)
def cached_height_validation(seed, noise):
    return S.height_validation(seed, noise, trials=8)


@st.cache_data(show_spinner=False)
def cached_denoiser_benchmark(seed, noise):
    return S.denoise_benchmark(seed, n=10, noise=noise)


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

    with st.spinner("Running synthetic-scene experiment..."):
        exp_result = cached_experiment(int(exp_seed), float(exp_noise))

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
        rl = cached_relook(int(exp_seed), float(rl_noise))

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
            int(exp_seed), float(exp_noise), int(round(swing / S.PIX_M))
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
    st.dataframe(rsst.rsst_table(track_invariance), width="stretch", hide_index=True)

    if scatter_data:
        fig = px.scatter(pd.DataFrame(scatter_data), x="Range (m)",
                         y="Shadow length (m)", color="Type", opacity=0.7,
                         title="Shadow length versus range")
        fig.update_layout(**DARK)
        st.plotly_chart(fig, width="stretch")


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

        x, y, z, patch = S.reconstruct_3d(twin["den"], twin["r"], twin["c"],
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
    st.subheader("📊 Denoiser Comparison")
    benchmark = cached_denoiser_benchmark(int(exp_seed), float(exp_noise))
    df_benchmark = pd.DataFrame([
        {"Method": n, "PSNR (dB)": round(v[0], 2),
         "SNR (dB)": round(v[1], 2), "MSE": round(v[2], 4)}
        for n, v in benchmark.items()
    ])
    st.dataframe(df_benchmark, width="stretch", hide_index=True)
    fig = px.bar(df_benchmark, x="Method", y="PSNR (dB)", color="PSNR (dB)",
                 color_continuous_scale="Viridis")
    fig.update_layout(xaxis_tickangle=-20, **DARK)
    st.plotly_chart(fig, width="stretch")


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
            route_n, route_e = S.standoff_route(hazard_list, standoff_r, 0, end_north)
            clearance = S.min_clearance(route_n, route_e, hazard_list)
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
    if E is None:
        st.info("`escc_novelty.py` is not in this folder, so the ESCC Novelty Lab "
                "is disabled. Add the file next to app.py and reload.")
    else:
        E.render_page()
