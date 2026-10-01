
"""
PRJ-44 | SAUG-HPI AUV Side-Scan Sonar Hazard Engine
Includes ESCC Novelty Lab.

Run:
    streamlit run app.py

Research prototype. Demonstration scenes are simulated.
Not for operational navigation.
"""

import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

import saug_core as S
import rsst
import escc_novelty as E


st.set_page_config(
    page_title="PRJ-44 | AUV Sonar Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = Path("sonar_missions.db")
DARK = {
    "plot_bgcolor": "#07111f",
    "paper_bgcolor": "#07111f",
    "font_color": "#eaf4ff",
}
DEG_PER_M = 1 / 111_000


# ------------------------------- DATABASE ------------------------------------

@st.cache_resource
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS mission_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            lat REAL,
            lon REAL,
            depth REAL,
            classification TEXT,
            confidence REAL,
            snr REAL,
            uncertainty REAL,
            hpi REAL
        )
    """)

    conn.commit()
    return conn


db = init_db()


# -------------------------------- STYLING -------------------------------------

st.markdown("""
<style>
.stApp {
    background: #07111f;
    color: #e8f4ff;
}

[data-testid="stSidebar"] {
    background: #0b192b;
}

h1, h2, h3 {
    color: #55d8ee;
}

div[data-testid="stMetric"] {
    background: #102239;
    border: 1px solid #214967;
    padding: 12px;
    border-radius: 12px;
}

.hero {
    padding: 22px;
    border-radius: 16px;
    background: linear-gradient(
        120deg, #0b2741, #07576a, #142b4b
    );
    border: 1px solid #23617a;
    margin-bottom: 16px;
}

.hero p {
    color: #c7dce9;
    margin-bottom: 0;
}

.note {
    padding: 12px;
    border-left: 4px solid #55d8ee;
    background: #102239;
    border-radius: 7px;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------- SIDEBAR -------------------------------------

with st.sidebar:
    st.markdown("## 🌊 PRJ-44")
    st.caption("SAUG-HPI · Shadow-aware sonar hazard analysis")

    page = st.radio(
        "MISSION CONTROL",
        [
            "Mission Overview",
            "Live Telemetry",
            "SAUG-HPI Study",
            "ESCC Novelty Lab",
            "3D Hazard Twin",
            "Denoiser Benchmarks",
            "Acoustic FFT",
            "GIS Mission Track",
            "Risk & Energy Planner",
            "Mission Database",
        ],
    )

    st.divider()

    live = st.checkbox(
        "Auto-advance live pings",
        value=True,
    )

    noise = st.slider(
        "Sea / speckle noise",
        0.10, 0.90, 0.45, 0.05,
    )

    unc_gate = st.slider(
        "Uncertainty gate",
        0.05, 0.60, 0.30, 0.05,
    )

    hpi_gate = st.slider(
        "HPI alert threshold",
        0.20, 0.90, 0.45, 0.05,
    )

    standoff = st.slider(
        "Hazard standoff radius (m)",
        15, 60, 30, 5,
    )

    relook = st.checkbox(
        "Uncertainty-triggered re-look",
        value=True,
    )

    if st.button("🔄 New mission scene", use_container_width=True):
        for key in (
            "scene",
            "ping_idx",
            "tracker",
            "track",
            "hazards",
            "last_analysis",
            "twin",
        ):
            st.session_state.pop(key, None)

        st.rerun()


# ----------------------------- SESSION STATE ---------------------------------

if "scene" not in st.session_state:
    seed = int(time.time()) % 10000

    st.session_state.seed = seed
    st.session_state.scene = S.make_scene(seed, length=1200)
    st.session_state.ping_idx = 0
    st.session_state.tracker = S.HazardTracker()
    st.session_state.track = []
    st.session_state.hazards = []
    st.session_state.last_analysis = None
    st.session_state.twin = None
    st.session_state.last_state = "SAFE"
    st.session_state.alerts = 0
    st.session_state.auv_lat0 = 15.4989
    st.session_state.auv_lon0 = 73.8278


tracker = st.session_state.tracker
tracker.unc_max = unc_gate
tracker.hpi_thr = hpi_gate

clean_strip, mines, clutter = st.session_state.scene

max_ping = max(
    0,
    (clean_strip.shape[0] - S.H) // S.STEP,
)

ping_idx = min(
    st.session_state.ping_idx,
    max_ping,
)


def auv_position(idx):
    north = idx * S.STEP * S.M_PER_ROW

    return (
        st.session_state.auv_lat0 + north * DEG_PER_M,
        st.session_state.auv_lon0 + 0.0002 * np.sin(idx / 6.0),
    )


def get_analysis(idx, adaptive=True):
    start = idx * S.STEP

    clean, noisy = S.render_ping(
        clean_strip,
        start,
        st.session_state.seed * 7919 + idx,
        noise,
    )

    if adaptive and relook:
        result = S.analyse_adaptive(
            noisy,
            clean_strip,
            start,
            idx,
            noise,
            unc_gate,
        )
    else:
        result = S.analyse_ping(noisy, seed=idx)
        result["relooked"] = False

    return clean, noisy, result, start


def ensure_live_analysis():
    idx = min(st.session_state.ping_idx, max_ping)

    clean, noisy, a, start = get_analysis(idx)

    geo = S.measure_object(
        a["denoised"],
        a["r"],
        a["c"],
    )

    geo["cls"] = S.size_class(geo)

    result = tracker.update(idx, a)
    lat, lon = auv_position(idx)

    st.session_state.last_analysis = {
        "clean": clean,
        "noisy": noisy,
        "analysis": a,
        "geo": geo,
        "result": result,
        "start": start,
        "idx": idx,
        "lat": lat,
        "lon": lon,
        "depth": round(-45 + 1.5 * np.sin(idx / 5), 1),
    }

    if live:
        st.session_state.track.append(
            (lat, lon, result["state"])
        )

    if result["state"] == "CRITICAL":
        north = (start + a["r"]) * S.M_PER_ROW
        east = a["c"] * S.M_PER_COL

        if all(
            np.hypot(north - h["n"], east - h["e"]) > 20
            for h in st.session_state.hazards
        ):
            st.session_state.hazards.append(
                {"n": north, "e": east}
            )

        st.session_state.twin = {
            "den": a["denoised"],
            "r": a["r"],
            "c": a["c"],
            "geo": geo,
            "p": a["p"],
            "ping": idx,
            "critical": True,
        }

    elif a["p"] >= 0.5 and st.session_state.twin is None:
        st.session_state.twin = {
            "den": a["denoised"],
            "r": a["r"],
            "c": a["c"],
            "geo": geo,
            "p": a["p"],
            "ping": idx,
            "critical": False,
        }

    if result["state"] != st.session_state.last_state and live:
        if result["state"] == "CRITICAL":
            st.session_state.alerts += 1

        db.execute(
            """
            INSERT INTO mission_logs
            (
                timestamp, lat, lon, depth,
                classification, confidence, snr,
                uncertainty, hpi
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                lat,
                lon,
                round(-45 + 1.5 * np.sin(idx / 5), 1),
                {
                    "CRITICAL": "Mine-like candidate",
                    "REVIEW": "Human review",
                    "SAFE": "Safe seabed",
                }[result["state"]],
                float(a["p"]),
                float(S.snr_db(clean, a["denoised"])),
                float(a["unc"]),
                float(result["hpi"]),
            ),
        )

        db.commit()

    st.session_state.last_state = result["state"]

    return st.session_state.last_analysis


# ----------------------------- CURRENT ANALYSIS ------------------------------

current = ensure_live_analysis()

a = current["analysis"]
geo = current["geo"]
res = current["result"]
clean = current["clean"]
noisy = current["noisy"]

lat, lon = current["lat"], current["lon"]

snr_in = S.snr_db(clean, noisy)
snr_out = S.snr_db(clean, a["denoised"])


# -------------------------------- HERO ----------------------------------------

st.markdown("""
<div class="hero">
    <h1>PRJ-44 · AUV SONAR INTELLIGENCE</h1>
    <p>
        SAUG-HPI · Shadow-aware denoising · uncertainty-gated detection
        · persistence confirmation · mission planning · ESCC research
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Research prototype. Demonstration scenes are simulated and "
    "should not be used for real navigation."
)


# ----------------------------- MISSION OVERVIEW ------------------------------

if page == "Mission Overview":

    logs = pd.read_sql_query(
        "SELECT * FROM mission_logs ORDER BY id DESC",
        db,
    )

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Current ping", f"{current['idx']} / {max_ping}")
    m2.metric("Current status", res["state"])
    m3.metric("SNR change", f"{snr_out - snr_in:+.2f} dB")
    m4.metric("Logged events", len(logs))

    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Current side-scan ping")

        st.image(
            noisy,
            caption="Simulated noisy sonar ping",
            clamp=True,
            use_container_width=True,
        )

        st.image(
            a["denoised"],
            caption="Shadow-aware denoising output",
            clamp=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Mission indicators")

        st.metric("Contact probability", f"{a['p']:.2f}")
        st.metric("Uncertainty", f"{a['unc']:.2f}")
        st.metric("Hazard Persistence Index", f"{res['hpi']:.2f}")
        st.metric("Hazard contacts mapped", len(st.session_state.hazards))

        st.markdown(
            '<div class="note">A persistent, low-uncertainty contact '
            'can be marked CRITICAL. Ambiguous contacts are routed to REVIEW.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.subheader("System workflow")

    st.dataframe(
        pd.DataFrame([
            ["1. Sonar ping", "Generated / simulated side-scan strip"],
            ["2. Denoising", "Shadow-aware filtering via saug_core.py"],
            ["3. Uncertainty", "Perturbation-based score stability"],
            ["4. Persistence", "HPI tracking across pings"],
            ["5. Mission record", "SQLite event logging"],
            ["6. ESCC", "Experimental highlight-shadow consistency"],
        ], columns=["Stage", "Description"]),
        hide_index=True,
        use_container_width=True,
    )


# ------------------------------ LIVE TELEMETRY -------------------------------

elif page == "Live Telemetry":

    st.subheader("🔴 Live Side-Scan Telemetry")

    st.caption(
        "Use the sidebar toggle to enable/disable automatic ping "
        "advancement. Refresh the page to advance a ping."
    )

    c1, c2, c3 = st.columns(3)

    c1.image(
        noisy,
        caption=f"Raw ping #{current['idx']}",
        clamp=True,
        use_container_width=True,
    )

    c2.image(
        a["denoised"],
        caption="Denoised ping",
        clamp=True,
        use_container_width=True,
    )

    c3.image(
        a["wmap"],
        caption="Shadow-pair protection map",
        clamp=True,
        use_container_width=True,
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Input SNR", f"{snr_in:.2f} dB")
    k2.metric("Output SNR", f"{snr_out:.2f} dB")
    k3.metric("SNR change", f"{snr_out - snr_in:+.2f} dB")
    k4.metric("Uncertainty", f"{a['unc']:.2f}")

    st.subheader("Alert decision")

    if res["state"] == "CRITICAL":
        st.error(
            f"🚨 CRITICAL candidate · HPI {res['hpi']:.2f} · "
            f"{res['hits']}/{tracker.n} matching pings"
        )
    elif res["state"] == "REVIEW":
        st.warning(
            f"⚠️ Human review recommended · uncertainty {a['unc']:.2f}"
        )
    else:
        st.success("No persistent hazard alert in the current ping.")

    st.plotly_chart(
        px.imshow(
            S.pair_map(a["denoised"])[0],
            color_continuous_scale="Magma",
            title="Highlight-to-shadow pair score",
        ).update_layout(**DARK),
        use_container_width=True,
    )

    if st.button("➡️ Advance to next ping"):
        st.session_state.ping_idx = min(
            st.session_state.ping_idx + 1,
            max_ping,
        )
        st.rerun()


# ------------------------------- SAUG-HPI STUDY -------------------------------

elif page == "SAUG-HPI Study":

    st.subheader("🧪 SAUG-HPI vs baseline alerting")

    st.markdown(
        '<div class="note">The experiment uses synthetic scenes with '
        'known generated ground truth. Results are simulation measurements, '
        'not real-world validation.</div>',
        unsafe_allow_html=True,
    )

    seed = st.number_input(
        "Random seed",
        min_value=1,
        max_value=99999,
        value=42,
    )

    exp_noise = st.slider(
        "Experiment noise",
        0.1,
        0.9,
        float(noise),
        0.1,
    )

    if st.button("Run SAUG-HPI experiment"):
        with st.spinner("Running experiment..."):
            st.session_state.exp_result = S.run_experiment(
                int(seed),
                n_scenes=3,
                noise=float(exp_noise),
            )

    if "exp_result" in st.session_state:
        result = st.session_state.exp_result

        names = {
            "A": "Brightness baseline",
            "B": "Shadow-pair baseline",
            "C": "SAUG-HPI",
        }

        rows = []

        for key, v in result.items():
            positives = v["tp"] + v["fn"]
            negatives = v["fp"] + v["tn"]

            rows.append({
                "Method": names.get(key, key),
                "Detection rate (%)": round(
                    100 * v["tp"] / max(positives, 1), 2
                ),
                "False alarm rate (%)": round(
                    100 * v["fp"] / max(negatives, 1), 2
                ),
                "Precision (%)": round(
                    100 * v["tp"] / max(v["tp"] + v["fp"], 1), 2
                ),
                "TP": v["tp"],
                "FP": v["fp"],
                "FN": v["fn"],
                "TN": v["tn"],
            })

        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )

        st.plotly_chart(
            px.bar(
                df,
                x="Method",
                y=["Detection rate (%)", "False alarm rate (%)"],
                barmode="group",
                title="Measured simulation comparison",
            ).update_layout(**DARK),
            use_container_width=True,
        )

    st.subheader("Uncertainty-triggered re-look study")

    if st.button("Run re-look study"):
        with st.spinner("Running re-look study..."):
            st.session_state.relook_result = S.relook_study(
                int(seed),
                float(exp_noise),
                n_scenes=4,
            )

    if "relook_result" in st.session_state:
        rr = st.session_state.relook_result

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Pings", rr["pings"])
        c2.metric("Unsure pings", rr["unsure"])
        c3.metric("Single-look correct", rr["single_ok"])
        c4.metric("After re-look correct", rr["fused_ok"])

    st.subheader("Range-Scaling Shadow Test (RSST)")

    if st.button("Run RSST / height-invariance test"):
        with st.spinner("Running invariance experiment..."):
            st.session_state.inv_result = S.invariance_experiment(
                int(seed),
                float(exp_noise),
                amp=2,
            )

    if "inv_result" in st.session_state:
        try:
            st.dataframe(
                rsst.rsst_table(st.session_state.inv_result),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"RSST table could not be rendered: {exc}")
            st.write("Check that rsst.py contains rsst_table(trk_inv).")


# ------------------------------- ESCC NOVELTY --------------------------------

elif page == "ESCC Novelty Lab":

    st.subheader("🔬 ESCC — Echo–Shadow Counterfactual Consistency")

    st.markdown(
        '<div class="note">Research prototype: compares an estimated '
        'shadow length with a dark region measured in the uploaded image. '
        'This score is not a hazard probability or validated accuracy.</div>',
        unsafe_allow_html=True,
    )

    # ESCC runs only when this page is selected.
    try:
        E.render_page()
    except AttributeError:
        st.error(
            "Your escc_novelty.py must contain a render_page() function. "
            "Please check that the complete ESCC code was committed."
        )
    except Exception as exc:
        st.error(f"ESCC module error: {exc}")


# -------------------------------- 3D HAZARD TWIN ------------------------------

elif page == "3D Hazard Twin":

    st.subheader("🧊 3D Hazard Digital Twin")

    twin = st.session_state.twin

    if twin is None:
        st.info(
            "No candidate detected yet. Advance the live ping "
            "and revisit this page."
        )
    else:
        g = twin["geo"]

        cols = st.columns(5)

        cols[0].metric("Height", f"{g['height_m']:.2f} m")
        cols[1].metric("Width", f"{g['width_m']:.1f} m")
        cols[2].metric("Shadow", f"{g['shadow_m']:.1f} m")
        cols[3].metric("Range", f"{g['range_m']:.1f} m")
        cols[4].metric(
            "Status",
            "Confirmed" if twin["critical"] else "Candidate",
        )

        x, y, z, patch = S.reconstruct_3d(
            twin["den"],
            twin["r"],
            twin["c"],
            g["height_m"],
        )

        fig = go.Figure(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=patch,
                colorscale="Cividis",
                showscale=False,
            )
        )

        fig.update_layout(
            height=550,
            scene=dict(
                xaxis_title="Range (m)",
                yaxis_title="Along-track (m)",
                zaxis_title="Height (m)",
            ),
            **DARK,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Pseudo-3D reconstruction from simulated sonar shadow geometry; "
            "not measured bathymetry."
        )

    with st.expander("Height validation experiment"):

        if st.button("Run height validation"):
            with st.spinner("Running validation..."):
                st.session_state.height_result = S.height_validation(
                    int(st.session_state.seed),
                    float(noise),
                    trials=8,
                )

        if "height_result" in st.session_state:
            st.dataframe(
                pd.DataFrame(st.session_state.height_result),
                hide_index=True,
                use_container_width=True,
            )


# ------------------------------- DENOISER BENCHMARKS --------------------------

elif page == "Denoiser Benchmarks":

    st.subheader("📊 Denoiser benchmark")

    seed = st.number_input(
        "Benchmark seed",
        min_value=1,
        max_value=99999,
        value=12,
    )

    bench_noise = st.slider(
        "Benchmark noise",
        0.1,
        0.9,
        0.35,
        0.05,
    )

    if st.button("Run denoiser benchmark"):
        with st.spinner("Comparing filters..."):
            st.session_state.bench = S.denoise_benchmark(
                int(seed),
                n=10,
                noise=float(bench_noise),
            )

    if "bench" in st.session_state:
        df = pd.DataFrame([
            {
                "Method": key,
                "PSNR (dB)": round(value[0], 2),
                "SNR (dB)": round(value[1], 2),
                "MSE": round(value[2], 4),
            }
            for key, value in st.session_state.bench.items()
        ])

        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
        )

        st.plotly_chart(
            px.bar(
                df,
                x="Method",
                y="PSNR (dB)",
                title="PSNR comparison",
            ).update_layout(**DARK),
            use_container_width=True,
        )


# -------------------------------- ACOUSTIC FFT -------------------------------

elif page == "Acoustic FFT":

    st.subheader("🧠 Acoustic FFT Spectrum")

    profile = noisy.mean(axis=0)
    profile = profile - profile.mean()

    spectrum = np.abs(np.fft.rfft(profile))
    freq = np.fft.rfftfreq(len(profile), d=1.0)

    df = pd.DataFrame({
        "Spatial frequency": freq,
        "Magnitude": spectrum,
    })

    st.plotly_chart(
        px.line(
            df,
            x="Spatial frequency",
            y="Magnitude",
            title="FFT of current sonar ping",
        ).update_layout(**DARK),
        use_container_width=True,
    )

    st.caption("Spectrum is calculated from the current simulated ping.")


# ------------------------------- GIS MISSION TRACK ----------------------------

elif page == "GIS Mission Track":

    st.subheader("🗺️ Mission track and hazard zones")

    track = st.session_state.track
    hazards = st.session_state.hazards

    if track:
        df = pd.DataFrame(
            track,
            columns=["lat", "lon", "state"],
        )

        color_map = {
            "SAFE": [0, 212, 255, 220],
            "REVIEW": [255, 183, 3, 230],
            "CRITICAL": [255, 0, 84, 255],
        }

        df["color"] = df["state"].map(color_map)

        layers = [
            pdk.Layer(
                "PathLayer",
                data=pd.DataFrame({
                    "path": [df[["lon", "lat"]].values.tolist()]
                }),
                get_path="path",
                get_color=[0, 180, 216],
                width_min_pixels=3,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position="[lon,lat]",
                get_fill_color="color",
                get_radius=5,
                pickable=True,
            ),
        ]

        if hazards:
            hdf = pd.DataFrame({
                "lat": [
                    st.session_state.auv_lat0 + h["n"] * DEG_PER_M
                    for h in hazards
                ],
                "lon": [
                    st.session_state.auv_lon0
                    + h["e"] / (
                        111000 * np.cos(
                            np.radians(st.session_state.auv_lat0)
                        )
                    )
                    for h in hazards
                ],
            })

            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=hdf,
                    get_position="[lon,lat]",
                    get_radius=standoff,
                    get_fill_color=[255, 0, 84, 75],
                    get_line_color=[255, 0, 84, 240],
                    stroked=True,
                    line_width_min_pixels=2,
                )
            )

        st.pydeck_chart(
            pdk.Deck(
                map_style="dark",
                initial_view_state=pdk.ViewState(
                    latitude=float(df.lat.iloc[-1]),
                    longitude=float(df.lon.iloc[-1]),
                    zoom=14,
                    pitch=25,
                ),
                layers=layers,
            )
        )

        st.caption(
            "Cyan = simulated AUV track · Red = estimated "
            "candidate hazard/standoff zone."
        )

    else:
        st.info(
            "Advance the live telemetry ping to start building the mission track."
        )

    if hazards:
        st.dataframe(
            pd.DataFrame(hazards),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No persistent hazard has been mapped yet.")


# ------------------------------ RISK & ENERGY PLANNER -------------------------

elif page == "Risk & Energy Planner":

    st.subheader("⚡ Risk-and-Energy-Aware Rescan Planner")

    st.markdown(
        '<div class="note">This is a planning demonstration. Contact risks '
        'and energy values are illustrative and must not be used for real '
        'AUV operations.</div>',
        unsafe_allow_html=True,
    )

    count = st.slider("Number of candidate contacts", 3, 15, 8)
    budget = st.slider("Available scan-energy budget (units)", 10, 200, 80, 5)

    seed = st.number_input(
        "Planner seed",
        min_value=1,
        max_value=99999,
        value=44,
    )

    rng = np.random.default_rng(int(seed))

    df = pd.DataFrame({
        "Contact ID": [f"C-{i+1:02d}" for i in range(count)],
        "Risk (%)": np.round(rng.uniform(15, 99, count), 1),
        "Uncertainty (%)": np.round(rng.uniform(5, 80, count), 1),
        "Scan Energy": rng.integers(5, 31, count),
    })

    df["Priority"] = (
        df["Risk (%)"] * 0.58
        + df["Uncertainty (%)"] * 0.22
        + (30 / df["Scan Energy"].clip(lower=1)) * 20
    ).round(2)

    df = df.sort_values(
        "Priority",
        ascending=False,
    ).reset_index(drop=True)

    used = 0
    selected = []

    for _, row in df.iterrows():
        energy = int(row["Scan Energy"])

        if used + energy <= budget:
            selected.append(row["Contact ID"])
            used += energy

    df["Recommended Rescan"] = df["Contact ID"].isin(selected)

    c1, c2, c3 = st.columns(3)

    c1.metric("Candidates", count)
    c2.metric("Contacts selected", len(selected))
    c3.metric("Energy used", f"{used}/{budget}")

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
    )

    st.plotly_chart(
        px.bar(
            df,
            x="Contact ID",
            y="Priority",
            color="Recommended Rescan",
            title="Rescan priority by contact",
        ).update_layout(**DARK),
        use_container_width=True,
    )

    st.download_button(
        "Download planner CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="risk_energy_plan.csv",
        mime="text/csv",
    )


# ------------------------------- MISSION DATABASE -----------------------------

elif page == "Mission Database":

    st.subheader("💾 SQLite Mission Database")

    logs = pd.read_sql_query(
        "SELECT * FROM mission_logs ORDER BY id DESC LIMIT 500",
        db,
    )

    if logs.empty:
        st.info(
            "No mission events have been logged yet. Let telemetry run "
            "and change alert states."
        )
    else:
        st.metric("Saved events", len(logs))

        st.dataframe(
            logs,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Export mission history CSV",
            logs.to_csv(index=False).encode("utf-8"),
            file_name="sonar_mission_logs.csv",
            mime="text/csv",
        )

    st.caption(f"Database file: {DB_PATH.resolve()}")


# -------------------------------- FOOTER --------------------------------------

st.divider()

st.caption(
    "PRJ-44 Research Prototype · SAUG-HPI / RSST / ESCC modules · "
    "Simulated examples only"
)
