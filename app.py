```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path

# ==========================================
# PRJ-44 AUV SONAR HAZARD CLASSIFICATION
# ==========================================

st.set_page_config(
    page_title="PRJ-44 | Sonar Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #07111f, #0b1e32);
    color: #e8f1ff;
}
[data-testid="stSidebar"] {
    background-color: #0a1728;
}
h1, h2, h3 {
    color: #63d9ff !important;
}
div[data-testid="stMetric"] {
    background: #102940;
    border: 1px solid #214b68;
    padding: 16px;
    border-radius: 12px;
}
.stButton button {
    background: #087ea4;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}
.stButton button:hover {
    background: #10a6d4;
    color: white;
}
.panel {
    background: #102940;
    border: 1px solid #214b68;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 12px;
}
.small-note {
    color: #a9bfd3;
    font-size: 13px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATABASE
# ==========================================

DB_PATH = Path(__file__).parent / "sonar_logs.db"


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_time TEXT,
                hazard TEXT,
                confidence REAL,
                risk REAL,
                noise REAL,
                snr REAL
            )
        """)


def save_scan(hazard, confidence, risk, noise, snr):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO scans
            (scan_time, hazard, confidence, risk, noise, snr)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            hazard,
            float(confidence),
            float(risk),
            float(noise),
            float(snr)
        ))


def load_scans():
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM scans ORDER BY id DESC",
            conn
        )


init_db()


# ==========================================
# SONAR DEMO PROCESSING
# ==========================================

def make_sonar_scene(seed, rows=180, cols=260):
    rng = np.random.default_rng(seed)

    x = np.linspace(0, 8 * np.pi, cols)
    y = np.linspace(0, 5 * np.pi, rows)

    xx, yy = np.meshgrid(x, y)

    seabed = (
        0.20 * np.sin(xx)
        + 0.15 * np.cos(yy)
        + 0.12 * np.sin(xx * 0.5 + yy)
    )

    image = seabed + rng.normal(0, 0.10, (rows, cols))

    # Simulated sonar returns
    objects = [
        (45, 65, 12, 0.9),
        (100, 150, 18, 1.1),
        (140, 215, 10, 0.7)
    ]

    for cy, cx, radius, strength in objects:
        yy2, xx2 = np.ogrid[:rows, :cols]
        mask = (yy2 - cy) ** 2 + (xx2 - cx) ** 2 <= radius ** 2
        image[mask] += strength

        # Simulated acoustic shadow
        shadow_start = min(cols, cx + radius)
        shadow_end = min(cols, shadow_start + radius * 3)
        if shadow_start < shadow_end:
            image[max(0, cy-radius):min(rows, cy+radius),
                  shadow_start:shadow_end] -= 0.28

    image = (image - image.min()) / (
        image.max() - image.min() + 1e-8
    )

    return image


def add_acoustic_noise(image, noise_level, seed):
    rng = np.random.default_rng(seed + 1000)

    gaussian = rng.normal(
        0, noise_level, image.shape
    )

    speckle = image * rng.normal(
        0, noise_level * 0.65, image.shape
    )

    noisy = image + gaussian + speckle
    return np.clip(noisy, 0, 1)


def denoise_image(image):
    # Lightweight local averaging for a runnable demo.
    # This is NOT a trained deconvolutional CNN.
    padded = np.pad(image, 1, mode="reflect")
    result = np.zeros_like(image)

    for dy in range(3):
        for dx in range(3):
            result += padded[
                dy:dy + image.shape[0],
                dx:dx + image.shape[1]
            ]

    return result / 9.0


def estimate_snr(reference, observed):
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean((reference - observed) ** 2) + 1e-10
    return float(10 * np.log10(signal_power / noise_power))


def detect_hazards(image, threshold=0.70):
    # Demo threshold-based candidate detection.
    # Replace with the project's trained classifier for real predictions.
    mask = image > threshold
    count = int(mask.sum())

    if count > 1500:
        label = "HIGH RETURN / POSSIBLE HAZARD"
    elif count > 400:
        label = "OBJECT CANDIDATE"
    else:
        label = "LOW RETURN / REVIEW"

    strength = float(np.clip(image.mean() * 100, 0, 100))
    return label, strength, mask


def calculate_metrics(clean, noisy, denoised):
    snr_before = estimate_snr(clean, noisy)
    snr_after = estimate_snr(clean, denoised)

    mse_before = float(np.mean((clean - noisy) ** 2))
    mse_after = float(np.mean((clean - denoised) ** 2))

    return {
        "snr_before": snr_before,
        "snr_after": snr_after,
        "snr_improvement": snr_after - snr_before,
        "mse_before": mse_before,
        "mse_after": mse_after
    }


def make_3d_surface(image):
    step = 5
    small = image[::step, ::step]

    yy, xx = np.mgrid[
        0:small.shape[0],
        0:small.shape[1]
    ]

    return xx, yy, small


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌊 PRJ-44")
st.sidebar.caption("AUV SONAR INTELLIGENCE SYSTEM")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Overview",
        "Live Sonar Analysis",
        "Denoising Laboratory",
        "3D Seabed View",
        "Experiment Benchmark",
        "Mission Planner",
        "Scan History",
        "Project Information"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### Scan Configuration")

seed = st.sidebar.number_input(
    "Scenario seed",
    min_value=1,
    max_value=999999,
    value=44,
    step=1
)

noise_level = st.sidebar.slider(
    "Acoustic noise level",
    min_value=0.01,
    max_value=0.40,
    value=0.15,
    step=0.01
)

st.sidebar.markdown(
    '<p class="small-note">Educational prototype using simulated sonar data.</p>',
    unsafe_allow_html=True
)


# ==========================================
# SHARED SCENE
# ==========================================

clean = make_sonar_scene(int(seed))
noisy = add_acoustic_noise(
    clean, float(noise_level), int(seed)
)
denoised = denoise_image(noisy)

hazard_label, confidence, hazard_mask = detect_hazards(
    denoised
)

risk_score = float(np.clip(
    confidence * 0.65
    + noise_level * 100 * 0.35,
    0,
    100
))

metrics = calculate_metrics(clean, noisy, denoised)


# ==========================================
# REUSABLE CHART
# ==========================================

def show_image(image, title, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.imshow(image, cmap=cmap, aspect="auto")
    ax.set_title(title, color="white")
    ax.set_xlabel("Range / pixels", color="white")
    ax.set_ylabel("Ping / pixels", color="white")
    ax.tick_params(colors="white")
    fig.patch.set_facecolor("#102940")
    ax.set_facecolor("#102940")
    st.pyplot(fig)
    plt.close(fig)


def metric_card(title, value, help_text=None):
    st.metric(title, value, help=help_text)


# ==========================================
# PAGE 1: OVERVIEW
# ==========================================

if page == "Mission Overview":

    st.title("🌊 AUV Sonar Intelligence Dashboard")
    st.write(
        "Side-scan sonar hazard analysis, acoustic denoising "
        "and mission-risk monitoring."
    )

    st.info(
        "DEMO MODE — Current data is simulated. "
        "Predictions and metrics are not validated real-world model results."
    )

    a, b, c, d = st.columns(4)

    a.metric("Hazard Candidate", hazard_label)
    b.metric("Return Intensity", f"{confidence:.1f}%")
    c.metric("Mission Risk Index", f"{risk_score:.1f}/100")
    d.metric("SNR Improvement", f"{metrics['snr_improvement']:.2f} dB")

    st.divider()

    left, right = st.columns(2)

    with left:
        show_image(noisy, "Raw Noisy Sonar", "gray")

    with right:
        show_image(denoised, "Denoised Sonar", "gray")

    st.subheader("System Modules")

    modules = [
        ("01", "Acoustic Input", "Sonar scene and noise simulation"),
        ("02", "Feature Denoising", "Lightweight image smoothing"),
        ("03", "Hazard Screening", "Threshold-based object candidate screening"),
        ("04", "Risk Monitoring", "Demonstration risk index"),
        ("05", "Experiment Tracking", "SQLite scan history"),
        ("06", "Mission Planning", "Risk-aware waypoint simulation")
    ]

    cols = st.columns(3)

    for i, item in enumerate(modules):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="panel">
                    <h3>{item[0]} · {item[1]}</h3>
                    <p>{item[2]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==========================================
# PAGE 2: LIVE SONAR ANALYSIS
# ==========================================

elif page == "Live Sonar Analysis":

    st.title("🔎 Sonar Scan Analysis")

    st.write(
        "Inspect a simulated sonar scene, its denoised output "
        "and the detected candidate regions."
    )

    uploaded = st.file_uploader(
        "Optional: upload a sonar image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
    )

    if uploaded is not None:
        from PIL import Image

        input_image = Image.open(uploaded).convert("L")
        input_image = input_image.resize((260, 180))
        source = np.asarray(input_image, dtype=np.float32) / 255.0

        noisy_live = source
        denoised_live = denoise_image(source)

        label_live, confidence_live, mask_live = detect_hazards(
            denoised_live
        )

        metrics_live = calculate_metrics(
            source, noisy_live, denoised_live
        )

        st.caption(
            "For uploaded images, the uploaded image is treated as the input reference. "
            "SNR values are not meaningful without a known clean reference."
        )
    else:
        source = clean
        noisy_live = noisy
        denoised_live = denoised
        label_live = hazard_label
        confidence_live = confidence
        mask_live = hazard_mask
        metrics_live = metrics

    c1, c2, c3 = st.columns(3)

    c1.metric("Screening Result", label_live)
    c2.metric("Mean Return Index", f"{confidence_live:.2f}%")
    c3.metric("Noise Level", f"{noise_level:.2f}")

    col1, col2 = st.columns(2)

    with col1:
        show_image(noisy_live, "Input Sonar", "gray")

    with col2:
        show_image(denoised_live, "Processed Sonar", "gray")

    st.subheader("Candidate Region Mask")
    show_image(mask_live.astype(float), "Threshold Candidate Mask", "inferno")

    if st.button("Save Current Scan to History"):
        save_scan(
            label_live,
            confidence_live,
            risk_score,
            noise_level,
            metrics_live["snr_after"]
        )
        st.success("Scan record saved to local SQLite history.")


# ==========================================
# PAGE 3: DENOISING LAB
# ==========================================

elif page == "Denoising Laboratory":

    st.title("🧪 Acoustic Denoising Laboratory")

    st.write(
        "Compare a simulated clean scene, noisy input and a "
        "lightweight smoothing result."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        show_image(clean, "Reference Scene", "gray")

    with col2:
        show_image(noisy, "Noisy Input", "gray")

    with col3:
        show_image(denoised, "Smoothed Output", "gray")

    st.subheader("Signal Quality Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Input SNR",
        f"{metrics['snr_before']:.2f} dB"
    )

    c2.metric(
        "Output SNR",
        f"{metrics['snr_after']:.2f} dB"
    )

    c3.metric(
        "SNR Change",
        f"{metrics['snr_improvement']:.2f} dB"
    )

    st.subheader("Mean Squared Error")

    error_df = pd.DataFrame({
        "Stage": ["Noisy Input", "Smoothed Output"],
        "MSE": [
            metrics["mse_before"],
            metrics["mse_after"]
        ]
    })

    st.bar_chart(error_df.set_index("Stage"))

    st.warning(
        "The current denoiser is a 3×3 averaging filter, not a "
        "deconvolutional CNN. Do not report these demo metrics as "
        "trained-model performance."
    )


# ==========================================
# PAGE 4: 3D VIEW
# ==========================================

elif page == "3D Seabed View":

    st.title("🗺️ 3D Seabed Intensity View")

    st.write(
        "A 3D surface representation of simulated sonar intensity. "
        "Height represents intensity, not measured physical seabed depth."
    )

    xx, yy, zz = make_3d_surface(denoised)

    fig = plt.figure(figsize=(11, 6))
    ax = fig.add_subplot(111, projection="3d")

    surface = ax.plot_surface(
        xx, yy, zz,
        cmap="viridis",
        linewidth=0,
        antialiased=True
    )

    ax.set_title("Simulated Sonar Intensity Surface", color="white")
    ax.set_xlabel("Range", color="white")
    ax.set_ylabel("Ping", color="white")
    ax.set_zlabel("Intensity", color="white")

    fig.colorbar(surface, shrink=0.5, aspect=10)
    fig.patch.set_facecolor("#102940")
    ax.set_facecolor("#102940")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("High-Return Candidate Locations")

    locations = np.argwhere(hazard_mask)

    if len(locations) > 0:
        sample = locations[::max(1, len(locations) // 100)]

        location_df = pd.DataFrame({
            "Ping / Row": sample[:, 0],
            "Range / Column": sample[:, 1],
            "Intensity": denoised[
                sample[:, 0], sample[:, 1]
            ]
        })

        st.dataframe(
            location_df.head(100),
            use_container_width=True
        )
    else:
        st.info("No pixels crossed the selected threshold.")


# ==========================================
# PAGE 5: BENCHMARK
# ==========================================

elif page == "Experiment Benchmark":

    st.title("📊 Experiment Benchmark")

    st.write(
        "Run repeatable simulated experiments at different "
        "noise levels and compare signal-quality measurements."
    )

    experiment_count = st.slider(
        "Number of scenes",
        min_value=3,
        max_value=30,
        value=10
    )

    if st.button("Run Benchmark Experiment"):
        rows = []
        progress = st.progress(0)

        for i in range(experiment_count):
            scene = make_sonar_scene(int(seed) + i)
            noisy_scene = add_acoustic_noise(
                scene,
                float(noise_level),
                int(seed) + i
            )
            output = denoise_image(noisy_scene)

            m = calculate_metrics(scene, noisy_scene, output)
            label, conf, mask = detect_hazards(output)

            rows.append({
                "Scene": i + 1,
                "Hazard Screening": label,
                "Return Index (%)": conf,
                "Input SNR (dB)": m["snr_before"],
                "Output SNR (dB)": m["snr_after"],
                "SNR Change (dB)": m["snr_improvement"],
                "Input MSE": m["mse_before"],
                "Output MSE": m["mse_after"]
            })

            progress.progress((i + 1) / experiment_count)

        results = pd.DataFrame(rows)

        st.session_state["benchmark_results"] = results

        st.success("Simulated benchmark completed.")

    if "benchmark_results" in st.session_state:
        results = st.session_state["benchmark_results"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mean Input SNR",
            f"{results['Input SNR (dB)'].mean():.2f} dB"
        )

        c2.metric(
            "Mean Output SNR",
            f"{results['Output SNR (dB)'].mean():.2f} dB"
        )

        c3.metric(
            "Mean SNR Change",
            f"{results['SNR Change (dB)'].mean():.2f} dB"
        )

        st.dataframe(results, use_container_width=True)

        st.subheader("SNR Comparison")

        chart_df = results[
            ["Scene", "Input SNR (dB)", "Output SNR (dB)"]
        ].set_index("Scene")

        st.line_chart(chart_df)

        csv = results.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Benchmark CSV",
            data=csv,
            file_name="prj44_benchmark.csv",
            mime="text/csv"
        )

        st.download_button(
            "Download Benchmark JSON",
            data=results.to_json(
                orient="records", indent=2
            ),
            file_name="prj44_benchmark.json",
            mime="application/json"
        )

        st.caption(
            "Accuracy, precision, recall and F1 are not calculated "
            "because this demo has no ground-truth hazard labels."
        )


# ==========================================
# PAGE 6: MISSION PLANNER
# ==========================================

elif page == "Mission Planner":

    st.title("🧭 Risk-Aware Mission Planner")

    st.write(
        "Create a simulated AUV route and inspect its waypoint "
        "risk values. Coordinates are illustrative, not GPS positions."
    )

    number_waypoints = st.slider(
        "Number of waypoints",
        min_value=5,
        max_value=30,
        value=12
    )

    route_seed = st.number_input(
        "Route seed",
        min_value=1,
        max_value=999999,
        value=44,
        key="route_seed"
    )

    rng = np.random.default_rng(int(route_seed))

    x = np.cumsum(rng.uniform(0.7, 1.3, number_waypoints))
    y = np.cumsum(rng.uniform(-0.5, 0.5, number_waypoints))

    waypoint_risks = rng.uniform(5, 95, number_waypoints)

    route_df = pd.DataFrame({
        "Waypoint": np.arange(1, number_waypoints + 1),
        "Route X (relative)": x.round(2),
        "Route Y (relative)": y.round(2),
        "Simulated Risk": waypoint_risks.round(1),
        "Suggested Action": [
            "Review" if r >= 65 else "Continue monitoring"
            for r in waypoint_risks
        ]
    })

    fig, ax = plt.subplots(figsize=(10, 5))

    points = ax.scatter(
        x, y,
        c=waypoint_risks,
        s=100,
        cmap="RdYlGn_r"
    )

    ax.plot(x, y, linestyle="--", alpha=0.7)

    for i in range(number_waypoints):
        ax.annotate(str(i + 1), (x[i], y[i]))

    fig.colorbar(points, ax=ax, label="Simulated Risk")
    ax.set_title("Illustrative AUV Route")
    ax.set_xlabel("Relative X")
    ax.set_ylabel("Relative Y")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(route_df, use_container_width=True)

    st.download_button(
        "Export Route CSV",
        data=route_df.to_csv(index=False).encode("utf-8"),
        file_name="prj44_mission_route.csv",
        mime="text/csv"
    )

    st.warning(
        "This is a synthetic route demonstration. It does not use "
        "real bathymetry, GPS, vehicle dynamics, or validated collision avoidance."
    )


# ==========================================
# PAGE 7: SCAN HISTORY
# ==========================================

elif page == "Scan History":

    st.title("🗄️ Scan History & Audit Log")

    history = load_scans()

    if history.empty:
        st.info(
            "No saved scans yet. Open Live Sonar Analysis and "
            "click 'Save Current Scan to History'."
        )
    else:
        st.metric("Stored Scan Records", len(history))

        st.dataframe(
            history,
            use_container_width=True
        )

        st.download_button(
            "Export History CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prj44_scan_history.csv",
            mime="text/csv"
        )

        if st.button("Delete All Local Scan History"):
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("DELETE FROM scans")

            st.success("Local scan history cleared.")
            st.rerun()


# ==========================================
# PAGE 8: PROJECT INFORMATION
# ==========================================

elif page == "Project Information":

    st.title("ℹ️ Project Information")

    st.markdown("""
    ### PRJ-44: AUV Side-Scan Sonar Hazard Classification Engine

    **Project objective**

    Demonstrate an interface for inspecting sonar images, reducing
    image noise, screening candidate regions and recording experiments.

    ### Proposed research workflow

    1. Acquire side-scan sonar images.
    2. Apply acoustic image preprocessing.
    3. Reconstruct or denoise distorted sonar imagery.
    4. Classify hazards using a trained model.
    5. Compare model outputs with ground-truth labels.
    6. Evaluate classification and signal-quality metrics.
    7. Store scan results and review mission risk.

    ### Current prototype implementation

    - Simulated sonar scene generation.
    - Adjustable synthetic noise.
    - 3×3 local averaging filter.
    - Intensity-threshold candidate screening.
    - SNR and MSE measurements on simulated reference images.
    - 3D intensity surface visualization.
    - Simulated route and waypoint-risk display.
    - SQLite scan history.
    - CSV and JSON experiment exports.

    ### Important limitations

    This version does **not** implement a trained deconvolutional CNN,
    validated MILCO/NOMBO classification, real GPS/GIS mapping,
    or measured underwater depth reconstruction.

    Classification accuracy, precision, recall, F1-score and confusion
    matrix require labelled test data and actual predictions.
    Do not present synthetic demonstration results as real-world accuracy.
    """)

    st.subheader("Environment")

    st.code("""
Python
Streamlit
NumPy
Pandas
Matplotlib
SQLite
Pillow (for uploaded images)
    """)

    st.subheader("Run Locally")

    st.code("""
pip install streamlit numpy pandas matplotlib pillow
streamlit run app.py
    """)


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#91a9bd;padding:8px">
        PRJ-44 · AUV Sonar Intelligence Prototype · Educational Demo
    </div>
    """,
    unsafe_allow_html=True
)
```
