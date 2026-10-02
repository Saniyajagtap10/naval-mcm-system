
"""
PRJ-44: AUV Side-Scan Sonar Hazard Classification Engine
SAUG-HPI | Shadow-Aware Denoising | ESCC | Mission Analytics

Run:
    streamlit run app.py

Note:
    Live Mission uses generated demonstration data.
    Upload a real sonar image for image-based analysis.
"""

import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="PRJ-44 | SAUG-HPI Sonar Engine",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "SAUG-HPI"
DB_PATH = Path("mission_history.db")


# ============================================================
# DATABASE
# ============================================================
def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source TEXT,
                hpi REAL,
                status TEXT,
                snr REAL,
                image_name TEXT
            )
        """)


def save_mission(source, hpi, status, snr, image_name):
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """INSERT INTO missions
               (timestamp, source, hpi, status, snr, image_name)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                time.strftime("%Y-%m-%d %H:%M:%S"),
                source,
                float(hpi),
                status,
                float(snr),
                image_name,
            ),
        )


def load_missions():
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT * FROM missions ORDER BY id DESC", con
        )


init_db()


# ============================================================
# SONAR IMAGE UTILITIES
# ============================================================
def to_gray_array(image):
    """Convert PIL image into normalized grayscale float image."""
    image = image.convert("RGB")
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (200, 160), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def make_demo_scene(seed=42, hazards=3, clutter=5):
    """Generate a synthetic sonar scene for demonstration only."""
    rng = np.random.default_rng(seed)
    h, w = 160, 200

    x = np.linspace(0, 1, w)[None, :]
    y = np.arange(h)[:, None]
    base = 0.46 + 0.07 * np.sin(2 * np.pi * (x * 5 + y / 75))
    base = np.broadcast_to(base, (h, w)).copy()
    base *= (1 - 0.13 * x)

    yy, xx = np.ogrid[:h, :w]
    ground_truth = []

    for _ in range(hazards):
        r = int(rng.integers(20, h - 25))
        c = int(rng.integers(20, w - 45))
        radius = int(rng.integers(5, 9))

        base[(yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2] += 0.34

        shadow_end = min(w, c + radius + int(radius * 3))
        shadow_start = min(w, c + radius)
        if shadow_start < shadow_end:
            base[
                max(0, r - radius):min(h, r + radius + 1),
                shadow_start:shadow_end,
            ] *= 0.24

        ground_truth.append((r, c))

    for i in range(clutter):
        r = int(rng.integers(12, h - 12))
        c = int(rng.integers(15, w - 15))
        radius = int(rng.integers(3, 7))

        if i % 2 == 0:
            base[(yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2] += 0.25
        else:
            base[
                max(0, r - radius):min(h, r + radius + 1),
                c:min(w, c + radius * 2),
            ] *= 0.55

    speckle = rng.gamma(4.0, 0.25, (h, w))
    noisy = np.clip(base * speckle + rng.normal(0, 0.045, (h, w)), 0, 1)

    return (
        np.clip(base, 0, 1).astype(np.float32),
        noisy.astype(np.float32),
        ground_truth,
    )


# ============================================================
# SAUG-HPI CORE
# ============================================================
def highlight_shadow_map(image):
    """
    Estimate bright highlights and down-range acoustic shadows.
    This is a computer-vision heuristic, not a trained detector.
    """
    image = np.clip(image, 0, 1).astype(np.float32)
    smooth = cv2.GaussianBlur(image, (5, 5), 0)

    u8 = (smooth * 255).astype(np.uint8)
    kernel = 31 if min(u8.shape) >= 31 else 3
    background = cv2.medianBlur(u8, kernel).astype(np.float32) / 255.0

    highlight = np.maximum(smooth - background, 0)
    dark_region = np.maximum(background - smooth, 0)

    highlight = cv2.blur(highlight, (9, 9))
    dark_region = cv2.blur(dark_region, (13, 9))

    # Side-scan shadow generally appears down-range of an object.
    shift = 13
    shifted_shadow = np.roll(dark_region, -shift, axis=1)
    shifted_shadow[:, -shift:] = 0

    pair = np.sqrt(highlight * shifted_shadow)
    pair[:5, :] = 0
    pair[-5:, :] = 0
    pair[:, :5] = 0

    return pair, highlight, shifted_shadow


def shadow_aware_denoise(image):
    """Use lighter smoothing near candidate structures."""
    image = np.clip(image, 0, 1).astype(np.float32)
    pair, _, _ = highlight_shadow_map(image)

    light = cv2.GaussianBlur(image, (0, 0), 0.8)
    heavy = cv2.GaussianBlur(image, (0, 0), 2.0)

    weight = np.clip(pair / 0.10, 0, 1)
    weight = cv2.GaussianBlur(weight, (0, 0), 5)
    weight = np.clip(weight * 2.5, 0, 1)

    denoised = weight * light + (1 - weight) * heavy
    return np.clip(denoised, 0, 1), weight


def calculate_snr(reference, test):
    """SNR relative to a reference image, when one is available."""
    signal_power = np.mean(np.square(reference))
    error_power = np.mean(np.square(reference - test)) + 1e-10
    return float(10 * np.log10((signal_power + 1e-10) / error_power))


def calculate_psnr(reference, test):
    mse = float(np.mean((reference - test) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def analyse_sonar(image, threshold=0.10):
    denoised, weight = shadow_aware_denoise(image)
    pair, highlight, shadow = highlight_shadow_map(denoised)

    peak = float(pair.max())
    location = np.unravel_index(np.argmax(pair), pair.shape)

    # Relative heuristic score. It is NOT a calibrated probability.
    hpi = float(np.clip(peak / max(threshold * 2.0, 1e-6), 0, 1) * 100)

    if peak >= threshold * 1.25:
        status = "POTENTIAL HAZARD"
    elif peak >= threshold * 0.70:
        status = "REVIEW REQUIRED"
    else:
        status = "NO STRONG PAIR DETECTED"

    return {
        "denoised": denoised,
        "weight": weight,
        "pair": pair,
        "highlight": highlight,
        "shadow": shadow,
        "peak": peak,
        "location": location,
        "hpi": hpi,
        "status": status,
    }


# ============================================================
# DISPLAY HELPERS
# ============================================================
def show_image(title, image, cmap="gray", marker=None):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
    if marker is not None:
        r, c = marker
        ax.scatter([c], [r], s=100, facecolors="none", edgecolors="red")
    ax.set_title(title)
    ax.set_xlabel("Range bins")
    ax.set_ylabel("Along-track rows")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def render_analysis(image, source, image_name, reference=None):
    result = analyse_sonar(image)
    hpi = result["hpi"]
    status = result["status"]

    st.subheader("Hazard assessment")
    c1, c2, c3 = st.columns(3)
    c1.metric("SAUG-HPI heuristic score", f"{hpi:.1f}/100")
    c2.metric("Peak highlight-shadow strength", f"{result['peak']:.4f}")
    c3.metric("Assessment", status)

    st.caption(
        "HPI is a relative image-analysis score, not a validated hazard "
        "probability. Results require human review."
    )

    tab1, tab2, tab3 = st.tabs([
        "Sonar & denoising",
        "Feature evidence",
        "Technical metrics",
    ])

    with tab1:
        left, right = st.columns(2)
        with left:
            show_image("Input sonar image", image)
        with right:
            show_image("Shadow-aware denoising", result["denoised"])

        if reference is not None:
            m1, m2 = st.columns(2)
            m1.metric(
                "Input PSNR",
                f"{calculate_psnr(reference, image):.2f} dB",
            )
            m2.metric(
                "Denoised PSNR",
                f"{calculate_psnr(reference, result['denoised']):.2f} dB",
            )
            st.caption(
                "PSNR is shown because a clean synthetic reference is available."
            )
        else:
            st.info(
                "No clean reference image is available for this upload. "
                "Reference-based PSNR/SNR cannot be measured honestly."
            )

    with tab2:
        left, right = st.columns(2)
        with left:
            show_image("Highlight-shadow pair map", result["pair"])
        with right:
            show_image(
                "Candidate location overlay",
                image,
                marker=result["location"],
            )

        r, c = result["location"]
        st.write(f"**Candidate pixel:** row {r}, range bin {c}")
        st.write(f"**Highlight response:** {result['highlight'].max():.4f}")
        st.write(f"**Shadow response:** {result['shadow'].max():.4f}")
        st.write(
            "The candidate is the strongest pixel in the pair map. "
            "It is not a confirmed mine or ordnance classification."
        )

    with tab3:
        metrics = {
            "Input mean intensity": float(np.mean(image)),
            "Denoised mean intensity": float(np.mean(result["denoised"])),
            "Input standard deviation": float(np.std(image)),
            "Denoised standard deviation": float(np.std(result["denoised"])),
            "Maximum pair strength": result["peak"],
            "HPI heuristic score": hpi,
        }
        st.dataframe(
            pd.DataFrame(
                [{"Metric": k, "Measured value": v} for k, v in metrics.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("Save this assessment to mission history", key=f"save_{source}_{image_name}"):
        save_mission(source, hpi, status, np.nan, image_name)
        st.success("Assessment saved to local mission history.")


# ============================================================
# VISUAL STYLE
# ============================================================
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at top left, #102a43 0%, transparent 40%),
        linear-gradient(140deg, #071521 0%, #0b1c2a 55%, #102334 100%);
}
[data-testid="stSidebar"] {
    background: #081522;
    border-right: 1px solid #23465c;
}
h1, h2, h3 {
    color: #eaf7ff !important;
}
p, label, .stMarkdown {
    color: #c8dbe8;
}
div[data-testid="stMetric"] {
    background: rgba(18, 43, 62, 0.78);
    border: 1px solid #244b63;
    border-radius: 12px;
    padding: 14px;
}
.stButton > button {
    border-radius: 9px;
    border: 1px solid #28b8c7;
    background: #0e5265;
    color: white;
    font-weight: 600;
}
.stButton > button:hover {
    border-color: #76e5ee;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🌊 PRJ-44 CONTROL")
st.sidebar.caption("AUV Side-Scan Sonar Research Console")

page = st.sidebar.radio(
    "Navigation",
    [
        "Mission Dashboard",
        "Live Mission Demo",
        "Sonar Image Analysis",
        "SAUG-HPI Novelty Lab",
        "Mission History",
        "About the Project",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**Analysis settings**")
seed = st.sidebar.slider("Demo scene seed", 1, 999, 42)
hazard_count = st.sidebar.slider("Demo hazard objects", 1, 6, 3)
clutter_count = st.sidebar.slider("Demo clutter objects", 0, 10, 5)

st.sidebar.caption(
    "Research prototype • Synthetic live scenes • "
    "Image analysis is not operational mine clearance advice"
)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div style="
        padding: 22px 24px;
        border: 1px solid #28536a;
        border-radius: 16px;
        background: linear-gradient(120deg, #102b40, #102334);
        margin-bottom: 18px;">
        <div style="color:#5fe1e8;font-size:12px;letter-spacing:3px;">
            AUTONOMOUS UNDERWATER VEHICLE • PRJ-44
        </div>
        <h1 style="margin:6px 0 4px 0;">SAUG-HPI Sonar Hazard Engine</h1>
        <p style="margin:0;">
            Side-scan sonar analysis • Shadow-aware denoising •
            Highlight–shadow evidence • Mission history
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE: DASHBOARD
# ============================================================
if page == "Mission Dashboard":
    history = load_missions()

    st.subheader("Mission overview")
    a, b, c, d = st.columns(4)

    a.metric("Saved assessments", len(history))
    b.metric(
        "Potential hazards",
        int((history["status"] == "POTENTIAL HAZARD").sum()) if len(history) else 0,
    )
    c.metric(
        "Review required",
        int((history["status"] == "REVIEW REQUIRED").sum()) if len(history) else 0,
    )
    d.metric("Engine", "READY", help="The local image-analysis app is loaded.")

    st.divider()
    st.subheader("Start analysis")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🛰️ Live Mission Demo")
        st.write("Generate a reproducible synthetic sonar scene and analyse it.")
        if st.button("Open demo analysis", use_container_width=True):
            st.session_state["goto_demo"] = True
            st.info("Select **Live Mission Demo** from the sidebar.")

    with right:
        st.markdown("### 🖼️ Real image input")
        st.write("Upload a sonar image for image-based analysis.")
        st.info("Use the **Sonar Image Analysis** page in the sidebar.")

    st.divider()
    st.subheader("Recent mission records")
    if history.empty:
        st.info("No saved assessments yet. Run an analysis and save its result.")
    else:
        st.dataframe(history.head(10), use_container_width=True, hide_index=True)


# ============================================================
# PAGE: SYNTHETIC LIVE DEMO
# ============================================================
elif page == "Live Mission Demo":
    st.subheader("Synthetic sonar mission")
    st.warning(
        "This is a simulated demonstration scene, not a real-time AUV or sonar sensor feed."
    )

    clean, noisy, truth = make_demo_scene(
        seed=seed,
        hazards=hazard_count,
        clutter=clutter_count,
    )

    st.caption(
        f"Scene generated with seed {seed}. "
        "Changing the seed generates a different reproducible scene."
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image("Synthetic clean reference", clean)
    with col2:
        show_image("Simulated noisy sonar", noisy)

    result = analyse_sonar(noisy)

    st.divider()
    st.subheader("Detection output")

    k1, k2, k3 = st.columns(3)
    k1.metric("SAUG-HPI heuristic", f"{result['hpi']:.1f}/100")
    k2.metric("Peak pair response", f"{result['peak']:.4f}")
    k3.metric("Status", result["status"])

    left, right = st.columns(2)
    with left:
        show_image("Denoised synthetic sonar", result["denoised"])
    with right:
        show_image("Highlight-shadow pair map", result["pair"])

    st.subheader("Reference-based denoising metrics")
    p1, p2, s1 = st.columns(3)
    p1.metric("Input PSNR", f"{calculate_psnr(clean, noisy):.2f} dB")
    p2.metric(
        "Denoised PSNR",
        f"{calculate_psnr(clean, result['denoised']):.2f} dB",
    )
    s1.metric(
        "Denoised SNR",
        f"{calculate_snr(clean, result['denoised']):.2f} dB",
    )

    st.caption(
        "These metrics use the known clean synthetic reference. "
        "They are not measured real-world performance."
    )

    if st.button("Save demo assessment"):
        save_mission(
            "Synthetic demo",
            result["hpi"],
            result["status"],
            calculate_snr(clean, result["denoised"]),
            f"synthetic_seed_{seed}",
        )
        st.success("Demo assessment saved.")


# ============================================================
# PAGE: UPLOADED IMAGE ANALYSIS
# ============================================================
elif page == "Sonar Image Analysis":
    st.subheader("Analyse a sonar image")
    st.write(
        "Upload a sonar image. The engine estimates a highlight-shadow pair, "
        "applies shadow-aware smoothing, and reports a heuristic score."
    )

    uploaded = st.file_uploader(
        "Upload sonar image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    )

    if uploaded is not None:
        try:
            pil_image = Image.open(uploaded)
            image = to_gray_array(pil_image)
            st.success(
                f"Image loaded: {uploaded.name} — "
                f"{pil_image.width} × {pil_image.height} pixels"
            )
            render_analysis(
                image,
                source="Uploaded image",
                image_name=uploaded.name,
            )
        except Exception as exc:
            st.error(f"Could not process this image: {exc}")
    else:
        st.info("Choose an image above to start analysis.")


# ============================================================
# PAGE: SAUG-HPI NOVELTY LAB
# ============================================================
elif page == "SAUG-HPI Novelty Lab":
    st.subheader("SAUG-HPI: shadow-aware hazard evidence")
    st.markdown("""
    **Research idea:** side-scan sonar objects may create a bright highlight
    followed by a darker acoustic shadow. The prototype estimates this paired
    structure and uses it as evidence for candidate localization.

    **Important:** the current implementation uses image-processing heuristics.
    It is not a trained deconvolutional CNN and its HPI score is not a calibrated
    probability of a mine or ordnance.
    """)

    clean, noisy, _ = make_demo_scene(
        seed=seed,
        hazards=hazard_count,
        clutter=clutter_count,
    )
    denoised, _ = shadow_aware_denoise(noisy)

    raw_pair, _, _ = highlight_shadow_map(noisy)
    clean_pair, _, _ = highlight_shadow_map(denoised)

    raw_peak = float(raw_pair.max())
    den_peak = float(clean_pair.max())
    raw_hpi = float(np.clip(raw_peak / 0.20, 0, 1) * 100)
    den_hpi = float(np.clip(den_peak / 0.20, 0, 1) * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Input HPI heuristic", f"{raw_hpi:.1f}/100")
    c2.metric("Denoised HPI heuristic", f"{den_hpi:.1f}/100")
    c3.metric(
        "Peak response change",
        f"{(den_peak - raw_peak):+.4f}",
    )

    left, right = st.columns(2)
    with left:
        show_image("Before denoising: pair map", raw_pair)
    with right:
        show_image("After denoising: pair map", clean_pair)

    st.subheader("ESCC-style counterfactual check")
    st.write(
        "This demonstration compares the image evidence before and after "
        "denoising. It is a preliminary consistency experiment, not a "
        "validated ESCC detector."
    )

    eps = 1e-6
    relative_change = abs(den_peak - raw_peak) / max(raw_peak, eps) * 100

    e1, e2 = st.columns(2)
    e1.metric("Relative peak change", f"{relative_change:.1f}%")
    e2.metric(
        "Consistency indicator",
        "STABLE" if relative_change < 25 else "CHANGED",
    )

    st.caption(
        "The 25% threshold is a configurable demonstration rule, not an "
        "experimentally validated acceptance threshold."
    )

    experiment = pd.DataFrame([
        {
            "Stage": "Noisy input",
            "Peak pair strength": raw_peak,
            "HPI heuristic": raw_hpi,
        },
        {
            "Stage": "After denoising",
            "Peak pair strength": den_peak,
            "HPI heuristic": den_hpi,
        },
    ])

    st.subheader("Experiment results")
    st.dataframe(experiment, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(experiment["Stage"], experiment["Peak pair strength"])
    ax.set_ylabel("Peak pair strength")
    ax.set_title("Highlight-shadow evidence comparison")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.download_button(
        "Download experiment CSV",
        data=experiment.to_csv(index=False).encode("utf-8"),
        file_name="saug_hpi_experiment.csv",
        mime="text/csv",
    )


# ============================================================
# PAGE: MISSION HISTORY
# ============================================================
elif page == "Mission History":
    st.subheader("Local mission database")
    history = load_missions()

    if history.empty:
        st.info("Mission history is empty. Run an analysis and save the result.")
    else:
        st.dataframe(history, use_container_width=True, hide_index=True)
        st.download_button(
            "Export mission history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="prj44_mission_history.csv",
            mime="text/csv",
        )

        counts = history["status"].value_counts().reset_index()
        counts.columns = ["Assessment", "Count"]
        st.subheader("Saved assessment summary")
        st.bar_chart(counts.set_index("Assessment"))

    st.caption(f"Local database file: {DB_PATH.resolve()}")


# ============================================================
# PAGE: ABOUT
# ============================================================
elif page == "About the Project":
    st.subheader("Project overview")

    st.markdown("""
    ### PRJ-44
    **An AUV Side-Scan Sonar Hazard Classification Engine using
    Deconvolutional Feature-De-noising CNNs**

    **Problem**
    Side-scan sonar imagery can contain speckle noise, acoustic reflections,
    and seabed clutter. These effects can make object interpretation difficult.

    **Current prototype functions**
    - Sonar image upload and grayscale preprocessing.
    - Shadow-aware image smoothing.
    - Highlight–shadow pair evidence map.
    - Candidate localization and SAUG-HPI heuristic score.
    - Synthetic reference-based PSNR and SNR measurements.
    - ESCC-style before/after evidence comparison.
    - SQLite mission history and CSV export.

    **Research limitation**
    This replacement implements a computer-vision baseline. It does not itself
    load or train a deconvolutional CNN. CNN-based classification, trained
    weights, a validated dataset, and evaluation against ground-truth labels
    are required before reporting real classification accuracy.

    **Safety**
    This is an academic research prototype. Do not use it to make operational
    mine-clearance or underwater navigation decisions.
    """)

st.sidebar.divider()
st.sidebar.caption("PRJ-44 • Academic research prototype")
