import numpy as np
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


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


def render_page():
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

    if upload is None:
        st.info("Upload a sonar image to calculate the ESCC score.")
        return

    try:
        image = Image.open(upload).convert("RGB")

        result = analyze_escc(image, height, angle, pixel)

        left, right = st.columns(2)

        with left:
            st.image(image, caption="Input sonar image", width="stretch")

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
