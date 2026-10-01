
"""
PRJ-44 | ESCC Novelty Lab
Echo–Shadow Counterfactual Consistency

Experimental prototype for side-scan sonar images.
The consistency score is NOT a hazard probability or validated accuracy.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image


def _gray_image(image):
    """Convert an uploaded image to normalized grayscale."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32)

    gray = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    low, high = np.percentile(gray, [1, 99])

    if high <= low:
        return np.zeros_like(gray, dtype=np.float32)

    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def _smooth(values, window=5):
    """Smooth a one-dimensional signal."""
    window = min(window, len(values))

    if window <= 1:
        return values.copy()

    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def analyze_escc(
    image,
    grazing_angle_deg=30.0,
    target_height_m=0.5,
    pixel_size_m=0.05,
    shadow_direction="right",
):
    """
    Estimate a candidate highlight and compare predicted and observed
    dark-region lengths.

    Assumptions:
    - Image horizontal axis corresponds to sonar range.
    - The brightest local region may represent a target highlight.
    - Target height, pixel scale, grazing angle, and shadow direction
      are supplied correctly by the user.

    This is a research prototype and requires dataset validation.
    """

    if not 1.0 < grazing_angle_deg < 89.0:
        raise ValueError("Grazing angle must be between 1 and 89 degrees.")

    if target_height_m <= 0 or pixel_size_m <= 0:
        raise ValueError("Target height and pixel size must be positive.")

    if shadow_direction not in ("left", "right"):
        raise ValueError("Choose left or right shadow direction.")

    gray_original = _gray_image(image)

    if gray_original.shape[0] < 4 or gray_original.shape[1] < 12:
        raise ValueError("The uploaded image is too small for analysis.")

    # Flip the working image so the expected shadow is on the right.
    gray = (
        gray_original[:, ::-1].copy()
        if shadow_direction == "left"
        else gray_original.copy()
    )

    height, width = gray.shape

    # Mean intensity along the sonar-range direction.
    profile = _smooth(gray.mean(axis=0), 5)

    margin = max(2, int(width * 0.05))
    valid_profile = profile[margin:width - margin]

    if valid_profile.size == 0:
        raise ValueError("Unable to locate a candidate region.")

    peak_col = int(np.argmax(valid_profile)) + margin
    peak_value = float(profile[peak_col])

    # Estimate highlight extent around the strongest peak.
    threshold = max(
        float(np.percentile(profile, 75)),
        peak_value * 0.70,
    )

    left = peak_col
    right = peak_col

    while left > margin and profile[left - 1] >= threshold:
        left -= 1

    while right < width - margin - 1:
        if profile[right + 1] < threshold:
            break
        right += 1

    highlight_width = max(1, right - left + 1)

    # Basic geometry model. Calibration is needed for real sonar data.
    angle_rad = np.deg2rad(grazing_angle_deg)
    expected_shadow_m = target_height_m / np.tan(angle_rad)
    expected_shadow_px = max(
        1,
        int(round(expected_shadow_m / pixel_size_m)),
    )

    # Look for the first contiguous dark region after the highlight.
    search_start = right + 1
    search_end = min(width, search_start + expected_shadow_px * 3)

    shadow_start = None
    shadow_end = None
    observed_shadow_px = 0
    shadow_intensity = None

    if search_start < search_end:
        region = profile[search_start:search_end]
        dark_threshold = float(np.percentile(profile, 25))
        dark_mask = region <= dark_threshold

        indices = np.flatnonzero(dark_mask)

        if indices.size:
            run_start = int(indices[0])
            run_end = run_start

            while (
                run_end + 1 < len(dark_mask)
                and dark_mask[run_end + 1]
            ):
                run_end += 1

            shadow_start = search_start + run_start
            shadow_end = search_start + run_end
            observed_shadow_px = run_end - run_start + 1

            shadow_intensity = float(
                np.mean(profile[shadow_start:shadow_end + 1])
            )

    # Compare predicted and observed lengths.
    length_error_px = abs(expected_shadow_px - observed_shadow_px)

    length_consistency = float(
        np.exp(-length_error_px / max(expected_shadow_px, 1))
    )

    highlight_intensity = float(np.mean(profile[left:right + 1]))

    if shadow_intensity is not None:
        contrast_score = float(
            np.clip(
                (highlight_intensity - shadow_intensity)
                / max(highlight_intensity, 1e-6),
                0.0,
                1.0,
            )
        )
    else:
        contrast_score = 0.0

    consistency_score = float(
        np.clip(
            0.75 * length_consistency + 0.25 * contrast_score,
            0.0,
            1.0,
        )
    )

    # This is a review recommendation, not a hazard classification.
    needs_review = consistency_score < 0.45

    # Map detected coordinates back to the original image orientation.
    if shadow_direction == "left":
        highlight_column = width - 1 - peak_col
        highlight_start = width - 1 - right
        highlight_end = width - 1 - left

        if shadow_start is not None:
            display_shadow_start = width - 1 - shadow_end
            display_shadow_end = width - 1 - shadow_start
        else:
            display_shadow_start = None
            display_shadow_end = None
    else:
        highlight_column = peak_col
        highlight_start = left
        highlight_end = right
        display_shadow_start = shadow_start
        display_shadow_end = shadow_end

    return {
        "highlight_column": highlight_column,
        "highlight_start": highlight_start,
        "highlight_end": highlight_end,
        "highlight_width_px": highlight_width,
        "expected_shadow_m": float(expected_shadow_m),
        "expected_shadow_px": expected_shadow_px,
        "observed_shadow_px": observed_shadow_px,
        "shadow_length_error_px": length_error_px,
        "length_consistency": length_consistency,
        "contrast_score": contrast_score,
        "consistency_score": consistency_score,
        "needs_review": needs_review,
        "shadow_start": display_shadow_start,
        "shadow_end": display_shadow_end,
        "gray_image": gray_original,
        "range_profile": profile,
    }


def render_page():
    """Render the ESCC research dashboard in Streamlit."""

    st.header("ESCC — Echo–Shadow Counterfactual Consistency")
    st.caption(
        "PRJ-44 research prototype | Experimental highlight-shadow analysis"
    )

    st.warning(
        "This prototype estimates shadow consistency. It does not prove "
        "that an object is hazardous, and its score is not classification "
        "accuracy. Validate it using labelled sonar data."
    )

    uploaded_file = st.file_uploader(
        "Upload a side-scan sonar image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="escc_image_upload",
    )

    col1, col2 = st.columns(2)

    with col1:
        angle = st.slider(
            "Grazing angle (degrees)",
            min_value=5,
            max_value=80,
            value=30,
        )

        target_height = st.number_input(
            "Assumed target height (metres)",
            min_value=0.05,
            max_value=10.0,
            value=0.5,
            step=0.05,
        )

    with col2:
        pixel_size = st.number_input(
            "Ground pixel size (metres/pixel)",
            min_value=0.001,
            max_value=5.0,
            value=0.05,
            step=0.01,
            format="%.3f",
        )

        direction = st.selectbox(
            "Expected shadow direction",
            ["right", "left"],
        )

    if uploaded_file is None:
        st.info("Upload a sonar image to start ESCC analysis.")
        return

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.subheader("Input sonar image")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Run ESCC Analysis", type="primary"):
            result = analyze_escc(
                image=image,
                grazing_angle_deg=float(angle),
                target_height_m=float(target_height),
                pixel_size_m=float(pixel_size),
                shadow_direction=direction,
            )

            st.subheader("Analysis measurements")

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Expected shadow",
                f"{result['expected_shadow_px']} px",
            )
            metric2.metric(
                "Observed dark region",
                f"{result['observed_shadow_px']} px",
            )
            metric3.metric(
                "Consistency score",
                f"{result['consistency_score']:.3f}",
            )

            st.progress(result["consistency_score"])

            if result["needs_review"]:
                st.warning(
                    "Low consistency: flag this image for review or "
                    "consider collecting another view."
                )
            else:
                st.info(
                    "Consistency is above the experimental threshold. "
                    "This does not confirm a hazard or rule one out."
                )

            st.subheader("Highlight and shadow visualization")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.imshow(result["gray_image"], cmap="gray")

            ax.axvline(
                result["highlight_column"],
                linestyle="--",
                label="Candidate highlight",
            )

            if result["shadow_start"] is not None:
                ax.axvspan(
                    result["shadow_start"],
                    result["shadow_end"],
                    alpha=0.35,
                    label="Observed dark region",
                )

            ax.set_xlabel("Image range coordinate (pixels)")
            ax.set_ylabel("Image row (pixels)")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Mean range-intensity profile")

            profile_fig, profile_ax = plt.subplots(figsize=(12, 3))
            profile_ax.plot(result["range_profile"])
            profile_ax.set_xlabel("Range column")
            profile_ax.set_ylabel("Normalized mean intensity")
            profile_ax.grid(True, alpha=0.25)
            st.pyplot(profile_fig)
            plt.close(profile_fig)

            st.subheader("Research output")

            output = {
                key: value
                for key, value in result.items()
                if key not in ("gray_image", "range_profile")
            }
            st.json(output)

            st.download_button(
                "Download measurements as JSON",
                data=__import__("json").dumps(output, indent=2),
                file_name="escc_measurements.json",
                mime="application/json",
            )

    except Exception as exc:
        st.error(f"ESCC analysis error: {exc}")


if __name__ == "__main__":
    render_page()
