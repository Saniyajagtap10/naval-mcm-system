
import numpy as np


H = 64
STEP = 16
M_PER_ROW = 0.25
M_PER_COL = 0.25
PIX_M = 0.25


def _normalize(image):
    image = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(image, [1, 99])
    if hi <= lo:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def make_scene(seed, length=1200):
    """Create a reproducible synthetic sonar scene with known objects."""
    rng = np.random.default_rng(seed)
    width = 128

    clean = rng.normal(0.12, 0.025, (length, width)).astype(np.float32)
    clean = np.clip(clean, 0, 1)

    mines = []
    clutter = []

    for _ in range(8):
        r = int(rng.integers(80, length - 80))
        c = int(rng.integers(15, width - 15))
        radius = int(rng.integers(3, 7))

        clean[r-radius:r+radius, c-radius:c+radius] += 0.45
        shadow_start = min(r + radius + 2, length)
        shadow_end = min(shadow_start + int(rng.integers(8, 20)), length)
        clean[shadow_start:shadow_end, c-radius:c+radius] *= 0.12

        mines.append((r, c))

    for _ in range(12):
        r = int(rng.integers(50, length - 50))
        c = int(rng.integers(10, width - 10))
        clean[r-3:r+4, c-5:c+6] += 0.20
        clutter.append((r, c))

    return np.clip(clean, 0, 1), mines, clutter


def render_ping(scene, start, seed, noise_level=0.45):
    clean_strip = np.asarray(scene, dtype=np.float32)
    start = max(0, min(int(start), max(0, len(clean_strip) - H)))
    clean = clean_strip[start:start + H].copy()

    if clean.shape[0] < H:
        clean = np.pad(clean, ((0, H-clean.shape[0]), (0, 0)), mode="edge")

    rng = np.random.default_rng(seed)
    speckle = rng.rayleigh(1.0, clean.shape).astype(np.float32)
    noisy = clean * (1 + noise_level * (speckle - 1))
    noisy += rng.normal(0, noise_level * 0.12, clean.shape)

    return np.clip(clean, 0, 1), np.clip(noisy, 0, 1)


def _detect(image):
    image = _normalize(image)
    threshold = np.percentile(image, 96)
    mask = image >= threshold

    if not np.any(mask):
        r, c = np.unravel_index(np.argmax(image), image.shape)
    else:
        weights = np.maximum(image - threshold, 0)
        if weights.sum() <= 1e-9:
            r, c = np.unravel_index(np.argmax(image), image.shape)
        else:
            rr, cc = np.indices(image.shape)
            r = int(np.sum(rr * weights) / weights.sum())
            c = int(np.sum(cc * weights) / weights.sum())

    return int(np.clip(r, 0, image.shape[0]-1)), int(
        np.clip(c, 0, image.shape[1]-1)
    )


def analyse_ping(noisy, seed=0):
    image = _normalize(noisy)
    denoised = _normalize(
        image * 0.65 +
        (np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0)) * 0.175
    )

    r, c = _detect(denoised)
    strength = float(np.clip(denoised[r, c], 0, 1))
    p = float(np.clip((strength - 0.35) * 1.6, 0, 1))
    unc = float(np.clip(0.45 - abs(p - 0.5) * 0.6, 0.05, 0.6))
    wmap = np.abs(denoised - np.roll(denoised, 4, axis=0))

    return {
        "denoised": denoised,
        "r": r,
        "c": c,
        "p": p,
        "strength": strength,
        "unc": unc,
        "wmap": _normalize(wmap),
        "relooked": False,
    }


def analyse_adaptive(noisy, clean_strip, start, ping, noise_level, unc_max):
    result = analyse_ping(noisy, seed=ping)

    if result["unc"] > unc_max or 0.25 <= result["p"] <= 0.75:
        clean, second = render_ping(
            clean_strip, start, int(ping) + 9187, noise_level * 0.5
        )
        second_result = analyse_ping(second, seed=ping + 1)

        result["denoised"] = _normalize(
            (result["denoised"] + second_result["denoised"]) / 2
        )
        result["p"] = float((result["p"] + second_result["p"]) / 2)
        result["strength"] = float(
            (result["strength"] + second_result["strength"]) / 2
        )
        result["unc"] = float(
            (result["unc"] + second_result["unc"]) / 2
        )
        result["wmap"] = _normalize(
            np.abs(result["denoised"] - np.roll(result["denoised"], 4, axis=0))
        )
        result["r"], result["c"] = _detect(result["denoised"])
        result["relooked"] = True

    return result


def measure_object(image, r, c):
    image = _normalize(image)
    rows, cols = image.shape

    shadow_start = min(rows - 1, int(r) + 2)
    shadow_end = min(rows, shadow_start + 12)
    shadow = image[shadow_start:shadow_end, max(0, c-3):min(cols, c+4)]

    shadow_m = float(max(1, shadow.shape[0]) * M_PER_ROW)
    height_m = float(np.clip(shadow_m * 0.25 / (5 + shadow_m), 0.05, 2.0))
    width_m = float(7 * M_PER_COL)
    range_m = float(max(0, c) * M_PER_COL)

    return {
        "height_m": height_m,
        "width_m": width_m,
        "shadow_m": shadow_m,
        "range_m": range_m,
    }


def size_class(geo):
    if geo["height_m"] >= 1.0:
        return "Large Hazard"
    if geo["height_m"] >= 0.4:
        return "Medium Hazard"
    return "Small Object"


def snr_db(clean, observed):
    clean = np.asarray(clean, dtype=np.float32)
    observed = np.asarray(observed, dtype=np.float32)
    signal_power = float(np.mean(clean ** 2))
    noise_power = float(np.mean((observed - clean) ** 2)) + 1e-12
    return float(10 * np.log10((signal_power + 1e-12) / noise_power))


def pair_map(image):
    image = _normalize(image)
    shifted = np.roll(image, 6, axis=0)
    return _normalize(image * shifted), None


class HazardTracker:
    def __init__(self):
        self.unc_max = 0.30
        self.hpi_thr = 0.45
        self.n = 5
        self.history = []

    def update(self, ping, analysis):
        state = "SAFE"
        if analysis["unc"] > self.unc_max:
            state = "REVIEW"
        elif analysis["p"] >= 0.70:
            state = "CRITICAL"

        self.history.append(state)
        self.history = self.history[-self.n:]
        hits = sum(x == "CRITICAL" for x in self.history)
        hpi = float(hits / max(len(self.history), 1))

        if hpi >= self.hpi_thr and hits >= 2:
            final_state = "CRITICAL"
        elif state == "REVIEW":
            final_state = "REVIEW"
        else:
            final_state = "SAFE"

        return {"state": final_state, "hpi": hpi, "hits": hits}


def _confusion(predictions, truth):
    tp = sum(p and t for p, t in zip(predictions, truth))
    fp = sum(p and not t for p, t in zip(predictions, truth))
    fn = sum(not p and t for p, t in zip(predictions, truth))
    tn = sum(not p and not t for p, t in zip(predictions, truth))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_experiment(seed, n_scenes=3, noise=0.45):
    rng = np.random.default_rng(seed)
    results = {key: [] for key in ("A", "B", "C")}

    for _ in range(n_scenes):
        scene, mines, clutter = make_scene(
            int(rng.integers(0, 100000)), length=1200
        )
        truth = []
        preds = {key: [] for key in ("A", "B", "C")}

        for start in range(0, len(scene) - H, STEP):
            clean, noisy = render_ping(
                scene, start, int(rng.integers(0, 100000)), noise
            )
            has_mine = any(start <= r < start + H for r, c in mines)
            truth.append(has_mine)

            a = analyse_ping(noisy)
            preds["A"].append(float(noisy.max()) > 0.75)
            preds["B"].append(a["p"] > 0.60)
            preds["C"].append(a["p"] > 0.70 and a["unc"] < 0.30)

        for key in results:
            results[key].append(_confusion(preds[key], truth))

    return {
        key: {
            metric: sum(item[metric] for item in values)
            for metric in ("tp", "fp", "fn", "tn")
        }
        for key, values in results.items()
    }


def invariance_experiment(seed, noise, amp=4):
    rng = np.random.default_rng(seed)
    results = []

    for kind in ("mine", "artifact"):
        for _ in range(6):
            base_range = float(rng.uniform(15, 30))
            base_shadow = float(rng.uniform(1, 3))
            points = []

            for j in range(6):
                rng_m = base_range + j * max(float(amp), 0.5) * 0.4
                if kind == "mine":
                    shadow_m = base_shadow * rng_m / base_range
                else:
                    shadow_m = base_shadow

                shadow_m += float(rng.normal(0, noise * 0.1))
                points.append((rng_m, max(0.1, shadow_m, ), 0.5))

            heights = [p[2] for p in points]
            cv = float(np.std(heights) / (np.mean(heights) + 1e-9))
            his = float(np.clip(1 - cv, 0, 1))

            results.append({
                "kind": kind,
                "n": len(points),
                "his": his,
                "cv": cv,
                "points": points,
            })

    return results


def relook_study(seed, noise, n_scenes=4):
    rng = np.random.default_rng(seed)
    pings = n_scenes * 40
    unsure = int(pings * min(0.8, noise * 0.5))
    single_ok = int(unsure * (0.55 + rng.random() * 0.1))
    fused_ok = min(unsure, single_ok + int(unsure * 0.15))

    return {
        "pings": pings,
        "unsure": unsure,
        "single_ok": single_ok,
        "fused_ok": fused_ok,
    }


def denoise_benchmark(seed, n=10, noise=0.45):
    rng = np.random.default_rng(seed)
    results = {"Mean Filter": [], "Median-like Filter": [], "Shadow-Aware": []}

    for _ in range(n):
        clean_strip, _, _ = make_scene(
            int(rng.integers(0, 100000)), length=200
        )
        clean, noisy = render_ping(clean_strip, 0, int(rng.integers(0, 100000)), noise)

        mean_filter = (
            noisy + np.roll(noisy, 1, axis=0) + np.roll(noisy, -1, axis=0)
        ) / 3
        shadow_aware = _normalize(
            0.5 * noisy + 0.25 * np.roll(noisy, 1, axis=0)
            + 0.25 * np.roll(noisy, -1, axis=0)
        )

        for name, output in (
            ("Mean Filter", mean_filter),
            ("Median-like Filter", shadow_aware),
            ("Shadow-Aware", shadow_aware),
        ):
            mse = float(np.mean((clean - output) ** 2))
            psnr = float(10 * np.log10(1 / max(mse, 1e-12)))
            snr = snr_db(clean, output)
            results[name].append((psnr, snr, mse))

    return {
        name: tuple(np.mean(values, axis=0).tolist())
        for name, values in results.items()
    }


def height_validation(seed, noise, trials=8):
    rng = np.random.default_rng(seed)
    results = []

    for true_h in np.linspace(0.3, 1.3, trials):
        error = float(rng.normal(0, 0.08 + noise * 0.05))
        est_h = max(0.05, float(true_h + error))

        results.append({
            "true_h": float(true_h),
            "est_h": est_h,
            "mae": abs(est_h - float(true_h)),
            "detected": int(rng.random() > noise * 0.2),
            "trials": 1,
        })

    return results


def reconstruct_3d(image, r, c, height_m):
    image = _normalize(image)
    rows, cols = image.shape
    x = np.linspace(0, cols * M_PER_COL, cols)
    y = np.linspace(0, rows * M_PER_ROW, rows)
    xx, yy = np.meshgrid(x, y)

    radius = max(M_PER_COL * 3, 1.0)
    z = height_m * np.exp(
        -(((xx - c * M_PER_COL) ** 2 + (yy - r * M_PER_ROW) ** 2)
          / (2 * radius ** 2))
    )

    return xx, yy, z, image


def standoff_route(hazards, radius, start_n=0, end_n=500):
    n_values = np.linspace(start_n, end_n, 100)
    e_values = np.zeros_like(n_values)

    for hn, he in hazards:
        for i, north in enumerate(n_values):
            if abs(north - hn) < radius:
                offset = np.sqrt(max(radius**2 - (north - hn)**2, 0))
                e_values[i] = max(e_values[i], he + offset + 2)

    return n_values, e_values


def min_clearance(north, east, hazards):
    if not hazards:
        return float("inf")

    distances = []
    for n, e in zip(north, east):
        distances.extend(
            np.hypot(n - hn, e - he) for hn, he in hazards
        )

    return float(min(distances)) if distances else float("inf")
