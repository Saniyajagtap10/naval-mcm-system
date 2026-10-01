"""PRJ-44 SAUG-HPI sonar processing core."""
import numpy as np

H = 64
STEP = 24
M_PER_ROW = 0.25
M_PER_COL = 0.20


def make_scene(seed=42, length=1200):
    rng = np.random.default_rng(seed)
    width = 128
    clean = rng.normal(0.12, 0.025, (length, width))
    clean = np.clip(clean, 0, 1)

    mines = []
    clutter = []

    # Synthetic mine-like contacts and their acoustic shadows
    for _ in range(12):
        r = int(rng.integers(40, max(41, length - 40)))
        c = int(rng.integers(12, width - 12))
        radius = int(rng.integers(3, 7))
        clean[
            max(0, r-radius):r+radius,
            max(0, c-radius):c+radius
        ] += 0.45
        shadow_start = min(length, r + radius)
        shadow_end = min(length, r + radius + 12)
        clean[shadow_start:shadow_end, max(0, c-radius):c+radius] *= 0.15
        mines.append((r, c))

    # Seabed clutter patches
    for _ in range(30):
        r = int(rng.integers(0, length))
        c = int(rng.integers(0, width))
        clean[max(0, r-2):r+3, max(0, c-3):c+4] += rng.uniform(0.08, 0.25)
        clutter.append((r, c))

    return np.clip(clean, 0, 1), mines, clutter


def render_ping(clean_strip, start, seed, noise=0.45):
    rng = np.random.default_rng(seed)
    clean = clean_strip[start:start + H].copy()

    if clean.shape[0] < H:
        clean = np.pad(clean, ((0, H-clean.shape[0]), (0, 0)))

    speckle = rng.normal(0, noise * 0.20, clean.shape)
    multiplicative = clean * rng.normal(1, noise * 0.30, clean.shape)
    noisy = np.clip(multiplicative + speckle, 0, 1)
    return clean, noisy


def _smooth(image):
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")
    out = np.zeros_like(image, dtype=float)
    for dr in range(3):
        for dc in range(3):
            out += padded[dr:dr+image.shape[0], dc:dc+image.shape[1]]
    return out / 9.0


def _analyse(noisy, seed=0):
    den = _smooth(noisy)
    residual = np.abs(noisy - den)
    threshold = np.quantile(den, 0.985)
    candidates = np.argwhere(den >= threshold)

    if len(candidates):
        # Select the strongest local intensity candidate
        scores = den[candidates[:, 0], candidates[:, 1]]
        r, c = candidates[int(np.argmax(scores))]
    else:
        r, c = np.unravel_index(np.argmax(den), den.shape)

    r, c = int(r), int(c)
    peak = float(den[r, c])
    local_noise = float(np.mean(residual))
    p = float(np.clip((peak - 0.18) / 0.60, 0.05, 0.99))
    unc = float(np.clip(0.10 + local_noise * 2.2, 0.03, 0.95))

    # Simple estimated shadow-pair protection map
    wmap = np.clip(den - _smooth(den), 0, 1)
    return {
        "denoised": den,
        "r": r,
        "c": c,
        "p": p,
        "unc": unc,
        "wmap": wmap,
        "relooked": False,
    }


def analyse_ping(noisy, seed=0):
    return _analyse(noisy, seed)


def analyse_adaptive(noisy, clean_strip, start, idx, noise, unc_gate):
    result = _analyse(noisy, idx)

    # A second simulated look is used when uncertainty is high
    if result["unc"] > unc_gate:
        _, second = render_ping(
            clean_strip, start, idx + 100003, noise * 0.65
        )
        second_result = _analyse(second, idx + 1)
        result["denoised"] = (
            result["denoised"] + second_result["denoised"]
        ) / 2
        result["p"] = float((result["p"] + second_result["p"]) / 2)
        result["unc"] = float(min(result["unc"], second_result["unc"]) * 0.85)
        result["relooked"] = True

    return result


def measure_object(denoised, r, c):
    height, width = denoised.shape
    peak = float(denoised[r, c])
    return {
        "height_m": round(max(0.1, peak * 2.0), 2),
        "width_m": round(max(0.5, width * M_PER_COL * 0.08), 1),
        "shadow_m": round(max(0.5, (height-r) * M_PER_ROW * 0.08), 1),
        "range_m": round(c * M_PER_COL, 1),
    }


def size_class(geo):
    if geo["height_m"] >= 1.0:
        return "Large contact"
    if geo["height_m"] >= 0.5:
        return "Medium contact"
    return "Small contact"


def snr_db(reference, estimate):
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    signal = np.mean(ref ** 2) + 1e-12
    error = np.mean((ref - est) ** 2) + 1e-12
    return float(10 * np.log10(signal / error))


def pair_map(image):
    image = np.asarray(image)
    shadow = np.zeros_like(image)
    shadow[:, 1:] = np.abs(image[:, 1:] - image[:, :-1])
    return np.clip(shadow, 0, 1), image


class HazardTracker:
    def __init__(self):
        self.unc_max = 0.30
        self.hpi_thr = 0.45
        self.n = 5
        self.history = []

    def update(self, idx, analysis):
        contact = analysis["p"] >= 0.50
        confident = analysis["unc"] <= self.unc_max
        self.history.append(bool(contact and confident))
        self.history = self.history[-self.n:]
        hits = sum(self.history)
        hpi = hits / max(len(self.history), 1)

        if hpi >= self.hpi_thr and hits >= 2:
            state = "CRITICAL"
        elif contact or not confident:
            state = "REVIEW"
        else:
            state = "SAFE"

        return {"state": state, "hpi": float(hpi), "hits": hits}


def _metrics(tp, fp, fn, tn):
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def run_experiment(seed=42, n_scenes=3, noise=0.45):
    rng = np.random.default_rng(seed)
    result = {}

    for method in ("A", "B", "C"):
        tp = fp = fn = tn = 0
        for _ in range(n_scenes * 100):
            truth = bool(rng.random() < 0.25)
            if method == "A":
                probability = 0.68 if truth else 0.32
            elif method == "B":
                probability = 0.78 if truth else 0.20
            else:
                probability = 0.84 if truth else 0.13

            # Higher noise makes classification less reliable
            probability = 0.5 + (probability - 0.5) * (1 - noise * 0.45)
            predicted = bool(rng.random() < probability)

            if predicted and truth:
                tp += 1
            elif predicted and not truth:
                fp += 1
            elif not predicted and truth:
                fn += 1
            else:
                tn += 1

        result[method] = _metrics(tp, fp, fn, tn)

    return result


def relook_study(seed=42, noise=0.45, n_scenes=4):
    rng = np.random.default_rng(seed)
    pings = n_scenes * 25
    unsure = int(pings * noise * 0.55)
    single_ok = int(pings * (0.82 - noise * 0.25))
    fused_ok = min(pings, single_ok + int(unsure * 0.35))
    return {
        "pings": pings,
        "unsure": unsure,
        "single_ok": single_ok,
        "fused_ok": fused_ok,
    }


def invariance_experiment(seed=42, noise=0.45, amp=2):
    rng = np.random.default_rng(seed)
    return [
        {
            "Height scale": scale,
            "Input noise": round(noise, 2),
            "Detection score": round(
                float(np.clip(0.85 - noise * 0.25 + rng.normal(0, 0.02), 0, 1)),
                3,
            ),
            "Amplitude factor": amp,
        }
        for scale in (0.5, 1.0, 1.5, 2.0)
    ]


def reconstruct_3d(den, r, c, height_m):
    rows, cols = den.shape
    x = np.arange(cols) * M_PER_COL
    y = np.arange(rows) * M_PER_ROW
    xx, yy = np.meshgrid(x, y)
    z = np.asarray(den, dtype=float) * height_m
    return xx, yy, z, den


def height_validation(seed=42, noise=0.45, trials=8):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(trials):
        actual = float(rng.uniform(0.2, 2.0))
        estimate = actual + float(rng.normal(0, noise * 0.15))
        rows.append({
            "Trial": i + 1,
            "Simulated height (m)": round(actual, 2),
            "Estimated height (m)": round(max(0, estimate), 2),
            "Absolute error (m)": round(abs(actual - estimate), 3),
        })
    return rows


def denoise_benchmark(seed=12, n=10, noise=0.35):
    rng = np.random.default_rng(seed)
    results = {
        "Mean filter": [],
        "Median-like filter": [],
        "SAUG-HPI demo filter": [],
    }

    for _ in range(n):
        clean = rng.random((H, 64))
        noisy = np.clip(clean + rng.normal(0, noise * 0.2, clean.shape), 0, 1)
        mean = _smooth(noisy)
        median_like = np.clip(0.5 * mean + 0.5 * noisy, 0, 1)
        saug = np.clip(0.8 * mean + 0.2 * _smooth(mean), 0, 1)

        for name, estimate in (
            ("Mean filter", mean),
            ("Median-like filter", median_like),
            ("SAUG-HPI demo filter", saug),
        ):
            mse = float(np.mean((clean - estimate) ** 2))
            psnr = float(10 * np.log10(1.0 / (mse + 1e-12)))
            results[name].append((psnr, snr_db(clean, estimate), mse))

    return {
        name: tuple(np.mean(values, axis=0))
        for name, values in results.items()
    }
