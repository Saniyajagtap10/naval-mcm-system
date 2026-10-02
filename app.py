"""
PRJ-44: SAUG-HPI AUV Sonar Hazard Engine  --  single-file Streamlit app.
Run:  streamlit run app.py
"""
import time
import sqlite3
from collections import deque

import numpy as np
import pandas as pd
import cv2
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ======================================================================
# CORE (saug_core)
# ======================================================================
W = 200          # range bins (columns)  -> acoustic range direction
H = 160          # along-track rows per ping window
STEP = 10        # rows the AUV advances per ping


# ---------------------------------------------------------------- scene model
def make_scene(seed, n_mines=3, n_clutter=5, length=900):
    """Long seabed strip (waterfall image) with mines (highlight+shadow)
    and clutter (bright rocks WITHOUT shadow, dark patches WITHOUT highlight)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, W)[None, :]
    rows = np.arange(length)[:, None]
    clean = 0.45 + 0.04 * np.sin(2 * np.pi * (x * 6 + rows / 90.0))
    clean = clean * (1.0 - 0.12 * x)                      # range falloff
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
        if i % 2 == 0:   # bright rock, no shadow
            clean[((yy - r) ** 2 + (xx - c) ** 2) <= rad ** 2] += 0.35
        else:            # dark patch, no highlight
            clean[(np.abs(yy - r) <= rad) & (xx > c) & (xx <= c + 2 * rad)] *= 0.3
        clutter.append((r, c))
    return np.clip(clean, 0, 1), mines, clutter


def render_ping(clean_strip, start, seed, noise=0.25):
    clean = clean_strip[start:start + H]
    rng = np.random.default_rng(seed)
    speckle = rng.gamma(4.0, 1 / 4.0, clean.shape)         # multiplicative speckle
    noisy = np.clip(clean * speckle + rng.normal(0, noise * 0.4, clean.shape), 0, 1)
    return clean.astype(np.float32), noisy.astype(np.float32)


# ------------------------------------------------------------- shadow pair map
def pair_map(img):
    """Highlight-then-shadow strength per pixel (intensity units)."""
    s = cv2.GaussianBlur(img, (5, 5), 0)
    bg = cv2.medianBlur((s * 255).astype(np.uint8), 31).astype(np.float32) / 255.0
    h = np.clip(s - bg, 0, None)
    d = np.clip(bg - s, 0, None)
    h_box = cv2.blur(h, (9, 11))
    d_box = cv2.blur(d, (15, 11))
    d_shift = np.roll(d_box, -13, axis=1)                  # shadow lies downrange
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


# --------------------------------------------------------- shadow-aware denoise
def shadow_aware_denoise(noisy):
    """Blend light filtering (near highlight/shadow structure) with heavy
    filtering (background), weighted by the shadow-pair prior."""
    u8 = (noisy * 255).astype(np.uint8)
    heavy = cv2.GaussianBlur(u8, (0, 0), 2.2)
    light = cv2.GaussianBlur(u8, (0, 0), 0.9)
    pm, _ = pair_map(noisy)
    wmap = cv2.GaussianBlur(np.clip(pm / 0.12, 0, 1), (0, 0), 8)
    wmap = np.clip(wmap * 3.0, 0, 1)
    out = wmap * light + (1 - wmap) * heavy
    return (out / 255.0).astype(np.float32), wmap


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def snr_db(clean, test):
    return 10 * np.log10(np.sum(clean ** 2) / max(np.sum((clean - test) ** 2), 1e-10))


# ------------------------------------------------------------- uncertainty
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
    unc = float(np.clip(np.std(probs) * 2.0, 0, 1))        # spread of probability
    return dict(p=p, unc=unc, strength=strength, brightness=bright, r=int(r), c=int(c),
                denoised=den, wmap=wmap)


# ------------------------------------------------------------- persistence index
class HazardTracker:
    """Hazard Persistence Index over the last N pings."""
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
        match = [h for h in self.hist if h["p"] >= self.cand_p
                 and abs(h["row"] - cur["row"]) <= self.tol and abs(h["col"] - cur["col"]) <= self.tol]
        persistence = len(match) / self.n
        mp = float(np.mean([h["p"] for h in match]))
        mu = float(np.mean([h["unc"] for h in match]))
        hpi = mp * (1 - mu) * persistence
