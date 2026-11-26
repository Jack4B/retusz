"""Histogram and clipping warning for the result (values as exported, in the working colour space).

Highlight clip = any channel >= HI (a blown channel loses detail even if the others survive);
shadow clip = all channels <= LO (pure black)."""
from __future__ import annotations
import numpy as np

LO, HI = 2, 254
HI_COLOR = (255, 40, 40)
LO_COLOR = (40, 90, 255)
_Y = np.array([0.2126, 0.7152, 0.0722], np.float32)

def histogram(rgb, bins=256):
    """Normalised histograms of R, G, B and luminance (each sums to 1)."""
    a = rgb.reshape(-1, 3); n = max(1, a.shape[0]); out = {}
    for i, k in enumerate("rgb"):
        out[k] = np.bincount(a[:, i], minlength=256)[:256].astype(np.float64)/n
    y = np.clip(a.astype(np.float32) @ _Y + 0.5, 0, 255).astype(np.uint8)
    out["l"] = np.bincount(y, minlength=256)[:256].astype(np.float64)/n
    if bins != 256:
        for k in out: out[k] = out[k].reshape(bins, -1).sum(1)
    return out

def clipping(rgb, lo=LO, hi=HI):
    """(shadow fraction, highlight fraction, shadow mask, highlight mask)."""
    mhi = rgb.max(-1) >= hi; mlo = rgb.max(-1) <= lo
    n = max(1, mhi.size)
    return float(mlo.sum())/n, float(mhi.sum())/n, mlo, mhi

def overlay(rgb, lo=LO, hi=HI):
    """Copy of rgb with clipped highlights red and clipped shadows blue."""
    _, _, mlo, mhi = clipping(rgb, lo, hi)
    out = rgb.copy(); out[mhi] = HI_COLOR; out[mlo] = LO_COLOR
    return out


SKIN_LINE_DEG = 52.0  # Universal portrait skin hue locus in OKLCh (~52° +- 8°)


def skin_vectorscope(rgb: np.ndarray, mask: np.ndarray | None = None, size: int = 64) -> dict:
    """Compute a 2D OKLab/OKLCh chroma vectorscope grid (size x size) and estimate the chroma-weighted
    mean skin hue angle in degrees relative to the universal 52° OKLCh Skin-Tone Line."""
    from . import tone as TN
    h, w = rgb.shape[:2]
    step = max(1, min(h, w) // 96)
    sub = rgb[::step, ::step].astype(np.float32) / 255.0
    ok = TN.lin_to_oklab(TN.srgb_to_lin(sub))
    a = ok[..., 1].ravel()
    b = ok[..., 2].ravel()
    C = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0

    if mask is not None and mask.size > 0:
        import cv2
        ms = cv2.resize(mask.astype(np.float32), (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST).ravel()
        w_sk = np.clip(ms, 0.0, 1.0) * C
    else:
        # Weight pixels within the broad skin-adjacent sector (10°..110°) by their chroma
        dh_broad = np.abs(((hue - SKIN_LINE_DEG + 180.0) % 360.0) - 180.0)
        w_sk = np.clip(1.0 - dh_broad / 55.0, 0.05, 1.0) * C

    if float(w_sk.sum()) > 1e-6:
        mean_a = float((a * w_sk).sum() / w_sk.sum())
        mean_b = float((b * w_sk).sum() / w_sk.sum())
        hue_deg = float((np.degrees(np.arctan2(mean_b, mean_a)) + 360.0) % 360.0)
        mean_c = float(np.hypot(mean_a, mean_b))
    else:
        hue_deg = SKIN_LINE_DEG
        mean_c = 0.0
    delta_deg = float(((hue_deg - SKIN_LINE_DEG + 180.0) % 360.0) - 180.0)

    # Project (a, b) in [-0.18, 0.18] onto a size x size density grid (x = +a, y = -b so +b is up)
    r_max = 0.18
    gx = np.clip(((a / r_max * 0.5 + 0.5) * (size - 1)).astype(np.int32), 0, size - 1)
    gy = np.clip((((-b) / r_max * 0.5 + 0.5) * (size - 1)).astype(np.int32), 0, size - 1)
    grid = np.zeros((size, size), np.float32)
    np.add.at(grid, (gy, gx), 1.0)
    mx = float(grid.max())
    if mx > 0:
        grid = np.sqrt(grid / mx)
    return {
        "grid": grid,
        "hue_deg": hue_deg,
        "delta_deg": delta_deg,
        "mean_c": mean_c,
        "skin_line_deg": SKIN_LINE_DEG,
    }

