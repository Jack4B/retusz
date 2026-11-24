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
