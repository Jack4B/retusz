"""White balance: 'Temperatura' / 'Odcień' as linear-light channel gains, plus the eyedropper.

temp, tint in -1..1. ln gains: R = +KT*temp, B = -KT*temp, G = -KG*tint, normalised so a neutral
keeps its Rec.709 luminance. Applied to the baked LUT output (after the colour match), so it costs
nothing per pixel and preview and export are identical. The eyedropper solves the temp/tint change
that makes a clicked patch neutral (closed form in log space)."""
from __future__ import annotations
import numpy as np

KT, KG = 0.35, 0.25
Y709 = np.array([0.2126, 0.7152, 0.0722])
MIN_V, MAX_V = 12, 250          # 8-bit patch limits for the eyedropper (too dark / clipped -> refuse)

def to_lin(x):
    """sRGB-encoded uint8 (or 0..1 float) -> linear float."""
    v = x.astype(np.float64)/255 if x.dtype == np.uint8 else np.asarray(x, np.float64)
    return np.where(v <= 0.04045, v/12.92, ((v+0.055)/1.055)**2.4)

def to_srgb01(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin*12.92, 1.055*lin**(1/2.4) - 0.055)

def to_srgb8(lin):
    return np.clip(to_srgb01(lin)*255 + 0.5, 0, 255).astype(np.uint8)

def gains(temp, tint):
    g = np.exp(np.array([KT*temp, -KG*tint, -KT*temp], np.float64))
    return g/float(Y709 @ g)

# sRGB (D65) -> CIE XYZ (IEC 61966-2-1) and CIE/ICC Bradford cone response matrix (Lam 1985)
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], np.float64)
_XYZ_TO_BFD = np.array([
    [+0.8951000, +0.2664000, -0.1614000],
    [-0.7502000, +1.7135000, +0.0367000],
    [+0.0389000, -0.0685000, +1.0296000],
], np.float64)
_SRGB_TO_BFD = _XYZ_TO_BFD @ _SRGB_TO_XYZ
_BFD_TO_SRGB = np.linalg.inv(_SRGB_TO_BFD)
_WHITE_BFD = _SRGB_TO_BFD @ np.ones(3, np.float64)

def bradford_matrix(temp, tint):
    """3x3 linear sRGB chromatic adaptation matrix (Bradford CAT, Lam 1985), scaled directly in Bradford
    cone space so M(t1+t2) = M(t1)@M(t2) up to luminance normalisation (exact eyedropper round-trip)."""
    s = np.exp(np.array([KT*temp, -KG*tint, -KT*temp], np.float64))
    M = _BFD_TO_SRGB @ np.diag(s) @ _SRGB_TO_BFD
    w = M @ np.ones(3, np.float64)
    return M / float(Y709 @ w)

def apply_bradford01(rgb01, temp, tint):
    """Apply Bradford CAT white balance to sRGB-encoded 0..1 array (exact on neutrals, cleaner skin/sky hues)."""
    if not temp and not tint: return rgb01
    M = bradford_matrix(temp, tint)
    lin = to_lin(rgb01) @ M.T
    out = to_srgb01(lin)
    return out.astype(rgb01.dtype, copy=False)

def apply01(rgb01, temp, tint):
    """rgb01: float array (..., 3) of sRGB-encoded values 0..1 (LUT entries)."""
    if not temp and not tint: return rgb01
    out = to_srgb01(to_lin(rgb01)*gains(temp, tint))
    return out.astype(rgb01.dtype, copy=False)

def solve(patch):
    """(dtemp, dtint) that neutralise the mean of an 8-bit patch in Bradford cone space, or None."""
    p = np.asarray(patch).reshape(-1, 3)
    if p.size == 0 or p.max() > MAX_V or p.mean(0).max() < MIN_V or p.mean(0).min() < 2: return None
    lms = (_SRGB_TO_BFD @ to_lin(p.astype(np.uint8)).mean(0)) / _WHITE_BFD
    if np.any(lms <= 1e-6): return None
    L, M_, S = lms
    return float(np.log(S/L)/(2*KT)), float(-np.log(np.sqrt(L*S)/M_)/KG)
