"""Fast Local Laplacian Filters (Fast LLF) for halo-free multi-scale local contrast and clarity.

References:
- Sylvain Paris, Samuel W. Hasinoff, Jan Kautz, "Local Laplacian Filters: Edge-aware Image Processing
  with a Laplacian Pyramid", ACM Transactions on Graphics (SIGGRAPH), 2011.
- Mathieu Aubry, Sylvain Paris, Samuel W. Hasinoff, Jan Kautz, Frédo Durand, "Fast Local Laplacian
  Filters: Theory and Applications", ACM Transactions on Graphics, 33(5), 2014.

Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
import cv2

LLF_SIDE = 256      # Canonical working scale for preview == export parity (ΔE < 1)
LLF_LEVELS = 8      # Discrete intensity sample levels gamma_k in [0, 100] (Aubry et al. 2014)
LLF_PYR_DEPTH = 3   # Number of Laplacian pyramid octaves


def _build_gaussian_pyr(img: np.ndarray, depth: int) -> list[np.ndarray]:
    pyr = [img]
    cur = img
    for _ in range(depth):
        h, w = cur.shape[:2]
        if min(h, w) <= 4:
            break
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _build_laplacian_pyr(img: np.ndarray, depth: int) -> list[np.ndarray]:
    g_pyr = _build_gaussian_pyr(img, depth)
    l_pyr = []
    for i in range(len(g_pyr) - 1):
        gh, gw = g_pyr[i].shape[:2]
        up = cv2.pyrUp(g_pyr[i + 1], dstsize=(gw, gh))
        l_pyr.append(g_pyr[i] - up)
    l_pyr.append(g_pyr[-1])
    return l_pyr


def _collapse_laplacian_pyr(l_pyr: list[np.ndarray]) -> np.ndarray:
    cur = l_pyr[-1]
    for i in range(len(l_pyr) - 2, -1, -1):
        gh, gw = l_pyr[i].shape[:2]
        cur = cv2.pyrUp(cur, dstsize=(gw, gh)) + l_pyr[i]
    return cur


def fast_llf_luminance(L: np.ndarray, alpha: float, sigma_r: float = 11.0,
                       num_levels: int = LLF_LEVELS, depth: int = LLF_PYR_DEPTH) -> np.ndarray:
    """Apply Fast Local Laplacian Filter (Aubry et al. 2014) to luminance L in [0, 100].
    - alpha > 0: boosts fine and medium local details (|I - gamma| <= sigma_r) without amplifying
      large step edges (|I - gamma| >> sigma_r), guaranteeing zero halo across high-contrast boundaries.
    - alpha < 0: smooths fine details while keeping large structural boundaries sharp."""
    if abs(alpha) < 1e-4:
        return L.astype(np.float32)
    L_f = np.clip(np.asarray(L, np.float32), 0.0, 100.0)
    g_pyr = _build_gaussian_pyr(L_f, depth)
    eff_depth = len(g_pyr) - 1
    if eff_depth < 1:
        return L_f

    gammas = np.linspace(0.0, 100.0, num_levels, dtype=np.float32)
    step = 100.0 / max(1, num_levels - 1)
    inv_2sig2 = np.float32(1.0 / (2.0 * (sigma_r ** 2)))
    a_f = np.float32(alpha)

    # Precompute Laplacian pyramids for each discrete intensity sample gamma_k
    remapped_l_pyrs = []
    for gk in gammas:
        d = L_f - gk
        # Smooth bell remapping function r_gamma(I): identity for |d| >> sigma_r (zero step-edge overshoot)
        r_img = L_f + a_f * d * np.exp(-(d * d) * inv_2sig2)
        remapped_l_pyrs.append(_build_laplacian_pyr(r_img, eff_depth))

    # Interpolate each pyramid level using the local Gaussian pyramid value G_l[L]
    out_l_pyr = []
    for lvl in range(eff_depth):
        g_lvl = np.clip(g_pyr[lvl], 0.0, 100.0)
        pos = g_lvl / step
        idx0 = np.clip(np.floor(pos).astype(np.int32), 0, num_levels - 2)
        t = (pos - idx0).astype(np.float32)
        acc = np.zeros_like(g_lvl, dtype=np.float32)
        for k in range(num_levels - 1):
            mk = (idx0 == k)
            if np.any(mk):
                l0 = remapped_l_pyrs[k][lvl]
                l1 = remapped_l_pyrs[k + 1][lvl]
                acc[mk] = (1.0 - t[mk]) * l0[mk] + t[mk] * l1[mk]
        out_l_pyr.append(acc)
    out_l_pyr.append(g_pyr[-1])
    return np.clip(_collapse_laplacian_pyr(out_l_pyr), 0.0, 100.0).astype(np.float32)


def fast_llf_gain(Lq: np.ndarray, skin_q: np.ndarray | None, clarity: float) -> np.ndarray:
    """Compute resolution-independent relative luminance gain map from Fast Local Laplacian Filter.
    Always resamples to canonical long side LLF_SIDE so the pyramid operates at the exact same
    spatial frequencies on preview and full-res export (ΔE < 1).
    Positive clarity boosts 3D micro-contrast with zero halo while gently sparing skin pores;
    negative clarity produces soft portrait glow while preserving edges."""
    if not clarity or abs(clarity) < 1e-4:
        return np.zeros(Lq.shape, np.float32)
    qh, qw = Lq.shape[:2]
    sc = 160.0 / max(qh, qw, 1)
    cw, ch = max(8, round(qw * sc)), max(8, round(qh * sc))
    interp_in = cv2.INTER_AREA if sc <= 1.0 else cv2.INTER_LINEAR
    L = Lq.astype(np.float32) if (ch, cw) == (qh, qw) else cv2.resize(Lq.astype(np.float32), (cw, ch), interpolation=interp_in)

    # Remap clarity in [-1, 1] to LLF detail strength alpha
    alpha = float(clarity) * (1.20 if clarity > 0 else 0.75)
    L_llf = fast_llf_luminance(L, alpha=alpha, sigma_r=11.0)
    det = np.clip(L_llf - L, -12.0, 12.0)

    l01 = np.clip(L / 100.0, 0.0, 1.0)
    w_mid = 0.25 + 0.75 * (4.0 * l01 * (1.0 - l01))
    if skin_q is not None:
        sk = skin_q.astype(np.float32) if skin_q.shape[:2] == (ch, cw) else cv2.resize(skin_q.astype(np.float32), (cw, ch), interpolation=interp_in)
        sk = np.clip(sk, 0.0, 1.0)
    else:
        sk = 0.0
    skin_w = (1.0 - 0.62 * sk) if clarity > 0 else (1.0 - 0.25 * sk)
    g = np.clip(w_mid * skin_w * det / np.maximum(L, 20.0), -0.25, 0.25).astype(np.float32)
    return g if (ch, cw) == (qh, qw) else cv2.resize(g, (qw, qh), interpolation=cv2.INTER_LINEAR)

