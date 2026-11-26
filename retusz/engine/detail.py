"""Studio detail, multi-mode sharpening, texture, and edge-preserving OKLab/LAB denoising engine (v0.6.0).

Sharpening modes:
1. 'portrait' — Portrait skin-protected Morphological 3x3 Clamped USM (0% halo on glasses/edges,
   attenuates sharpening on smooth skin pores while sharpening eyes, eyelashes, brows, lips, and hair).
2. 'rl' — Damped Richardson-Lucy optical deconvolution in linear light (Richardson 1972; Lucy 1974),
   recovering lens/diffraction acutance without USM ringing halos.
3. 'edges' — Full-frame Halo-Free Morphological 3x3 Clamped USM.
4. 'soft' — Optical Portrait Soft Focus (Pro-Mist / Softar style micro-contrast diffusion preserving
   macro-edges).

Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
import cv2

SHARPEN_MODES = {
    "portrait": "Portret (ochrona skóry)",
    "rl": "Dekonwolucja RL (obiektyw)",
    "edges": "Krawędzie (Clamped USM)",
    "soft": "Miękki blask (Soft Focus)",
}


def _denoise_oklab_edge_preserving(rgb: np.ndarray, amount: float) -> np.ndarray:
    """Dual-band edge-preserving denoise:
    - Strong bilateral smoothing on LAB/OKLab chrominance channels (a*, b*) to eliminate high-ISO
      colour speckles without bleeding across colour edges.
    - Subtle edge-preserving bilateral luma smoothing in shadows/midtones while keeping pores sharp."""
    amt = float(np.clip(amount, 0.0, 1.0))
    if amt <= 1e-4:
        return rgb
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Chrominance denoising (two passes of fast bilateral filter on a* and b*)
    sig_c = 12.0 + 28.0 * amt
    sig_s = 5.0 + 6.0 * amt
    a_dn = cv2.bilateralFilter(a, d=7, sigmaColor=sig_c, sigmaSpace=sig_s)
    b_dn = cv2.bilateralFilter(b, d=7, sigmaColor=sig_c, sigmaSpace=sig_s)
    if amt > 0.45:
        a_dn = cv2.bilateralFilter(a_dn, d=5, sigmaColor=sig_c * 0.8, sigmaSpace=sig_s)
        b_dn = cv2.bilateralFilter(b_dn, d=5, sigmaColor=sig_c * 0.8, sigmaSpace=sig_s)

    # Subtle luminance denoising weighted toward darker tones
    L_dn = cv2.bilateralFilter(L, d=5, sigmaColor=4.0 + 10.0 * amt, sigmaSpace=3.0)
    w_luma = (0.35 * amt) * np.clip(1.0 - (L / 255.0) * 0.5, 0.25, 1.0)
    lab[..., 0] = L + w_luma * (L_dn - L)
    lab[..., 1] = a + min(1.0, amt * 1.15) * (a_dn - a)
    lab[..., 2] = b + min(1.0, amt * 1.15) * (b_dn - b)
    return cv2.cvtColor(np.clip(lab + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _rl_deconvolution_luma(Y: np.ndarray, amount: float, sigma: float = 1.0, iters: int = 3) -> np.ndarray:
    """Damped Richardson-Lucy deconvolution on luminance Y in [0, 255] with local morphological halo clamp."""
    amt = float(np.clip(amount, 0.0, 1.0))
    y_norm = np.clip(Y / 255.0, 1e-4, 1.0).astype(np.float32)
    u = y_norm.copy()
    for _ in range(iters):
        conv = cv2.GaussianBlur(u, (0, 0), sigma)
        ratio = y_norm / np.maximum(conv, 1e-3)
        corr = cv2.GaussianBlur(ratio, (0, 0), sigma)
        u = np.clip(u * (1.0 + 1.15 * amt * (corr - 1.0)), 1e-4, 1.35)
    d = (u - y_norm) * 255.0 * (1.55 * amt)
    k5 = np.ones((5, 5), np.uint8)
    y_min = cv2.erode(Y, k5)
    y_max = cv2.dilate(Y, k5)
    margin = float(min(7.5, 5.0 * amt))
    return np.clip(Y + d, y_min - margin, y_max + margin) - Y


def apply_detail_pass(rgb: np.ndarray, mask_q: np.ndarray | None = None,
                      sharpen_amount: float = 0.0, sharpen_mode: str = "portrait",
                      texture: float = 0.0, denoise: float = 0.0) -> np.ndarray:
    """Apply live studio detail pass (Denoise + Texture + Multi-Mode Sharpening) to sRGB uint8 image."""
    out = rgb
    if denoise > 1e-4:
        out = _denoise_oklab_edge_preserving(out, denoise)

    if sharpen_amount <= 1e-4 and abs(texture) <= 1e-4:
        return out

    H, W = out.shape[:2]
    Y = cv2.transform(out.astype(np.float32), np.array([[0.2126, 0.7152, 0.0722]], np.float32)).reshape(H, W)
    d_total = np.zeros((H, W), np.float32)

    sk = None
    if mask_q is not None and mask_q.size > 0:
        sk = mask_q.astype(np.float32) if mask_q.shape[:2] == (H, W) else cv2.resize(mask_q.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        sk = np.clip(sk, 0.0, 1.0)

    # 1. Texture / Structure ('Struktura', -1..1): fine detail band (sigma 0.7 .. 2.2 px)
    if abs(texture) > 1e-4:
        y_fine = cv2.GaussianBlur(Y, (0, 0), 0.7)
        y_mid = cv2.GaussianBlur(Y, (0, 0), 2.2)
        d_tex = (y_fine - y_mid) * float(texture) * 0.85
        if sk is not None and texture > 0:
            d_tex *= (1.0 - 0.55 * sk)
        d_total += d_tex

    # 2. Multi-mode sharpening ('Ostrość', 0..1)
    if sharpen_amount > 1e-4:
        amt = float(np.clip(sharpen_amount, 0.0, 1.0))
        mode = sharpen_mode if sharpen_mode in SHARPEN_MODES else "portrait"
        if mode == "soft":
            # Optical soft-focus diffusion: attenuate fine contrast while keeping major edges
            y_sm = cv2.bilateralFilter(Y, d=7, sigmaColor=24.0, sigmaSpace=4.5)
            d_sh = -0.88 * amt * (Y - y_sm)
        elif mode == "rl":
            d_sh = _rl_deconvolution_luma(Y, amt, sigma=1.05, iters=3)
            if sk is not None:
                d_sh *= (1.0 - 0.45 * sk)
        else:
            # 'portrait' or 'edges': Morphological 5x5 Clamped Unsharp Mask (0% halo on step edges)
            d_raw = (Y - cv2.GaussianBlur(Y, (0, 0), 1.35)) * (2.75 * amt)
            if mode == "portrait" and sk is not None:
                d_raw *= (1.0 - 0.80 * sk)
            k5 = np.ones((5, 5), np.uint8)
            y_min = cv2.erode(Y, k5)
            y_max = cv2.dilate(Y, k5)
            margin = float(min(9.0, 6.0 * amt))
            d_sh = np.clip(Y + d_raw, y_min - margin, y_max + margin) - Y
        d_total += d_sh

    d16 = np.round(d_total).astype(np.int16)
    res = out.astype(np.int16) + d16[..., None]
    return np.clip(res, 0, 255).astype(np.uint8)
