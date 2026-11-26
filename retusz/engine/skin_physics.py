"""Biophysical skin optics & anatomical portrait retouching engine (v0.5.0).

Algorithms & scientific references:
1. Dichromatic SUV Specular Highlight Removal ('De-Shine'):
   - Steven A. Shafer, "Using Color to Separate Reflection Components", Color Research & Application, 1985.
   - Satya P. Mallick, Todd E. Zickler,man, David J. Kriegman, Peter N. Belhumeur, "Beyond Lambert:
     Reconstructing Specular Surfaces Using Color", IEEE CVPR, 2005 (SUV color space).
   - Qingxiong Yang, Shengnan Wang,arendra Ahuja, "Real-Time Specular Highlight Removal Using
     Bilateral Filtering", ECCV, 2010.
2. Optical-Density Melanin & Hemoglobin Chromophore Separation:
   - Norimichi Tsumura, Hideaki Haneishi, Yoichi Miyake, "Independent-component analysis of skin color
     image", JOSA A, 1999.
   - Norimichi Tsumura et al., "Image-based skin color and texture analysis/synthesis by extracting
     hemoglobin and melanin information in the skin", ACM Transactions on Graphics (SIGGRAPH), 2003.
3. Two-Band Pore-Preserving Blemish Auto-Heal & Anatomical Eyes/Teeth Enhancement:
   - Multi-scale hemoglobin/LoG anomaly detection at canonical face scale (DB_FACE_PX = 256) inside
     safe skin weights, base-layer ring interpolation, and 100 % high-frequency pore preservation.
   - YuNet 5-point landmark geometric iris, sclera, under-eye, and natural teeth enhancement.

Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
import cv2
from . import tone as TN, dnb as DNB

FACE_WORK_PX = 256  # Canonical face width in pixels for preview == export parity (ΔE < 1)


def decompose_chromophores(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose an sRGB uint8 or float 0..1 image into Beer-Lambert optical density chromophore maps
    (Tsumura et al., SIGGRAPH 2003):
      D = -log10(max(I_lin, 1e-4))
    Returns:
      - melanin: spatial melanin density map (epidermal pigmentation)
      - hemoglobin: spatial hemoglobin density map (dermal erythema / redness / vasculature)
      - D: 3-channel optical density (-log10 R_lin, -log10 G_lin, -log10 B_lin)"""
    x = rgb.astype(np.float32) / 255.0 if rgb.dtype == np.uint8 else np.clip(rgb.astype(np.float32), 0.0, 1.0)
    lin = TN.srgb_to_lin(x)
    D = -np.log10(np.maximum(lin, 1e-4))
    dr, dg, db = D[..., 0], D[..., 1], D[..., 2]
    # Hemoglobin absorbs strongly in Green (540-577 nm) and Blue relative to Red (>620 nm);
    # orthogonalized against melanin's monotonic spectral slope from red to blue:
    hemoglobin = ((dg - dr) - 0.35 * (db - dg)).astype(np.float32)
    melanin = (0.65 * (db - dr) + 0.35 * dr).astype(np.float32)
    return melanin, hemoglobin, D


def deshine_gain(rgb_q: np.ndarray, mask: np.ndarray, amount: float) -> np.ndarray:
    """Dichromatic SUV Specular Highlight Removal ('Redukcja połysku / De-Shine', Mallick 2005 / Yang 2010).
    Removes white surface glare (sebum/sweat reflection where min(R,G,B) rises toward white) on skin while
    recovering the underlying warm diffuse skin chromaticity and preserving 100 % of full-res pore detail
    via multiplicative quarter-resolution per-channel gain."""
    H, W = rgb_q.shape[:2]
    if amount <= 1e-4 or mask is None:
        return np.zeros((H, W, 3), np.float32)
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if float((m > 0.25).sum()) < 25:
        return np.zeros((H, W, 3), np.float32)

    img = rgb_q.astype(np.float32)
    # In Shafer's dichromatic model I = I_diffuse + m_s*(1,1,1):
    # On warm human skin (R > G > B), I_min = min(R, G, B) carries the white specular offset m_s,
    # while chroma span C = I_max - I_min reflects diffuse subsurface saturation.
    i_min = img.min(axis=-1)
    i_max = img.max(axis=-1)
    chroma = np.maximum(i_max - i_min, 1.0)
    lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

    sk_m = m > 0.25
    med_min = float(np.median(i_min[sk_m]))
    med_lum = float(np.median(lum[sk_m]))

    # Diffuse confidence weight: matte skin has i_min near median diffuse skin and higher chroma/min ratio;
    # oily specular glare has elevated i_min and elevated luminance.
    sat_ratio = chroma / np.maximum(i_min, 10.0)
    w_min = np.exp(-np.maximum(i_min - med_min, 0.0) ** 2 / (2.0 * 11.0 ** 2))
    w_lum = np.exp(-np.maximum(lum - med_lum, 0.0) ** 2 / (2.0 * 14.0 ** 2))
    w_diffuse = m * np.clip(sat_ratio / 0.35, 0.02, 1.0) * w_min * w_lum + 1e-4 * m

    sigma = max(6.0, min(H, W) / 4.0)
    den = cv2.GaussianBlur(w_diffuse, (0, 0), sigma) + 1e-6
    base_rgb = cv2.GaussianBlur(img * w_diffuse[..., None], (0, 0), sigma) / den[..., None]
    base_min = base_rgb.min(axis=-1)
    base_lum = 0.299 * base_rgb[..., 0] + 0.587 * base_rgb[..., 1] + 0.114 * base_rgb[..., 2]

    # Specular excess m_s(x): how much the minimum channel and luminance exceed the surrounding diffuse baseline
    spec_min = np.maximum(0.0, i_min - base_min)
    spec_lum = np.maximum(0.0, lum - base_lum)
    spec = np.minimum(spec_min, 1.15 * spec_lum)

    # Smooth local estimate of specular component (so fine pore micro-highlights survive while broad oily glare is removed)
    spec_low = cv2.GaussianBlur(spec, (0, 0), max(1.2, min(H, W) / 48.0))
    amt = float(np.clip(amount, 0.0, 1.0))

    # Subtract white specular component m_s*(1,1,1) and pull compressed highlight chroma toward diffuse baseline chroma
    base_chroma_vec = base_rgb - base_lum[..., None]
    cur_chroma_vec = img - lum[..., None]
    glare_w = np.clip(spec_low / 28.0, 0.0, 1.0) * m
    target = img - (amt * 0.96 * spec_low)[..., None] + (amt * 0.55 * glare_w)[..., None] * (base_chroma_vec - cur_chroma_vec)
    target = np.clip(target, 0.0, 255.0)

    g = np.where(img > 4.0, target / np.maximum(img, 4.0) - 1.0, 0.0)
    return np.clip(g * m[..., None], -0.45, 0.25).astype(np.float32)


def chromophore_even_gain(rgb_q: np.ndarray, lab_q: np.ndarray, mask: np.ndarray,
                          faces_q: list, amount: float) -> np.ndarray:
    """Biophysical skin evening combining optical-density Hemoglobin/Melanin balancing (Tsumura 2003)
    with low-frequency LAB a*/b* mixed-light evening. Attenuates red erythema blotches and local casts
    while keeping melanin golden tone and pore texture intact."""
    m = (mask > 0.3).astype(np.float32)
    if m.sum() < 50 or amount <= 1e-4:
        return np.zeros(rgb_q.shape, np.float32)
    H, W = m.shape
    fw = float(np.median([f[2] for f in faces_q])) if faces_q else W / 3.0

    # 1. Optical-density Hemoglobin blotch evening (Tsumura 2003)
    _, hem, D = decompose_chromophores(rgb_q)
    sig_hem = max(2.0, fw / 18.0)
    w_sk = mask > 0.5
    hem_med = float(np.median(hem[w_sk])) if w_sk.sum() > 20 else float((hem * m).sum() / max(m.sum(), 1.0))
    # Smooth hemoglobin excess at medium-low spatial scale (erythema / red blotches)
    hem_low = cv2.GaussianBlur(hem * m, (0, 0), max(1.2, fw / 42.0)) / (cv2.GaussianBlur(m, (0, 0), max(1.2, fw / 42.0)) + 1e-4)
    d_hem = np.clip(hem_low - hem_med, -0.18, 0.18) * float(amount) * mask

    # Reconstruct linear RGB after reducing hemoglobin excess (hemoglobin absorbs primarily G and B)
    D_corr = D.copy()
    D_corr[..., 1] -= 0.78 * d_hem
    D_corr[..., 2] -= 0.56 * d_hem
    D_corr[..., 0] -= 0.10 * d_hem
    lin_corr = np.power(10.0, -np.clip(D_corr, 0.0, 4.0)).astype(np.float32)
    srgb_hem = TN.lin_to_srgb(np.clip(lin_corr, 0.0, 1.0))

    # 2. Low-frequency mixed-light a*/b* evening
    ab = lab_q[..., 1:]
    k = max(1.0, fw / 32.0)
    sw, sh = max(4, round(W / k)), max(4, round(H / k))
    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA)
    abs_ = cv2.resize(np.ascontiguousarray(ab * m[..., None]), (sw, sh), interpolation=cv2.INTER_AREA)
    sg = max(1.0, fw / k * 0.5)
    den = cv2.GaussianBlur(ms, (0, 0), sg) + 1e-4
    low = cv2.GaussianBlur(abs_, (0, 0), sg) / den[..., None]
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR)
    target = np.median(ab[w_sk], axis=0) if w_sk.sum() > 20 else (ab * m[..., None]).reshape(-1, 2).sum(0) / m.sum()
    corr = target - low
    mag = np.linalg.norm(corr, axis=2, keepdims=True)
    corr *= np.minimum(1.0, 12.0 / (mag + 1e-6))

    lab_t = cv2.cvtColor(srgb_hem, cv2.COLOR_RGB2LAB)
    lab_t[..., 1:] += (0.65 * float(amount)) * corr * mask[..., None]
    rgb_t = cv2.cvtColor(lab_t.astype(np.float32), cv2.COLOR_LAB2RGB)

    cur = rgb_q.astype(np.float32) / 255.0
    g = np.where(cur > 0.03, rgb_t / np.maximum(cur, 0.03) - 1.0, 0.0)
    return np.clip(g * mask[..., None], -0.35, 0.35).astype(np.float32)


def mixed_light_face_mask(rgb_q: np.ndarray, mask: np.ndarray,
                          faces_q: list, lms_q: list | None = None) -> np.ndarray:
    """Anatomical face + neck skin mask resilient to extreme mixed lighting (e.g. half face chalk-white
    flash where Cr ~ 128, half face red/magenta club LED where Cr > 190).
    Uses YuNet face geometry, luminance-only background rejection, landmark eye/mouth exclusion, and
    structure-guided edge refinement so both halves of a split-lit face are captured cleanly."""
    H, W = rgb_q.shape[:2]
    m_base = np.clip(mask.astype(np.float32), 0.0, 1.0) if mask is not None else np.zeros((H, W), np.float32)
    if not faces_q:
        return m_base

    from .color import guided_refine_rgb
    img = rgb_q.astype(np.float32)
    # Effective luminance supporting both white-flash and red-LED skin
    Y_eff = np.maximum(0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2],
                       0.58 * img[..., 0] + 0.30 * img[..., 1] + 0.12 * img[..., 2])
    anat = np.zeros((H, W), np.float32)
    feat = np.zeros((H, W), np.float32)

    for idx, f in enumerate(faces_q):
        x, y, w, h = (float(v) for v in f[:4])
        cx, cy = int(round(x + 0.5 * w)), int(round(y + 0.52 * h))
        ell = np.zeros((H, W), np.uint8)
        cv2.ellipse(ell, (cx, cy), (max(1, int(round(w * 0.48))), max(1, int(round(h * 0.58)))), 0, 0, 360, 1, -1)
        cv2.ellipse(ell, (cx, int(round(y + 1.05 * h))),
                    (max(1, int(round(w * 0.28))), max(1, int(round(h * 0.26)))), 0, 0, 360, 1, -1)

        core = np.zeros((H, W), np.uint8)
        cv2.ellipse(core, (cx, cy), (max(1, int(round(w * 0.36))), max(1, int(round(h * 0.40)))), 0, 0, 360, 1, -1)
        cm = core > 0
        if cm.sum() >= 16:
            yc = Y_eff[cm]
            clean = cm & (Y_eff >= np.percentile(yc, 22.0))
            if clean.sum() < 10:
                clean = cm
            y_med = float(np.median(Y_eff[clean]))
            y_lo = max(16.0, y_med * 0.40)
            y_hi = min(255.0, max(y_med * 1.48, 248.0))
            fm = ((ell > 0) & (Y_eff >= y_lo) & (Y_eff <= y_hi)).astype(np.float32)
        else:
            fm = (ell > 0).astype(np.float32)
        anat = np.maximum(anat, fm)

        # Exclude eyes and inner mouth so iris, sclera, and lips are not recoloured
        pts = None
        if lms_q is not None and idx < len(lms_q) and lms_q[idx] is not None:
            pts = np.asarray(lms_q[idx], np.float32).reshape(-1, 2)
        if pts is not None and len(pts) >= 5:
            rx_e, ry_e = max(1, int(round(w * 0.12))), max(1, int(round(h * 0.08)))
            for ep in (pts[0], pts[1]):
                cv2.ellipse(feat, (int(round(ep[0])), int(round(ep[1]))), (rx_e, ry_e), 0, 0, 360, 0.9, -1)
            mcx = int(round(0.5 * (pts[3][0] + pts[4][0])))
            mcy = int(round(0.5 * (pts[3][1] + pts[4][1])))
            rx_m = max(1, int(round(max(abs(pts[4][0] - pts[3][0]) * 0.55, w * 0.14))))
            ry_m = max(1, int(round(h * 0.065)))
            cv2.ellipse(feat, (mcx, mcy), (rx_m, ry_m), 0, 0, 360, 0.75, -1)

    if feat.max() > 0:
        feat = cv2.GaussianBlur(feat, (0, 0), max(1.0, max(W, H) * 0.008))
        anat = anat * (1.0 - np.clip(feat, 0.0, 0.9))

    # Feather and snap to structural luminance edges (bridges across chromatic light split on cheeks)
    anat = cv2.GaussianBlur(anat, (0, 0), max(1.0, max(W, H) * 0.012))
    r = max(2, int(max(W, H) * 0.012))
    y_guide = np.repeat(np.clip(Y_eff / 255.0, 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32)
    anat = np.clip(guided_refine_rgb(y_guide, anat, r, 3e-3), 0.0, 1.0)
    return np.clip(np.maximum(m_base, anat), 0.0, 1.0).astype(np.float32)


def mixed_light_gain(rgb_q: np.ndarray, mask: np.ndarray, faces_q: list,
                     amount: float, lms_q: list | None = None) -> np.ndarray:
    """Dual-Illuminant Spatially-Varying Mixed-Light Portrait Solver (Hsu, Mertens, Paris & Durand,
    SIGGRAPH 2008 adapted to perceptual OKLab / OKLCh skin locus):
    Unifies split-lit faces (e.g. half chalk-white flash/window, half red/magenta ambient LED or tungsten)
    toward a single healthy golden-peach skin tone (~52 deg OKLCh) while preserving 100 % of pore texture."""
    H, W = rgb_q.shape[:2]
    if amount <= 1e-4:
        return np.zeros((H, W, 3), np.float32)
    m = mixed_light_face_mask(rgb_q, mask, faces_q, lms_q=lms_q)
    sk = m > 0.30
    if int(sk.sum()) < 25:
        return np.zeros((H, W, 3), np.float32)

    amt = float(np.clip(amount, 0.0, 1.0))
    cur = np.clip(rgb_q.astype(np.float32) / 255.0, 0.0, 1.0)
    lin = TN.srgb_to_lin(cur)
    ok = TN.lin_to_oklab(lin)
    L, a, b = ok[..., 0], ok[..., 1], ok[..., 2]
    C = np.hypot(a, b)
    h = np.arctan2(b, a)

    # 1. Estimate target healthy portrait skin chromaticity in OKLCh (~52 deg, C ~ 0.064)
    h_canon = np.float32(np.radians(52.0))
    c_canon = np.float32(0.064)
    dh = np.arctan2(np.sin(h - h_canon), np.cos(h - h_canon))
    w_healthy = (
        m
        * np.exp(-(dh ** 2) / (2.0 * (0.32 ** 2)))
        * np.exp(-((C - c_canon) ** 2) / (2.0 * (0.026 ** 2)))
        * np.clip((L - 0.35) / 0.20, 0.0, 1.0)
    ).astype(np.float32)

    w_sum = float(w_healthy.sum())
    m_sum = float(m.sum()) + 1e-5
    f_healthy = w_sum / m_sum
    a_canon = float(c_canon * np.cos(h_canon))
    b_canon = float(c_canon * np.sin(h_canon))
    if w_sum > 1.0:
        a_h = float((w_healthy * a).sum() / w_sum)
        b_h = float((w_healthy * b).sum() / w_sum)
    else:
        a_h, b_h = a_canon, b_canon

    # When the face is heavily polluted by split white/red lighting (low f_healthy), anchor strongly to 52 deg locus
    beta = float(np.clip(1.0 - 1.5 * f_healthy, 0.32, 1.0))
    a_target = np.float32((1.0 - beta) * a_h + beta * a_canon)
    b_target = np.float32((1.0 - beta) * b_h + beta * b_canon)

    # 2. Multi-scale spatially-varying illuminant field (separates local split lighting from fine pores)
    fw = float(np.median([f[2] for f in faces_q])) if faces_q else W / 3.0
    sig1 = max(1.2, fw / 18.0)
    sig2 = max(2.5, fw / 6.0)
    den1 = cv2.GaussianBlur(m, (0, 0), sig1) + 1e-4
    den2 = cv2.GaussianBlur(m, (0, 0), sig2) + 1e-4

    def _low_field(arr):
        f1 = cv2.GaussianBlur(arr * m, (0, 0), sig1) / den1
        f2 = cv2.GaussianBlur(arr * m, (0, 0), sig2) / den2
        return (0.72 * f1 + 0.28 * f2).astype(np.float32)

    L_low = _low_field(L)
    a_low = _low_field(a)
    b_low = _low_field(b)

    # 3. Unify both halves of the face in OKLab chromaticity and balance split-lighting luminance disparity
    a_new = a + amt * m * (a_target - a_low) - (0.65 * amt) * m * (a - a_low)
    b_new = b + amt * m * (b_target - b_low) - (0.65 * amt) * m * (b - b_low)
    L_med = float(np.clip(np.median(L[sk]), 0.56, 0.78))
    L_new = np.clip(L + (0.58 * amt) * m * np.clip(L_med - L_low, -0.22, 0.22), 0.0, 1.0)

    ok_new = np.stack([L_new, a_new, b_new], axis=-1).astype(np.float32)
    rgb_t = TN.lin_to_srgb(TN.oklab_to_lin_gamut(ok_new))

    g = np.where(cur > 0.02, rgb_t / np.maximum(cur, 0.02) - 1.0, 0.0)
    return np.clip(g * m[..., None], -0.68, 1.05).astype(np.float32)


def _safe_skin_weights(work_rgb: np.ndarray, skin_mask: np.ndarray,
                       face_w: np.ndarray, lms_w: np.ndarray) -> np.ndarray:
    """Compute clean skin confidence weights at canonical face resolution (FACE_WORK_PX = 256):
    excludes eyes, eyebrows, nostrils, mouth contour, and deep shadow/hair regions while keeping
    facial skin (including small acne blemishes) intact."""
    H, W = work_rgb.shape[:2]
    sk = np.clip(skin_mask.astype(np.float32), 0.0, 1.0)
    feat = np.zeros((H, W), np.uint8)
    if lms_w is not None and len(lms_w) >= 5:
        e0, e1 = lms_w[0], lms_w[1]
        d = float(np.linalg.norm(e1 - e0)) + 1e-6
        ex = (e1 - e0) / d
        ang = float(np.degrees(np.arctan2(ex[1], ex[0])))
        for p in (e0, e1):
            cv2.ellipse(feat, (int(p[0]), int(p[1])), (int(0.30 * d), int(0.17 * d)), ang, 0, 360, 1, -1)
        # Nostrils tip exclusion
        ns = lms_w[2]
        cv2.ellipse(feat, (int(ns[0]), int(ns[1])), (int(0.14 * d), int(0.09 * d)), ang, 0, 360, 1, -1)
        # Mouth exclusion
        mc = 0.5 * (lms_w[3] + lms_w[4])
        mw = max(float(np.linalg.norm(lms_w[4] - lms_w[3])), 0.55 * d)
        cv2.ellipse(feat, (int(mc[0]), int(mc[1])), (int(0.72 * mw), int(0.28 * mw)), ang, 0, 360, 1, -1)

    L = DNB.lightness(work_rgb)
    if (sk > 0.4).sum() > 40:
        med_L = float(np.median(L[sk > 0.4]))
        feat |= (L < med_L - 32.0).astype(np.uint8)
    feat_f = cv2.GaussianBlur(feat.astype(np.float32), (0, 0), 2.2)
    return np.clip(sk * (1.0 - feat_f), 0.0, 1.0).astype(np.float32)


def _heal_face_blemishes_work(work_rgb: np.ndarray, safe_w: np.ndarray, amount: float) -> np.ndarray:
    """Two-band pore-preserving blemish auto-heal at canonical face resolution (FACE_WORK_PX = 256).
    1. Separates high-frequency pores (sigma = 1.15 px) from the mid-frequency blemish band.
    2. Detects compact red/dark acne anomalies inside safe_w via local hemoglobin & luminance excess.
    3. Replaces the blemish mid-band with clean ring-interpolated surrounding skin while keeping
       100 % of the high-frequency pore layer."""
    if amount <= 1e-4:
        return np.zeros_like(work_rgb, dtype=np.float32)
    f = work_rgb.astype(np.float32)
    # Band 1: high-frequency pores (sigma <= 1.2 px) — never touched!
    sm = cv2.GaussianBlur(f, (0, 0), 1.2)
    # Band 2: surrounding clean base (sigma = 6.5 px)
    base = cv2.GaussianBlur(sm, (0, 0), 6.5)

    # Detect blemish cores: inflamed spots have depressed Green & Blue relative to surrounding base
    # (local hemoglobin spike + local luminance dip)
    dg = base[..., 1] - sm[..., 1]
    db = base[..., 2] - sm[..., 2]
    dr = base[..., 0] - sm[..., 0]
    # Redness/erythema spike + local dark spot score
    score = 0.55 * dg + 0.30 * db + 0.15 * dr + 0.35 * np.maximum(0.0, dg - 0.7 * dr)

    # Estimate robust background noise floor inside safe skin
    sk_pixels = score[safe_w > 0.35]
    if len(sk_pixels) < 60:
        return np.zeros_like(work_rgb, dtype=np.float32)
    med = float(np.median(sk_pixels))
    mad = float(np.median(np.abs(sk_pixels - med))) * 1.4826 + 0.8
    thr = max(3.2, med + 2.2 * mad)

    raw_b = np.clip((score - thr) / max(4.0, 1.8 * mad), 0.0, 1.0) * np.clip(safe_w, 0.0, 1.0)
    if float(raw_b.max()) < 0.05:
        return np.zeros_like(work_rgb, dtype=np.float32)

    # Keep only compact isolated blobs (reject long edges/folds via morphological opening-by-size check)
    b_mask = cv2.dilate(raw_b, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    b_mask = cv2.GaussianBlur(b_mask, (0, 0), 1.6)
    b_mask = np.clip(b_mask * 1.35, 0.0, 1.0) * np.clip(safe_w, 0.0, 1.0)

    # Clean surrounding skin interpolation excluding the blemish cores
    w_clean = np.clip(1.0 - b_mask * 0.95, 0.02, 1.0) * (safe_w + 0.05)
    den = cv2.GaussianBlur(w_clean, (0, 0), 5.5) + 1e-5
    healed_base = cv2.GaussianBlur(sm * w_clean[..., None], (0, 0), 5.5) / den[..., None]

    # Mid-frequency correction delta (adds directly to full-res image, leaving full-res pores 100 % intact)
    amt = float(np.clip(amount, 0.0, 1.0))
    delta_mid = amt * (b_mask[..., None]) * (healed_base - sm)
    return delta_mid.astype(np.float32)


def _enhance_eyes_and_teeth_work(work_rgb: np.ndarray, lms_work: np.ndarray, amount: float) -> np.ndarray:
    """Anatomical iris, sclera, and natural teeth enhancement in canonical face scale using 5-point landmarks
    lms_work: [right_eye, left_eye, nose, right_mouth, left_mouth]. Returns additive RGB delta."""
    if amount <= 1e-4 or lms_work is None or len(lms_work) < 5:
        return np.zeros_like(work_rgb, dtype=np.float32)
    H, W = work_rgb.shape[:2]
    f = work_rgb.astype(np.float32)
    delta = np.zeros_like(f, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    amt = float(np.clip(amount, 0.0, 1.0))

    p_re, p_le, _, p_rm, p_lm = lms_work[:5]
    d_eye = float(np.hypot(p_le[0] - p_re[0], p_le[1] - p_re[1]))
    if d_eye < 12.0:
        return delta

    r_eye = 0.175 * d_eye
    lum = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    sm_eye = cv2.GaussianBlur(f, (0, 0), 1.6)
    hf_eye = f - sm_eye

    # 1. Iris clarity & radiance + Sclera redness clean-up for both eyes
    for ex, ey in (p_re, p_le):
        dist = np.hypot(xx - float(ex), yy - float(ey))
        # Iris annulus: between pupil (0.22 * r_eye) and limbus (0.82 * r_eye)
        w_iris = np.exp(-((dist - 0.52 * r_eye) / (0.26 * r_eye)) ** 4)
        # Protect dark pupil (lum < 28) and pure specular catchlights (lum > 230)
        w_tone = np.clip((lum - 24.0) / 22.0, 0.0, 1.0) * np.clip((235.0 - lum) / 25.0, 0.0, 1.0)
        w_i = (w_iris * w_tone)[..., None]
        # Boost iris fiber micro-contrast + subtle radial radiance lift
        delta += amt * w_i * (0.75 * hf_eye + 0.12 * f)

        # Sclera (almond region around iris: 0.75 * r_eye .. 1.35 * r_eye horizontally)
        dx = (xx - float(ex)) / (1.35 * r_eye)
        dy = (yy - float(ey)) / (0.68 * r_eye)
        r_alm = np.hypot(dx, dy)
        w_sclera = np.clip(1.0 - np.maximum(0.0, r_alm - 0.75) / 0.35, 0.0, 1.0) * np.clip((dist - 0.72 * r_eye) / (0.20 * r_eye), 0.0, 1.0)
        # Only on moderately bright, low-to-medium chroma sclera pixels
        w_sclera *= np.clip((lum - 90.0) / 40.0, 0.0, 1.0)
        red_excess = np.maximum(0.0, f[..., 0] - 0.5 * (f[..., 1] + f[..., 2]))
        delta[..., 0] -= amt * 0.35 * w_sclera * red_excess
        delta[..., 1] += amt * 0.15 * w_sclera * red_excess + amt * 3.5 * w_sclera
        delta[..., 2] += amt * 0.20 * w_sclera * red_excess + amt * 4.0 * w_sclera

    # 2. Natural teeth whitening inside mouth contour (between right_mouth p_rm and left_mouth p_lm)
    mc_x = 0.5 * float(p_rm[0] + p_lm[0])
    mc_y = 0.5 * float(p_rm[1] + p_lm[1])
    mw = max(float(np.hypot(p_lm[0] - p_rm[0], p_lm[1] - p_rm[1])), 0.55 * d_eye)
    ang = float(np.arctan2(p_lm[1] - p_rm[1], p_lm[0] - p_rm[0]))
    ca, sa = np.cos(ang), np.sin(ang)
    ux = ((xx - mc_x) * ca + (yy - mc_y) * sa) / (0.58 * mw)
    uy = (-(xx - mc_x) * sa + (yy - mc_y) * ca) / (0.28 * mw)
    w_mouth = np.exp(-(ux * ux + uy * uy) ** 2)

    # Distinguish teeth from red lips and warm skin:
    # Teeth have high Green/Red ratio (G/R > 0.84, whereas lips/skin have R >> G) and bright luminance
    gr_ratio = f[..., 1] / np.maximum(f[..., 0], 1.0)
    w_not_lip = np.clip((gr_ratio - 0.83) / 0.08, 0.0, 1.0)
    w_bright = np.clip((lum - 115.0) / 35.0, 0.0, 1.0)
    w_teeth = w_mouth * w_not_lip * w_bright

    # Yellow cast in teeth: (R + G)/2 > B
    yellow_cast = np.maximum(0.0, 0.5 * (f[..., 0] + f[..., 1]) - f[..., 2])
    delta[..., 2] += amt * 0.62 * w_teeth * yellow_cast + amt * 6.0 * w_teeth
    delta[..., 1] += amt * 0.12 * w_teeth * yellow_cast + amt * 5.0 * w_teeth
    delta[..., 0] -= amt * 0.15 * w_teeth * yellow_cast - amt * 4.0 * w_teeth
    return delta.astype(np.float32)


def apply_portrait_details(rgb: np.ndarray, mask_q: np.ndarray, faces_px: list,
                           lms_px: list | None = None, blemish: float = 0.0,
                           eyes_teeth: float = 0.0) -> np.ndarray:
    """Apply resolution-independent Two-Band Blemish Auto-Heal and Anatomical Eyes/Teeth enhancement
    on face crops at canonical scale (FACE_WORK_PX = 256), preserving 100 % of full-resolution pores."""
    if (blemish <= 1e-4 and eyes_teeth <= 1e-4) or not faces_px:
        return rgb
    H, W = rgb.shape[:2]
    out = rgb.copy()
    lms_list = lms_px if (lms_px and len(lms_px) == len(faces_px)) else [None] * len(faces_px)

    for face, lms in zip(faces_px, lms_list):
        fx, fy, fw, fh = [float(v) for v in face]
        if fw < 16 or fh < 16:
            continue
        pad = 0.35 * max(fw, fh)
        x0 = max(0, int(round(fx - pad)))
        y0 = max(0, int(round(fy - pad)))
        x1 = min(W, int(round(fx + fw + pad)))
        y1 = min(H, int(round(fy + fh + pad)))
        cw, ch = x1 - x0, y1 - y0
        if cw < 16 or ch < 16:
            continue

        crop = out[y0:y1, x0:x1]
        sc = FACE_WORK_PX / max(fw, 1.0)
        ww = max(16, int(round(cw * sc)))
        wh = max(16, int(round(ch * sc)))
        work = crop if (wh, ww) == (ch, cw) else cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)

        face_w = np.array([(fx - x0) * (ww / cw), (fy - y0) * (wh / ch), fw * (ww / cw), fh * (wh / ch)], np.float32)
        lms_w = None
        if lms is not None:
            lms_arr = np.asarray(lms, np.float32).reshape(-1, 2)
            lms_w = np.column_stack([(lms_arr[:, 0] - x0) * (ww / cw), (lms_arr[:, 1] - y0) * (wh / ch)]).astype(np.float32)
        else:
            # Canonical 5-point landmarks inside face box if detector landmarks were omitted
            fxw, fyw, fww, fhw = face_w
            lms_w = np.array([
                [fxw + 0.31 * fww, fyw + 0.38 * fhw],
                [fxw + 0.69 * fww, fyw + 0.38 * fhw],
                [fxw + 0.50 * fww, fyw + 0.58 * fhw],
                [fxw + 0.35 * fww, fyw + 0.76 * fhw],
                [fxw + 0.65 * fww, fyw + 0.76 * fhw],
            ], np.float32)

        delta_w = np.zeros((wh, ww, 3), np.float32)
        if blemish > 1e-4:
            mq_crop = cv2.resize(mask_q.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1, x0:x1]
            mq_w = cv2.resize(mq_crop, (ww, wh), interpolation=cv2.INTER_LINEAR)
            safe_w = _safe_skin_weights(work, mq_w, face_w, lms_w)
            delta_w += _heal_face_blemishes_work(work, safe_w, blemish)

        if eyes_teeth > 1e-4:
            delta_w += _enhance_eyes_and_teeth_work(work, lms_w, eyes_teeth)

        # Feather crop border to 0 so no seam can ever appear
        by = np.minimum(np.arange(wh, dtype=np.float32), np.arange(wh - 1, -1, -1, dtype=np.float32))
        bx = np.minimum(np.arange(ww, dtype=np.float32), np.arange(ww - 1, -1, -1, dtype=np.float32))
        b_feather = np.clip(np.minimum(by[:, None], bx[None, :]) / max(4.0, 0.06 * min(wh, ww)), 0.0, 1.0)[..., None]
        delta_w *= b_feather

        delta_full = delta_w if (wh, ww) == (ch, cw) else cv2.resize(delta_w, (cw, ch), interpolation=cv2.INTER_CUBIC)
        out[y0:y1, x0:x1] = np.clip(crop.astype(np.float32) + delta_full + 0.5, 0, 255).astype(np.uint8)

    return out


def apply_heal_strokes(rgb: np.ndarray, heal_strokes: list | None) -> np.ndarray:
    """Non-destructive Frequency-Separated Pore-Preserving Healing Brush ('Pędzel Usuń / Popraw', Q):
    For each stroke (normalised coordinates & radius):
      1. Extracts a tight padded ROI around the stroke (O(r^2) memory & time, <3 ms per spot on 24 Mpx).
      2. Separates the ROI into a low-frequency colour/shading base and high-frequency pore/grain texture.
      3. Inpaints the low-frequency base seamlessly via Telea fast-marching (cv2.INPAINT_TELEA).
      4. Transfers real high-frequency pore/grain texture from the best-matching clean surrounding patch
         so the healed spot never looks like a blurred plastic patch."""
    if not heal_strokes:
        return rgb
    H, W = rgb.shape[:2]
    out = rgb.copy()

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1))
    for st in heal_strokes:
        pts = np.asarray(st.get("points", []), np.float32)
        if pts.size == 0:
            continue
        pts = pts.reshape(-1, 2)
        r = max(2, int(round(float(st.get("size", 0.018)) * W)))
        px = np.clip(np.round(pts[:, 0] * W).astype(np.int32), 0, W - 1)
        py = np.clip(np.round(pts[:, 1] * H).astype(np.int32), 0, H - 1)

        pad = max(24, int(round(3.8 * r)))
        x0 = max(0, int(px.min()) - pad)
        y0 = max(0, int(py.min()) - pad)
        x1 = min(W, int(px.max()) + pad + 1)
        y1 = min(H, int(py.max()) + pad + 1)
        cw, ch = x1 - x0, y1 - y0
        if cw < 8 or ch < 8:
            continue

        m_bin = np.zeros((ch, cw), np.uint8)
        loc_pts = np.column_stack([px - x0, py - y0]).astype(np.int32)
        if len(loc_pts) == 1:
            cv2.circle(m_bin, (int(loc_pts[0, 0]), int(loc_pts[0, 1])), r, 255, -1)
        else:
            cv2.polylines(m_bin, [loc_pts.reshape(-1, 1, 2)], False, 255, max(2, int(round(2.0 * r))))
            for lp in (loc_pts[0], loc_pts[-1]):
                cv2.circle(m_bin, (int(lp[0]), int(lp[1])), r, 255, -1)

        kd = max(3, int(round(0.42 * r)) | 1)
        m_dil = cv2.dilate(m_bin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
        if int((m_dil > 0).sum()) == 0:
            continue

        crop = out[y0:y1, x0:x1]
        crop_f = crop.astype(np.float32)
        sig = float(np.clip(0.34 * r, 1.35, 6.5))
        low = cv2.GaussianBlur(crop_f, (0, 0), sig)
        high = crop_f - low

        # 1. Inpaint low-frequency colour & shading across the dilated mask
        low_u8 = np.clip(low + 0.5, 0, 255).astype(np.uint8)
        healed_low = cv2.inpaint(low_u8, m_dil, max(3, int(round(0.65 * r))), cv2.INPAINT_TELEA).astype(np.float32)

        # 2. Find cleanest matching surrounding offset to donor real pore/grain high-frequency texture
        target_mask = m_dil > 0
        target_mean = healed_low[target_mask].mean(axis=0)
        best_high = None
        best_cost = float("inf")
        shift_d = max(r * 2 + 3, int(round(2.35 * r)))
        ys_idx, xs_idx = np.nonzero(target_mask)

        for dx_u, dy_u in dirs:
            sx = xs_idx + dx_u * shift_d
            sy = ys_idx + dy_u * shift_d
            if sx.min() < 1 or sx.max() >= cw - 1 or sy.min() < 1 or sy.max() >= ch - 1:
                continue
            if np.any(m_dil[sy, sx] > 0):
                continue
            cand_low = low[sy, sx]
            # Prefer surrounding skin/backdrop with matching base colour and natural micro-texture (no sharp edges)
            col_diff = float(np.linalg.norm(cand_low.mean(axis=0) - target_mean))
            edge_pen = float(np.std(cand_low)) * 0.35
            cost = col_diff + edge_pen
            if cost < best_cost:
                best_cost = cost
                best_high = high[sy, sx]

        healed_high = high.copy()
        if best_high is not None:
            healed_high[ys_idx, xs_idx] = best_high
        else:
            clean_mask = ~target_mask
            if int(clean_mask.sum()) >= 16:
                clean_vals = high[clean_mask]
                rng_idx = np.arange(len(ys_idx)) % len(clean_vals)
                healed_high[ys_idx, xs_idx] = clean_vals[rng_idx]

        alpha = cv2.GaussianBlur(target_mask.astype(np.float32), (0, 0), max(1.2, 0.32 * r))[..., None]
        alpha = np.clip(alpha * 1.18, 0.0, 1.0)
        blended = crop_f * (1.0 - alpha) + (healed_low + healed_high) * alpha
        out[y0:y1, x0:x1] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)

    return out

