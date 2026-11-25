"""Dodge & Burn v2: halo-free skin evening + soft facial contour light, luminance only.

Why v2 (0.1.4 feedback "przy większych wartościach wygląda tragicznie"): v1 removed a fixed Gaussian
frequency band of L* inside a blurred skin mask - the band contained the face's own modelling (flat face),
dark frame/brow pixels entered the skin averages (rims at glasses, eyes, lips), the strength was linear and
unbounded, and lowering L* with a*b* kept made darkened skin orange.

v2, all analysis on a crop resampled so the face is DB_FACE_PX wide (identical for preview and export):
1. *Clean skin weights*: the app's skin mask minus anything that deviates strongly from the local skin level
   (glasses frames, eyes, brows, lips, nostrils, specular spots), minus landmark-placed eye/mouth areas.
2. *Blotch map* = skin-normalised small blur (pores out) minus an *edge-preserving base* of the same skin:
   a weighted guided filter (He, Sun, Tang 2010) - real shading edges (nose, jaw) have a variance far above
   eps and stay in the base, low-contrast blotches do not. Candidates kept for the quality-gate comparison:
   "gauss" (v1-like Gaussian base) and "wls" (weighted least squares, Farbman et al. 2008, CG solve).
3. Redescending weight x/(1+(x/T)^2): large deviations (moles, real shadows) are barely touched.
4. Contour light: smooth landmark-placed shapes (dodge forehead / nose bridge / cheekbones / chin, burn below
   the cheekbones and along the jaw), amplitude a few L*.
5. Total field soft-capped by CAP_L*tanh(x/CAP_L); it is smooth, so pores and fine texture are never touched.
6. Full resolution: the skin mask is refined with a fast guided filter (He & Sun 2015) using the image's own
   luminance as guide, so the field stops exactly at the skin's edges (no glow into hair, glasses, background);
   the change is applied as a luminance-only gain in linear light (chromaticity kept: no orange skin), never
   pushing a channel into clipping.
Ideas from the papers above only; no code copied."""
from __future__ import annotations
import numpy as np, cv2

DB_FACE_PX = 256        # analysis scale: face width in px
CAP_L = 10.0            # soft cap of |dL*|
BLOTCH_T = 10.0         # L*: redescending scale of the evening term
EVEN_K = 1.0            # share of the blotch map removed at 100 % (before the redescending weight)
CONTOUR_L = (4.0, 5.0)  # dodge, burn amplitude (L*) at 100 %
FEATURE_DEV = 10.0      # L*: |L - local skin level| above this is a feature, not skin
DARK_L = 30.0           # L* below the face-centre skin level: hair, beard, deep background (dark brown hair passes
                        # the app's skin colour test, and a wide dark area is its own "local level")
_PRE = 80.0             # pre-blur (pores out) = face width / _PRE
_LW = np.array([0.2126, 0.7152, 0.0722], np.float32)

# ---------------------------------------------------------------- small helpers
def _nblur(x, w, s):
    return cv2.GaussianBlur(x*w, (0, 0), s)/(cv2.GaussianBlur(w, (0, 0), s) + 1e-6)

def _box(x, r): return cv2.boxFilter(x, -1, (2*r+1, 2*r+1))

def _srgb_to_lin(x): return np.where(x <= 0.04045, x/12.92, ((x + 0.055)/1.055)**2.4).astype(np.float32)
def _lin_to_srgb(x): return np.where(x <= 0.0031308, x*12.92, 1.055*np.power(np.maximum(x, 0), 1/2.4) - 0.055).astype(np.float32)
def _Y_to_L(Y): return np.where(Y > 0.008856, 116*np.cbrt(Y) - 16, 903.3*Y).astype(np.float32)
def _L_to_Y(L): return np.where(L > 8.0, ((L + 16)/116)**3, L/903.3).astype(np.float32)

_LUT_LIN = None
def lightness(rgb8):
    """CIE L* (D65) of an sRGB uint8 image, float32 (per-channel linearisation through a 256-entry table)."""
    global _LUT_LIN
    if _LUT_LIN is None: _LUT_LIN = _srgb_to_lin(np.arange(256, dtype=np.float32)/255)
    Y = _LUT_LIN[rgb8[..., 0]]*_LW[0]; Y += _LUT_LIN[rgb8[..., 1]]*_LW[1]; Y += _LUT_LIN[rgb8[..., 2]]*_LW[2]
    return _Y_to_L(Y)

from .color import fast_guided_rgb

# ---------------------------------------------------------------- edge-preserving bases (candidates)
def weighted_guided(I, w, r, eps):
    """Self-guided filter with per-pixel data weights (normalised box means in both passes): non-skin
    pixels neither enter local statistics nor bias the averaged linear coefficients (a, b)."""
    W = _box(w, r) + 1e-6
    m = _box(I*w, r)/W; v = _box(I*I*w, r)/W - m*m
    a = v/(v + eps); b = m - a*m
    return (_box(a*w, r)/W)*I + (_box(b*w, r)/W)

def wls(g, w, lam, alpha=1.2, eps=1e-4, iters=250):
    """Weighted least squares smoothing (Farbman, Fattal, Lischinski, Szeliski 2008), data weight w:
    min sum w (u-g)^2 + lam sum a_x u_x^2 + a_y u_y^2, a = 1/(|grad g/100|^alpha + eps). Jacobi-preconditioned CG."""
    gx = np.abs(np.diff(g, axis=1))/100; gy = np.abs(np.diff(g, axis=0))/100
    ax = lam/(gx**alpha + eps)*1e-4; ay = lam/(gy**alpha + eps)*1e-4
    d = w + 1e-3
    def A(u):
        out = d*u
        fx = ax*np.diff(u, axis=1); fy = ay*np.diff(u, axis=0)
        out[:, :-1] -= fx; out[:, 1:] += fx; out[:-1, :] -= fy; out[1:, :] += fy
        return out
    diag = d.copy(); diag[:, :-1] += ax; diag[:, 1:] += ax; diag[:-1, :] += ay; diag[1:, :] += ay
    b = d*g; u = g.copy(); r = b - A(u); z = r/diag; p = z.copy(); rz = float((r*z).sum())
    for _ in range(iters):
        Ap = A(p); al = rz/max(float((p*Ap).sum()), 1e-12)
        u += al*p; r -= al*Ap
        if float(np.abs(r).max()) < 1e-3: break
        z = r/diag; rz2 = float((r*z).sum()); p = z + (rz2/max(rz, 1e-12))*p; rz = rz2
    return u.astype(np.float32)

def base_of(Lf, wc, fw, method):
    if method == "gauss": return _nblur(Lf, wc, fw/8)
    if method == "wls": return wls(Lf, wc, lam=(fw/12)**2)
    return weighted_guided(Lf, wc, max(2, int(fw/12)), 6.0**2)

# ---------------------------------------------------------------- landmarks
def bbox_landmarks(face):
    """Typical YuNet layout from a face box (x, y, w, h): right eye, left eye, nose, mouth right, mouth left."""
    x, y, w, h = face
    return np.float32([[x+0.31*w, y+0.40*h], [x+0.69*w, y+0.40*h], [x+0.5*w, y+0.58*h], [x+0.36*w, y+0.76*h], [x+0.64*w, y+0.76*h]])

def _face_frame(lm):
    e0, e1 = lm[0], lm[1]; c = (e0 + e1)/2; d = float(np.linalg.norm(e1 - e0)) + 1e-6
    ex = (e1 - e0)/d; ey = np.float32([-ex[1], ex[0]])            # ey points down the face
    if np.dot(ey, lm[2] - c) < 0: ey = -ey
    return c, d, ex, ey

def _shapes(shape, lm):
    """Contour light (+) and shadow (-) templates, each peak 1, in the frame of one face."""
    h, w = shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, d, ex, ey = _face_frame(lm)
    U = (xx - c[0])*ex[0] + (yy - c[1])*ex[1]; V = (xx - c[0])*ey[0] + (yy - c[1])*ey[1]
    g = lambda u, v, su, sv: np.exp(-0.5*(((U - u*d)/(su*d))**2 + ((V - v*d)/(sv*d))**2))
    mouth_v = float(np.dot((lm[3] + lm[4])/2 - c, ey))/d
    dodge = (g(0, -0.80, 0.55, 0.28) + 0.8*g(0, 0.35, 0.13, 0.40) + 0.6*g(-0.55, 0.45, 0.22, 0.16)
             + 0.6*g(0.55, 0.45, 0.22, 0.16) + 0.7*g(0, mouth_v + 0.55, 0.28, 0.18))
    burn = (g(-0.85, 0.95, 0.20, 0.32) + g(0.85, 0.95, 0.20, 0.32) + 0.7*g(-0.95, 1.45, 0.25, 0.30) + 0.7*g(0.95, 1.45, 0.25, 0.30))
    return np.minimum(dodge, 1.0), np.minimum(burn, 1.0)

# ---------------------------------------------------------------- the field
def dnb_field(L, skin, faces, lms, strength, method="guided"):
    """dL* (float32, same size as L) for L* and the soft skin mask at analysis scale (face ~ DB_FACE_PX).
    faces: [(x, y, w, h)], lms: list of 5x2 arrays or None.
    Returns (field, mask): the smooth, soft-capped dL* defined everywhere, and the skin mask it applies to
    (skin minus features, not dilated) - the caller refines that mask's edges at its own resolution."""
    if strength <= 0 or not len(faces): return np.zeros_like(L), np.zeros_like(L)
    fw = float(np.median([f[2] for f in faces]))
    lms = [np.asarray(l, np.float32) for l in lms] if lms else [bbox_landmarks(f) for f in faces]
    sk = (skin > 0.5).astype(np.float32)
    # 1. clean skin: drop features (dark frames, eyes, brows, lips, specular) by their deviation from the skin level
    level = _nblur(L, sk, fw/14)
    feat0 = (np.abs(L - level) > FEATURE_DEV).astype(np.uint8)
    cen = np.zeros(L.shape, np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(cen, (int(x + 0.25*w), int(y + 0.5*h)), (int(x + 0.75*w), int(y + 0.8*h)), 1, -1)
    ref = L[(cen > 0) & (sk > 0)]
    if ref.size > 20: feat0 |= (L < np.median(ref) - DARK_L).astype(np.uint8)
    feat = feat0.copy()
    for lm in lms:
        c, d, ex, ey = _face_frame(lm)
        for p in lm[:2]:
            cv2.ellipse(feat, (int(p[0]), int(p[1])), (int(0.32*d), int(0.18*d)), float(np.degrees(np.arctan2(ex[1], ex[0]))), 0, 360, 1, -1)
        mc = (lm[3] + lm[4])/2; mw = float(np.linalg.norm(lm[4] - lm[3]))
        cv2.ellipse(feat, (int(mc[0]), int(mc[1])), (int(0.75*mw), int(0.28*mw)), float(np.degrees(np.arctan2(ex[1], ex[0]))), 0, 360, 1, -1)
    k = max(3, int(fw/40)) | 1
    feat = cv2.dilate(feat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
    wc = sk*(1 - feat)
    if wc.sum() < 50: return np.zeros_like(L), wc
    # 2. blotch map on clean skin: small blur (pores out) - edge-preserving base (face modelling stays)
    Lf = _nblur(L, wc, fw/_PRE)                        # also fills features with the surrounding skin level
    base = base_of(Lf, wc + 1e-3, fw, method)
    x = Lf - base
    even = -EVEN_K*x/(1 + (x/BLOTCH_T)**2)              # 3. redescending: large real features barely touched
    fill = _nblur(even, wc, fw/40)                      # across features: the surrounding skin's correction (no rims)
    t = np.clip(cv2.GaussianBlur(wc, (0, 0), 1.0), 0, 1)
    even = t*even + (1 - t)*fill
    # 4. contour light, kept off features
    dodge = np.zeros_like(L); burn = np.zeros_like(L)
    for lm in lms:
        a, b = _shapes(L.shape, lm); dodge = np.maximum(dodge, a); burn = np.maximum(burn, b)
    contour = CONTOUR_L[0]*dodge - CONTOUR_L[1]*burn
    raw = strength*(even + contour)
    field = CAP_L*np.tanh(raw/CAP_L)                    # 5. soft cap
    mask = np.clip(np.minimum(skin, 1 - feat0), 0, 1)   # features themselves never change
    return field.astype(np.float32), mask.astype(np.float32)

# ---------------------------------------------------------------- full-resolution application
def fast_guided(I, p, r, eps, s):
    """Fast guided filter (He & Sun 2015): coefficients at 1/s resolution, applied with the full-res guide I."""
    h, w = I.shape
    if s > 1:
        Is = cv2.resize(I, (max(2, round(w/s)), max(2, round(h/s))), interpolation=cv2.INTER_AREA)
        ps = cv2.resize(p, (Is.shape[1], Is.shape[0]), interpolation=cv2.INTER_AREA); rs = max(1, round(r/s))
    else:
        Is, ps, rs = I, p, r
    mI, mp = _box(Is, rs), _box(ps, rs)
    a = (_box(Is*ps, rs) - mI*mp)/(_box(Is*Is, rs) - mI*mI + eps); b = mp - a*mI
    a, b = _box(a, rs), _box(b, rs)
    if s > 1:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR); b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    return a*I + b

def apply_dL(rgb8, dL, rows=256):
    """Luminance-only change in linear light: every channel scaled by Y'/Y (chromaticity kept), never pushing a
    channel above 0.98 by brightening. Pixels with |dL| < 1e-3 are returned bit-identical. Works in row strips
    (24 Mpx originals: bounded memory)."""
    out = rgb8.copy()
    for y in range(0, rgb8.shape[0], rows):
        d = dL[y:y+rows]; act = np.abs(d) > 1e-3
        if not act.any(): continue
        lin = _srgb_to_lin(rgb8[y:y+rows][act].astype(np.float32)/255)
        Y = lin @ _LW; L = _Y_to_L(Y)
        g = _L_to_Y(np.clip(L + d[act], 0, 100))/np.maximum(Y, 1e-6)
        mx = lin.max(1)
        g = np.where(g > 1, np.minimum(g, np.maximum(1.0, 0.98/np.maximum(mx, 1e-6))), g)
        blk = out[y:y+rows]; blk[act] = np.clip(_lin_to_srgb(lin*g[:, None])*255 + 0.5, 0, 255).astype(np.uint8)
    return out

def dodge_burn_v2(rgb, skin, faces_px, lms_px, strength, method="guided"):
    """Auto D&B on uint8 sRGB `rgb` (any resolution). skin: soft 0..1 mask (any size, resized here);
    faces_px: [(x, y, w, h)] in rgb pixels; lms_px: list of 5x2 landmark arrays in rgb pixels, or None.
    One region covers all faces; the change is feathered to zero at its border."""
    if strength <= 0 or not len(faces_px): return rgb
    H, W = rgb.shape[:2]
    x0 = max(0, int(min(x - 0.6*w for x, y, w, h in faces_px))); y0 = max(0, int(min(y - 0.4*h for x, y, w, h in faces_px)))
    x1 = min(W, int(max(x + 1.6*w for x, y, w, h in faces_px))); y1 = min(H, int(max(y + 1.9*h for x, y, w, h in faces_px)))
    if x1 - x0 < 8 or y1 - y0 < 8: return rgb
    fw = float(np.median([f[2] for f in faces_px]))
    sm = skin if skin.shape[:2] == (H, W) else cv2.resize(skin.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    crop = rgb[y0:y1, x0:x1]; mc = sm[y0:y1, x0:x1].astype(np.float32)
    k = min(8.0, DB_FACE_PX/max(fw, 1.0))                       # analysis scale: face = DB_FACE_PX (preview == export)
    aw, ah = max(8, round((x1-x0)*k)), max(8, round((y1-y0)*k))
    interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
    La = lightness(cv2.resize(crop, (aw, ah), interpolation=interp))
    ma = cv2.resize(mc, (aw, ah), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    kx, ky = aw/(x1-x0), ah/(y1-y0)
    fa = [((x-x0)*kx, (y-y0)*ky, w*kx, h*ky) for x, y, w, h in faces_px]
    la = None if not lms_px else [(np.asarray(l, np.float32) - [x0, y0])*[kx, ky] for l in lms_px]
    fa_, ma_ = dnb_field(La, ma, fa, la, strength, method)
    if not np.any(fa_*ma_): return rgb
    # back to the crop's resolution: the smooth field as is, the mask's edges snapped to the image's own RGB
    # boundaries (fast 3-channel guided filter, He & Sun 2015), so nothing glows into hair, frames or background
    field = cv2.resize(fa_, (x1-x0, y1-y0), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(ma_, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    s = max(1.0, (x1-x0)/aw)
    mask = np.clip(fast_guided_rgb(crop.astype(np.float32)/255.0, mask, max(2, round(3*s)), 1e-3, s), 0, 1)
    dL = field*mask
    m = max(4, int(0.15*fw))                                    # feather at the region border (not at image borders)
    wy = np.clip(np.minimum(np.arange(y1-y0), np.arange(y1-y0)[::-1])/m, 0, 1)
    wx = np.clip(np.minimum(np.arange(x1-x0), np.arange(x1-x0)[::-1])/m, 0, 1)
    if y0 == 0: wy[:m] = 1
    if y1 == H: wy[-m:] = 1
    if x0 == 0: wx[:m] = 1
    if x1 == W: wx[-m:] = 1
    dL *= (wy[:, None]*wx[None, :]).astype(np.float32)
    out = rgb.copy(); out[y0:y1, x0:x1] = apply_dL(crop, dL)
    return out
