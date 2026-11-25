"""Photographic tone & colour engine (published methods, MIT/public-domain maths):

- OKLab perceptual colour space (Björn Ottosson 2020, public domain / MIT): uniform hue linearity
  (no CIELAB blue->purple shift, natural skin tones under contrast and saturation changes).
- Exposure (EV) in linear light (x 2^EV) with a chromaticity-preserving filmic highlight shoulder
  (ratio-preserving compression above knee=0.82 + smooth highlight roll toward white).
- Perceptual tone controls in OKLab L: Contrast (monotonic S-curve), Highlights/Shadows (upper/lower
  mid-tones), Whites/Blacks (endpoints), applied with proportional chroma and hue-preserving gamut mapping.
- OKLab Vibrance & Saturation with skin-hue protection.
- Clarity: halo-free edge-preserving local mid-tone contrast via guided filter (He, Sun, Tang 2010),
  skin-protected for positive punch and soft portrait glow for negative values."""
from __future__ import annotations
import numpy as np, cv2

_Y709 = np.array([0.2126, 0.7152, 0.0722], np.float32)

# OKLab matrices (Ottosson 2020, https://bottosson.github.io/posts/oklab/)
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
], np.float32)

_M2 = np.array([
    [+0.2104542553, +0.7936177850, -0.0040720468],
    [+1.9779984951, -2.4285922050, +0.4505937099],
    [+0.0259040371, +0.7827717662, -0.8086757660],
], np.float32)

_M2_INV = np.array([
    [1.0, +0.3963377774, +0.2158037573],
    [1.0, -0.1055613458, -0.0638541728],
    [1.0, -0.0894841775, -1.2914855480],
], np.float32)

_M1_INV = np.array([
    [+4.0767416621, -3.3077115913, +0.2309699292],
    [-1.2684380046, +2.6097574011, -0.3413193965],
    [-0.0041960863, -0.7034186147, +1.7076147010],
], np.float32)


def srgb_to_lin(rgb01):
    x = np.clip(np.asarray(rgb01, np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x/12.92, np.power((x + 0.055)/1.055, 2.4)).astype(np.float32)


def lin_to_srgb(lin):
    x = np.clip(np.asarray(lin, np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, x*12.92, 1.055*np.power(x, 1.0/2.4) - 0.055).astype(np.float32)


def lin_to_oklab(lin):
    """Linear sRGB (..., 3) -> OKLab (L in 0..1, a, b)."""
    lms = np.asarray(lin, np.float32) @ _M1.T
    lms_ = np.cbrt(lms)
    return (lms_ @ _M2.T).astype(np.float32)


def oklab_to_lin(ok):
    """OKLab (..., 3) -> linear sRGB (unclamped)."""
    lms_ = np.asarray(ok, np.float32) @ _M2_INV.T
    lms = lms_*lms_*lms_
    return (lms @ _M1_INV.T).astype(np.float32)


def gamut_map_lin(lin):
    """Compress out-of-gamut linear RGB toward neutral luminance Y (preserves luminance)."""
    lin = np.asarray(lin, np.float32)
    Y = np.clip(lin @ _Y709, 0.0, 1.0)[..., None]
    d = lin - Y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(d > 1e-6, (1.0 - Y)/d, np.inf)
        t_lo = np.where(d < -1e-6, -Y/d, np.inf)
    t = np.clip(np.minimum(t_hi, t_lo).min(axis=-1, keepdims=True), 0.0, 1.0)
    return np.clip(Y + d*t, 0.0, 1.0).astype(np.float32)


def oklab_to_lin_gamut(ok):
    """OKLab -> linear sRGB with constant-hue, constant-lightness OKLab chroma clipping (Ottosson 2021).
    Only pixels outside [0, 1]^3 run 8-step bisection on chroma scale t in [0, 1]: 0 deg OKLab hue drift."""
    lin = oklab_to_lin(ok)
    oob = (lin.min(axis=-1) < -1e-5) | (lin.max(axis=-1) > 1.0 + 1e-5)
    if not np.any(oob):
        return np.clip(lin, 0.0, 1.0).astype(np.float32)
    o = ok[oob]
    L = np.clip(o[:, 0:1], 0.0, 1.0); ab = o[:, 1:3]
    lo = np.zeros((len(o), 1), np.float32); hi = np.ones((len(o), 1), np.float32)
    for _ in range(9):
        mid = 0.5*(lo + hi)
        cand = oklab_to_lin(np.concatenate([L, ab*mid], axis=-1))
        inside = (cand.min(axis=-1, keepdims=True) >= 0.0) & (cand.max(axis=-1, keepdims=True) <= 1.0)
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    fixed = np.clip(oklab_to_lin(np.concatenate([L, ab*lo], axis=-1)), 0.0, 1.0)
    out = np.clip(lin, 0.0, 1.0).astype(np.float32)
    out[oob] = fixed
    return out


def apply_exposure_lin(lin, ev: float, knee: float = 0.82):
    """Linear-light exposure (x 2^EV) with chromaticity-preserving filmic highlight shoulder."""
    if not ev:
        return lin
    x = np.asarray(lin, np.float32) * np.float32(2.0 ** float(ev))
    if ev < 0.0:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    mx = x.max(axis=-1, keepdims=True)
    over = np.maximum(mx - knee, 0.0)
    span = 1.0 - knee
    mx_out = np.where(mx > knee, knee + span*(1.0 - np.exp(-over/span)), mx)
    # preserve exact RGB ratios up to shoulder, then roll saturated blown speculars gently toward white
    scaled = x * (mx_out / np.maximum(mx, 1e-6))
    desat = 1.0 - np.exp(-np.maximum(mx - 1.0, 0.0)*0.45)
    out = scaled*(1.0 - desat) + mx_out*desat
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_tone_lin(lin, contrast: float = 0.0, highlights: float = 0.0, shadows: float = 0.0,
                   whites: float = 0.0, blacks: float = 0.0):
    """Perceptual tone controls in OKLab L: monotonic S-curve contrast around 18 % grey (L=0.58),
    mid-shadow/mid-highlight recovery, and black/white endpoint controls. Constant-hue gamut clipped."""
    if not (contrast or highlights or shadows or whites or blacks):
        return np.asarray(lin, np.float32)
    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    L = np.clip(ok[..., 0], 0.0, 1.0)
    dL = np.zeros_like(L)
    if contrast:
        piv = np.float32(0.58)                                  # ~19 % linear mid-grey
        t = np.where(L <= piv, L/piv*0.5, 0.5 + (L - piv)/(1.0 - piv)*0.5)
        dL += np.float32(contrast*0.14) * np.sin(np.pi*t) * (2.0*t - 1.0)
    if shadows:
        w_sh = np.clip(1.0 - L/0.68, 0.0, 1.0)**2 * (1.0 - np.exp(-L/0.06))
        dL += np.float32(shadows*0.20) * w_sh
    if highlights:
        w_hi = np.clip((L - 0.35)/0.65, 0.0, 1.0)**2 * (1.0 - L**6)
        dL += np.float32(highlights*0.20) * w_hi
    if blacks:
        dL += np.float32(blacks*0.15) * (1.0 - L)**3
    if whites:
        dL += np.float32(whites*0.14) * (L**3)
    L_new = np.clip(L + dL, 0.0, 1.0)
    ratio = np.where(L > 1e-4, L_new/np.maximum(L, 1e-4), 1.0)
    c_scale = np.clip(ratio, 0.65, 1.20)
    hi_taper = np.where((L_new > L) & (L_new > 0.85),
                        np.clip((1.0 - L_new)/np.maximum(1.0 - L, 0.08), 0.0, 1.0)**0.35, 1.0)
    c_scale = (c_scale * hi_taper).astype(np.float32)
    ok_out = np.empty_like(ok)
    ok_out[..., 0] = L_new
    ok_out[..., 1] = ok[..., 1]*c_scale
    ok_out[..., 2] = ok[..., 2]*c_scale
    return oklab_to_lin_gamut(ok_out)


def apply_color_oklab(lin, saturation: float = 0.0, vibrance: float = 0.0, is_skin: bool = False):
    """Perceptual saturation and skin-sparing vibrance in OKLab (constant perceived hue, gamut-mapped)."""
    if not (saturation or vibrance):
        return np.asarray(lin, np.float32)
    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    a, b = ok[..., 1], ok[..., 2]
    C = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    # skin tones sit near OKLab hue ~48 deg (warm peach through deep brown)
    dh = np.abs(((hue - 48.0 + 180.0) % 360.0) - 180.0)
    skin_hue = np.clip(1.0 - dh/28.0, 0.0, 1.0) * np.clip(C/0.025, 0.0, 1.0)
    prot = 0.35 if is_skin else (1.0 - 0.65*skin_hue)
    f = np.maximum(0.0, (1.0 + vibrance*np.clip(1.0 - C/0.18, 0.0, 1.0)*prot)*(1.0 + saturation)).astype(np.float32)
    ok_out = ok.copy()
    ok_out[..., 1] *= f; ok_out[..., 2] *= f
    return oklab_to_lin_gamut(ok_out)


def apply_skin_oklch(lin, skin_cast: float = 0.0, skin_hue: float = 0.0):
    """OKLCh skin-locus cast reduction and skin hue control (applied to the skin 3D LUT so lightness L
    and fine pore detail remain 100 % intact):
    - skin_cast (0..1): pulls anomalous green/magenta/yellow colour casts on skin towards the natural
      portrait complexion locus in OKLCh (~50 deg) and gently tames excessive redness spikes.
    - skin_hue (-1..1): rotates skin hue angle (- = cooler/rosier, + = warmer/golden)."""
    if not (skin_cast or skin_hue):
        return np.asarray(lin, np.float32)
    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    L, a, b = ok[..., 0], ok[..., 1], ok[..., 2]
    C = np.hypot(a, b)
    h = np.arctan2(b, a)
    h_target = np.float32(np.radians(50.0))
    dh = np.arctan2(np.sin(h - h_target), np.cos(h - h_target))
    w_locus = np.exp(-((dh / 1.15) ** 4)) * np.clip(C / 0.015, 0.0, 1.0)
    h_new = h - np.float32(skin_cast) * 0.88 * w_locus * dh + np.float32(skin_hue) * 0.16 * w_locus
    c_excess = np.maximum(C - 0.11, 0.0)
    C_new = C - np.float32(max(0.0, skin_cast)) * 0.35 * w_locus * c_excess
    ok_out = np.empty_like(ok)
    ok_out[..., 0] = L
    ok_out[..., 1] = C_new * np.cos(h_new)
    ok_out[..., 2] = C_new * np.sin(h_new)
    return oklab_to_lin_gamut(ok_out)



CLARITY_SIDE = 256   # fixed analysis scale so preview and full-res export produce identical clarity gain


def clarity_gain(Lq, skin_q, amount: float):
    """Halo-free edge-preserving local mid-tone contrast via self-guided filter (He, Sun, Tang 2010),
    computed at a fixed working size (CLARITY_SIDE) for resolution independence (preview == export).
    Positive: crisp 3D micro-contrast, strongly sparing skin pores. Negative: soft dreamy portrait glow."""
    if not amount:
        return np.zeros(Lq.shape, np.float32)
    qh, qw = Lq.shape
    sc = min(1.0, CLARITY_SIDE/max(qh, qw))
    cw, ch = max(4, round(qw*sc)), max(4, round(qh*sc))
    L = Lq.astype(np.float32) if (ch, cw) == (qh, qw) else cv2.resize(Lq.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    r = max(2, int(round(max(ch, cw)/42))); k = (2*r + 1, 2*r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k)
    m = box(L); v = box(L*L) - m*m
    a = v/(v + 8.0**2); b = m - a*m
    base = box(a)*L + box(b)
    det = np.clip(L - base, -10.0, 10.0)
    l01 = np.clip(L/100.0, 0.0, 1.0)
    w_mid = 4.0*l01*(1.0 - l01)
    if skin_q is not None:
        sk = skin_q.astype(np.float32) if skin_q.shape == (ch, cw) else cv2.resize(skin_q.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
        sk = np.clip(sk, 0.0, 1.0)
    else:
        sk = 0.0
    skin_w = (1.0 - 0.65*sk) if amount > 0 else (1.0 - 0.25*sk)
    g = np.clip(float(amount)*0.22*w_mid*skin_w*det/np.maximum(L, 22.0), -0.25, 0.25).astype(np.float32)
    return g if (ch, cw) == (qh, qw) else cv2.resize(g, (qw, qh), interpolation=cv2.INTER_LINEAR)
