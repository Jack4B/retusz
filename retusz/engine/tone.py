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
                   whites: float = 0.0, blacks: float = 0.0, brightness: float = 0.0, dehaze: float = 0.0):
    """Perceptual tone controls in OKLab L: monotonic S-curve contrast around 18 % grey (L=0.58),
    studio midtone brightness (pinned at black 0 and white 1), atmospheric dehaze,
    mid-shadow/mid-highlight recovery, and black/white endpoint controls. Constant-hue gamut clipped."""
    if not (contrast or highlights or shadows or whites or blacks or brightness or dehaze):
        return np.asarray(lin, np.float32)
    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    L = np.clip(ok[..., 0], 0.0, 1.0)
    dL = np.zeros_like(L)
    if brightness:
        # Midtone brightness: smooth bell pinned at L=0 and L=1
        dL += np.float32(brightness * 0.34) * L * (1.0 - L) * (1.15 - 0.30 * L)
    if dehaze:
        # Atmospheric veil removal: deepens hazy shadows/midtones while keeping pure white clean
        dL -= np.float32(dehaze * 0.30) * L * (1.0 - 0.72 * (L ** 2))
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
    if dehaze:
        c_scale = np.clip(c_scale * (1.0 + np.float32(dehaze * 0.38) * (1.0 - 0.55 * L_new)), 0.45, 1.55)
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


def apply_skin_fill_oklch(lin, skin_fill: float = 0.0):
    """Dedicated Skin Fill Light ('Dopełnienie skóry', -1..1) in OKLCh inside the guided skin LUT:
    - Positive (0..1): acts like a soft silver/warm studio reflector exclusively on skin — lifts shadowed
      and midtone skin while protecting dark pupils/eyelashes (L <= 0.20), rolling off smoothly below L_max=0.95
      so bright forehead highlights never clip, and scaling chroma (L_new/L)^0.50 so lifted skin never turns grey.
    - Negative (-1..0): gently calms harsh on-camera flash skin highlights back into natural midtones."""
    sf = float(np.clip(skin_fill, -1.0, 1.0))
    if abs(sf) < 1e-5:
        return np.asarray(lin, np.float32)
    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    L = np.clip(ok[..., 0], 0.0, 1.0)
    a, b = ok[..., 1], ok[..., 2]
    C = np.hypot(a, b)
    h = np.arctan2(b, a)

    # Protect black pupils, eyelashes, mascara & dark brows (OKLab L <= 0.20 corresponds to sRGB <= 26)
    w_dark_prot = np.clip((L - 0.20) / 0.11, 0.0, 1.0)
    dh_skin = np.arctan2(np.sin(h - np.radians(48.0)), np.cos(h - np.radians(48.0)))
    w_hue = np.exp(-((dh_skin / 1.05) ** 4)).astype(np.float32)
    w_loc = (w_dark_prot * (0.35 + 0.65 * w_hue)).astype(np.float32)

    if sf > 0.0:
        t_hi = np.clip((0.91 - L) / 0.32, 0.0, 1.0)
        w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi)
        d = np.float32(0.175 * sf) * w_loc * w_hi
        room = np.maximum(0.96 - L, 1e-4)
        L_new = np.clip(L + np.minimum(d, room * (1.0 - np.exp(-d / room))), 0.0, 0.985)
        ratio_l = L_new / np.maximum(L, 0.20)
        k_c = np.clip(np.power(np.maximum(ratio_l, 1.0), 0.50), 1.0, 1.20)
        hi_tame = 1.0 - 0.35 * np.clip((C - 0.095) / 0.055, 0.0, 1.0)
        C_new = C * (1.0 + (k_c - 1.0) * hi_tame)
    else:
        w_neg = w_loc * (np.clip((L - 0.28) / 0.42, 0.0, 1.0) ** 1.10) * (1.0 - (L ** 12))
        L_new = np.clip(L + np.float32(0.18 * sf) * w_neg, 0.0, 1.0)
        c_rec = 1.0 + np.float32(0.08 * (-sf)) * w_neg * np.clip((0.09 - C) / 0.06, 0.0, 1.0)
        C_new = C * c_rec

    ok_out = np.empty_like(ok)
    ok_out[..., 0] = L_new.astype(np.float32)
    ok_out[..., 1] = (C_new * np.cos(h)).astype(np.float32)
    ok_out[..., 2] = (C_new * np.sin(h)).astype(np.float32)
    return oklab_to_lin_gamut(ok_out)


# 8 perceptual OKLCh hue centers (degrees): Red, Orange, Yellow, Green, Aqua, Blue, Purple, Magenta
HSL_CHANNELS = (
    ("red", "Czerwony", 26.0, 26.0),
    ("orange", "Pomarańczowy", 54.0, 24.0),
    ("yellow", "Żółty", 92.0, 26.0),
    ("green", "Zielony", 138.0, 34.0),
    ("aqua", "Akwamaryna", 190.0, 30.0),
    ("blue", "Niebieski", 248.0, 36.0),
    ("purple", "Fioletowy", 296.0, 28.0),
    ("magenta", "Magenta", 334.0, 28.0),
)


def apply_color_grading_oklab(
    lin,
    shadow_tone: float = 0.0,
    highlight_tone: float = 0.0,
    hsl_green: float = 0.0,
    hsl_blue: float = 0.0,
    midtone_tone: float = 0.0,
    grading_balance: float = 0.0,
    hsl_hue=None,
    hsl_sat=None,
    hsl_lum=None,
    is_skin: bool = False,
):
    """Perceptual 3-Way OKLab Color Grading (shadow_tone, midtone_tone, highlight_tone + grading_balance)
    and Full 8-Channel OKLCh HSL Mixer (hsl_hue, hsl_sat, hsl_lum + quick hsl_green/hsl_blue)
    with skin-locus protection."""
    has_hsl8 = any(
        x is not None and any(abs(float(v)) > 1e-4 for v in x)
        for x in (hsl_hue, hsl_sat, hsl_lum)
    )
    if not (shadow_tone or midtone_tone or highlight_tone or hsl_green or hsl_blue or has_hsl8):
        return np.asarray(lin, np.float32)

    ok = lin_to_oklab(np.clip(lin, 0.0, 1.0))
    L = np.clip(ok[..., 0], 0.0, 1.0).copy()
    a = ok[..., 1].copy()
    b = ok[..., 2].copy()

    if hsl_green or hsl_blue or has_hsl8:
        C = np.hypot(a, b)
        h = np.arctan2(b, a)
        h_deg = np.degrees(h) % 360.0
        dh_skin = np.abs(((h_deg - 50.0 + 180.0) % 360.0) - 180.0)
        w_not_skin = np.float32(0.0) if is_skin else (1.0 - np.exp(-((dh_skin / 32.0) ** 4))).astype(np.float32)
        c_gate = np.clip(C / 0.018, 0.0, 1.0).astype(np.float32)

        if has_hsl8:
            hh = [float(v) for v in (hsl_hue or (0.0,) * 8)]
            hs = [float(v) for v in (hsl_sat or (0.0,) * 8)]
            hl = [float(v) for v in (hsl_lum or (0.0,) * 8)]
            d_h_tot = np.zeros_like(h)
            c_mult_tot = np.ones_like(C)
            d_l_tot = np.zeros_like(L)
            for idx, (_, _, center_deg, width_deg) in enumerate(HSL_CHANNELS):
                vh = hh[idx] if idx < len(hh) else 0.0
                vs = hs[idx] if idx < len(hs) else 0.0
                vl = hl[idx] if idx < len(hl) else 0.0
                if abs(vh) <= 1e-4 and abs(vs) <= 1e-4 and abs(vl) <= 1e-4:
                    continue
                dh_c = ((h_deg - center_deg + 180.0) % 360.0) - 180.0
                w_c = np.exp(-((dh_c / width_deg) ** 2)).astype(np.float32) * c_gate
                if is_skin:
                    # Inside the skin LUT: strongly protect skin from background Red/Green/Blue/Purple edits;
                    # allow only gentle, controlled response on Orange (idx=1) if user intentionally tweaks it.
                    w_c = w_c * (0.28 if idx == 1 else 0.06)
                if abs(vh) > 1e-4:
                    d_h_tot += np.radians(22.0 * vh) * w_c
                if abs(vs) > 1e-4:
                    c_mult_tot *= np.clip(1.0 + (0.65 * vs if vs < 0 else 0.55 * vs) * w_c, 0.15, 1.75)
                if abs(vl) > 1e-4:
                    d_l_tot += np.float32(0.085 * vl) * w_c * (1.0 - (L ** 6))
            h = h + d_h_tot
            C = C * c_mult_tot
            L = np.clip(L + d_l_tot, 0.0, 1.0)

        if hsl_green:
            hg = float(np.clip(hsl_green, -1.0, 1.0))
            dh_g = ((h_deg - 132.0 + 180.0) % 360.0) - 180.0
            w_g = (np.exp(-((dh_g / 36.0) ** 2)) * np.clip(C / 0.022, 0.0, 1.0) * w_not_skin).astype(np.float32)
            c_mult_g = np.clip(1.0 + (0.48 * hg if hg < 0 else 0.42 * hg) * w_g, 0.25, 1.65)
            h = h + np.radians(-10.0 * hg) * w_g
            L = np.clip(L + np.float32(0.045 * hg) * w_g, 0.0, 1.0)
            C = C * c_mult_g

        if hsl_blue:
            hb = float(np.clip(hsl_blue, -1.0, 1.0))
            dh_b = ((h_deg - 252.0 + 180.0) % 360.0) - 180.0
            w_b = (np.exp(-((dh_b / 40.0) ** 2)) * np.clip(C / 0.018, 0.0, 1.0) * w_not_skin).astype(np.float32)
            c_mult_b = np.clip(1.0 + (0.58 * hb if hb < 0 else 0.48 * hb) * w_b, 0.20, 1.65)
            L = np.clip(L - np.float32(0.055 * hb) * w_b, 0.0, 1.0)
            C = C * c_mult_b

        a = (C * np.cos(h)).astype(np.float32)
        b = (C * np.sin(h)).astype(np.float32)

    piv = float(np.clip(0.55 - 0.20 * float(grading_balance or 0.0), 0.25, 0.82))
    sk_att = np.float32(0.38 if is_skin else 1.0)
    if shadow_tone:
        st = np.float32(np.clip(shadow_tone, -1.0, 1.0)) * sk_att
        w_sh = (np.clip(1.0 - L / max(piv + 0.25, 0.35), 0.0, 1.0) ** 2) * np.clip(L / 0.03, 0.0, 1.0)
        a = a + st * np.float32(0.014) * w_sh
        b = b + st * np.float32(0.036) * w_sh

    if midtone_tone:
        mt = np.float32(np.clip(midtone_tone, -1.0, 1.0)) * np.float32(0.45 if is_skin else 1.0)
        w_mid = np.exp(-((L - piv) ** 2) / (2.0 * (0.22 ** 2))).astype(np.float32) * np.clip(L / 0.05, 0.0, 1.0) * (1.0 - L ** 8)
        a = a + mt * np.float32(0.012) * w_mid
        b = b + mt * np.float32(0.030) * w_mid

    if highlight_tone:
        ht = np.float32(np.clip(highlight_tone, -1.0, 1.0)) * np.float32(0.65 if is_skin else 1.0)
        w_hi = (np.clip((L - (piv - 0.25)) / max(1.0 - (piv - 0.25), 0.25), 0.0, 1.0) ** 2) * (1.0 - (L ** 10))
        a = a + ht * np.float32(0.010) * w_hi
        b = b + ht * np.float32(0.034) * w_hi

    ok_out = np.empty_like(ok)
    ok_out[..., 0] = L
    ok_out[..., 1] = a
    ok_out[..., 2] = b
    return oklab_to_lin_gamut(ok_out)


def skin_ita_deg(rgb: np.ndarray, faces_n=()) -> float | None:
    """Compute Individual Typology Angle ITA° = arctan2(L* - 50, b*) * 180/pi in CIE 1976 LAB
    (Chardon, Cretois & Hourseau 1991) on clean facial skin.
    ITA° > 41°: very light/fair; 28..41°: intermediate; 10..28°: tan; -30..10°: brown/dark; < -30°: deep dark."""
    if not faces_n:
        return None
    H, W = rgb.shape[:2]
    s = min(1.0, 320.0 / max(H, W))
    sm = rgb if s >= 1.0 else cv2.resize(rgb, (max(2, int(round(W * s))), max(2, int(round(H * s)))),
                                          interpolation=cv2.INTER_AREA)
    sh, sw = sm.shape[:2]
    # Exact CIE 1976 L* (0..100), a*, b* (-128..127) via float32 sRGB -> LAB
    lab = cv2.cvtColor(sm.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    itas = []
    for f in faces_n:
        fx, fy, fw, fh = (float(v) for v in f[:4])
        x0 = max(0, int(round((fx + 0.22 * fw) * sw)))
        x1 = min(sw, int(round((fx + 0.78 * fw) * sw)))
        y0 = max(0, int(round((fy + 0.22 * fh) * sw)))
        y1 = min(sh, int(round((fy + 0.78 * fh) * sw)))
        if x1 > x0 + 3 and y1 > y0 + 3:
            patch_L = lab[y0:y1, x0:x1, 0]
            patch_b = lab[y0:y1, x0:x1, 2]
            clean = patch_L >= np.percentile(patch_L, 25.0)
            L_med = float(np.median(patch_L[clean])) if clean.any() else float(np.median(patch_L))
            b_med = float(np.median(patch_b[clean])) if clean.any() else float(np.median(patch_b))
            ita = float(np.degrees(np.arctan2(L_med - 50.0, max(b_med, 1.0))))
            itas.append(ita)
    return float(np.median(itas)) if itas else None


def auto_tone_params(rgb: np.ndarray, faces_n=()) -> dict:
    """Auto Ton AI v2 in CIELAB + OKLab perceptual space:
    - Separates CIE L* (0..100) and OKLab L (0..1) explicitly.
    - Adapts skin lightness target via Individual Typology Angle ITA° (Chardon 1991) so dark/deep
      complexions are never bleached to L*=67 and low-key studio intent is respected.
    - Blends scene luminance distribution with face exposure (max ±0.70 EV face-driven shift on dark/low-key scenes)
      while preserving highlight headroom (linear P99.5 < 0.98) and detecting split mixed lighting."""
    H, W = rgb.shape[:2]
    s = min(1.0, 320.0 / max(H, W))
    sm = rgb if s >= 1.0 else cv2.resize(rgb, (max(2, int(round(W * s))), max(2, int(round(H * s)))),
                                          interpolation=cv2.INTER_AREA)
    lin = srgb_to_lin(sm.astype(np.float32) / 255.0)
    ok = lin_to_oklab(lin)
    L = np.clip(ok[..., 0], 0.0, 1.0)
    C = np.hypot(ok[..., 1], ok[..., 2])
    lab_cie = cv2.cvtColor(sm.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    p01, p15, p50, p88, p99 = (float(v) for v in np.percentile(L, (1.0, 15.0, 50.0, 88.0, 99.0)))
    lin_p995 = float(np.percentile(lin.max(axis=-1), 99.5))

    sh, sw = sm.shape[:2]
    face_mask = np.zeros((sh, sw), bool)
    bg_excl = np.zeros((sh, sw), bool)
    face_L_ok = None
    face_ita = None
    mixed_light_score = 0.0
    face_vals_ok = []
    face_itas = []

    for f in (faces_n or ()):
        fx, fy, fw, fh = (float(v) for v in f[:4])
        bx0 = max(0, int(round((fx - 0.05 * fw) * sw)))
        bx1 = min(sw, int(round((fx + 1.05 * fw) * sw)))
        by0 = max(0, int(round((fy - 0.05 * fh) * sw)))
        by1 = min(sh, int(round((fy + 1.05 * fh) * sw)))
        bg_excl[by0:by1, bx0:bx1] = True
        x0 = max(0, int(round((fx + 0.18 * fw) * sw)))
        x1 = min(sw, int(round((fx + 0.82 * fw) * sw)))
        y0 = max(0, int(round((fy + 0.22 * fh) * sw)))
        y1 = min(sh, int(round((fy + 0.78 * fh) * sw)))
        if x1 > x0 + 4 and y1 > y0 + 2:
            face_mask[y0:y1, x0:x1] = True
            patch_ok = L[y0:y1, x0:x1]
            face_vals_ok.append(float(np.percentile(patch_ok, 60.0)))
            p_l_cie = lab_cie[y0:y1, x0:x1, 0]
            p_b_cie = lab_cie[y0:y1, x0:x1, 2]
            cl = p_l_cie >= np.percentile(p_l_cie, 25.0)
            lm_ = float(np.median(p_l_cie[cl])) if cl.any() else float(np.median(p_l_cie))
            bm_ = float(np.median(p_b_cie[cl])) if cl.any() else float(np.median(p_b_cie))
            face_itas.append(float(np.degrees(np.arctan2(lm_ - 50.0, max(bm_, 1.0)))))

            xm = (x0 + x1) // 2
            a_p = ok[y0:y1, x0:x1, 1]
            b_p = ok[y0:y1, x0:x1, 2]
            d_lr = float(np.hypot(ok[y0:y1, x0:xm, 1].mean() - ok[y0:y1, xm:x1, 1].mean(),
                                  ok[y0:y1, x0:xm, 2].mean() - ok[y0:y1, xm:x1, 2].mean()))
            sp_ab = float(np.hypot(np.percentile(a_p, 85) - np.percentile(a_p, 15),
                                   np.percentile(b_p, 85) - np.percentile(b_p, 15)))
            if d_lr > 0.028 or sp_ab > 0.052:
                sc = float(np.clip(max(d_lr - 0.022, sp_ab - 0.042) * 8.5, 0.0, 0.90))
                mixed_light_score = max(mixed_light_score, sc)

    if face_vals_ok:
        face_L_ok = float(np.median(face_vals_ok))
        face_ita = float(np.median(face_itas))

    bg_L = L[~bg_excl] if (~bg_excl).sum() > 32 else L.ravel()
    bg_p50 = float(np.percentile(bg_L, 50.0))
    bg_p88 = float(np.percentile(bg_L, 88.0))
    is_low_key = (bg_p50 < 0.28 and bg_p88 < 0.36)
    is_high_key = (bg_p50 > 0.74)

    # Phototype-adaptive target OKLab L derived from ITA° (Chardon 1991; CIE L* relative to scene white reference)
    if face_L_ok is not None and face_L_ok > 0.08:
        eff_ita = face_ita if face_ita is not None else 25.0
        if not is_low_key and p99 < 0.72 and bg_p88 >= face_L_ok * 0.92:
            # Globally underexposed scene (even background highlights are dim): estimate intrinsic ITA° after white-point normalization
            l_norm = min(92.0, (face_L_ok / max(p99, 0.32)) * 78.0)
            eff_ita = max(eff_ita, float(np.degrees(np.arctan2(l_norm - 50.0, 16.0))))
        t_ita = float(np.clip((eff_ita + 35.0) / 75.0, 0.0, 1.0))
        target_skin_ok = 0.54 + 0.18 * t_ita  # 0.54 for deep dark phototypes .. 0.72 for fair phototypes
        ev_lim = 0.85 if is_low_key else 1.15
        ev_face = float(np.clip(3.0 * np.log2(np.clip(target_skin_ok / face_L_ok, 0.52, 1.95)), -ev_lim, ev_lim))
        target_scene_ok = 0.42 if is_low_key else (0.68 if is_high_key else 0.58)
        ev_scene = float(np.clip(3.0 * np.log2(np.clip(target_scene_ok / max(p50, 0.12), 0.52, 1.95)), -1.15, 1.20))
        face_frac = float(np.clip(face_mask.mean() * 3.2, 0.42, 0.74))
        ev_raw = face_frac * ev_face + (1.0 - face_frac) * ev_scene
        if is_low_key or (eff_ita < 10.0):
            ev_raw = float(np.clip(ev_raw, -0.95, 0.68))
    else:
        target_mid_L = 0.44 if is_low_key else (0.68 if is_high_key else 0.58)
        ev_raw = 3.0 * np.log2(np.clip(target_mid_L / max(p50, 0.10), 0.55, 1.85))

    # Highlight headroom protection: keep linear P99.5 < 0.98
    max_ev_hi = float(np.log2(max(0.55, 0.975 / max(lin_p995, 0.22))))
    ev_cap = 0.68 if (is_low_key and (face_ita is not None and face_ita < 10.0)) else 1.30
    ev = float(np.clip(0.86 * ev_raw, -1.25, min(ev_cap, max_ev_hi + 0.35)))

    l_mult = float(2.0 ** (ev / 3.0))
    p15_post = min(1.0, p15 * l_mult)
    p88_post = min(1.0, p88 * l_mult)
    p99_post = min(1.0, p99 * l_mult)

    sh_target = 0.26 if is_low_key else 0.42
    sh_deficit = max(0.0, sh_target - p15_post) + max(0.0, ev_raw - ev) * 0.22
    shadows = float(np.clip(sh_deficit * (0.95 if is_low_key else 1.45), -0.10, 0.25 if is_low_key else 0.65))

    hi_excess = max(0.0, p88_post - 0.76) + max(0.0, p99_post - 0.92) * 1.2
    highlights = float(np.clip(-hi_excess * 2.1, -0.75, 0.15))

    spread = p88_post - p15_post
    contrast = float(np.clip((0.46 - spread) * (0.30 if is_low_key else 0.55), -0.10, 0.22))
    whites = float(np.clip((0.92 - p99_post) * (0.35 if is_low_key else 0.75), -0.20, 0.22))
    blacks = float(np.clip((0.12 - min(1.0, p01 * l_mult)) * 0.60, -0.22, 0.12))

    mean_c = float(np.mean(C))
    vibrance = float(np.clip((0.085 - mean_c) * 2.0, 0.0, 0.20))

    res = {
        "exposure": round(ev, 2),
        "contrast": round(contrast, 2),
        "highlights": round(highlights, 2),
        "shadows": round(shadows, 2),
        "whites": round(whites, 2),
        "blacks": round(blacks, 2),
        "vibrance": round(vibrance, 2),
    }
    if face_L_ok is not None and not is_low_key and (bg_p88 - face_L_ok > 0.18) and face_L_ok < 0.62:
        sf_auto = float(np.clip((bg_p88 - face_L_ok - 0.14) * 1.35, 0.0, 0.65))
        if sf_auto > 0.05:
            res["skin_fill"] = round(sf_auto, 2)
    if mixed_light_score > 0.05:
        res["mixed_light"] = round(mixed_light_score, 2)
    return res


CLARITY_SIDE = 256   # fixed analysis scale so preview and full-res export produce identical clarity gain


def clarity_gain(Lq, skin_q, amount: float):
    """Halo-free edge-preserving local mid-tone contrast via Fast Local Laplacian Filter
    (Paris, Hasinoff & Kautz 2011; Aubry et al. 2014), computed at a fixed working size for
    resolution independence (preview == export).
    Positive: crisp 3D micro-contrast with zero halo, sparing skin pores. Negative: soft portrait glow."""
    from .llf import fast_llf_gain
    return fast_llf_gain(Lq, skin_q, amount)


