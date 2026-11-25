"""Retusz engine (v3). Split into stages so the GUI stays responsive:

  analyse_reference()  once per reference (and per AI model)
  fit()                once per photo: AI/IDT global map, skin map, zone and vignette measurements
  bake()               per slider change: every per-colour step folded into two RGB->RGB 33^3 LUTs
  render()             one full-res pass + quarter-res masks + D&B on face crops + brush
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict, fields
import numpy as np, cv2
from . import color as C, faces as F, render as R, modflows as MF, relight as RL, dnb as DNB
from . import wb as WB, geometry as GEO, tone as TN

GRID = 33
ANALYSIS_SIDE = 1200   # masks, zones, vignette, skin statistics
FIT_SIDE = 800         # AI encoder / IDT input

@dataclass
class Params:
    """User controls. Ranges: strength/skin/vignette/db/plastic/skin_even/wide 0..1,
    exposure -2..2 EV, contrast/shadows/highlights/whites/blacks/saturation/vibrance/clarity/temp/tint -1..1."""
    strength: float = 1.0
    skin: float = 1.0
    exposure: float = 0.0           # 'Ekspozycja' -2..2 EV (linear light + filmic shoulder)
    contrast: float = 0.0           # 'Kontrast' -1..1 (OKLab S-curve)
    shadows: float = 0.0
    highlights: float = 0.0
    whites: float = 0.0             # 'Biele' -1..1 (OKLab upper endpoint)
    blacks: float = 0.0             # 'Czernie' -1..1 (OKLab black point)
    vignette: float = 0.3
    db: float = 0.4
    plastic: float = 0.3            # 'Plastyka 3D'
    clarity: float = 0.0            # 'Przejrzystość' -1..1 (guided local mid-tone contrast, spares skin)
    light_angle: float | None = None  # 'Kierunek światła' in degrees, None = auto (from the reference)
    saturation: float = 0.0         # 'Nasycenie' -1..1
    vibrance: float = 0.0           # 'Jaskrawość' -1..1 (OKLab, boosts dull colours, spares skin)
    skin_even: float = 0.3          # 'Równanie skóry' 0..1 (local colour casts on skin, mixed light)
    wide: float = 0.0               # 'Pełne barwy' 0..1: colours as sRGB numbers look on a wide-gamut (AdobeRGB) monitor
    temp: float = 0.0               # 'Temperatura' -1..1 (white balance, Bradford CAT)
    tint: float = 0.0               # 'Odcień' -1..1 (+ = magenta)
    angle: float = 0.0              # straighten, degrees -45..45 (per photo, see geometry.py)
    crop: list | None = None        # [x0, y0, x1, y1] normalised in the straightened frame (per photo)
    aspect: str = "orig"            # crop preset key (geometry.ASPECTS)
    strokes: list = field(default_factory=list)  # manual dodge/burn, see brush_map()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        p = cls(**{k: v for k, v in (d or {}).items() if k in names})
        if p.light_angle is not None: p.light_angle = float(p.light_angle) % 360
        p.exposure = float(np.clip(p.exposure, -2.0, 2.0))
        for k in ("strength", "skin", "vignette", "db", "plastic", "skin_even", "wide"):
            setattr(p, k, float(np.clip(getattr(p, k), 0, 1)))
        for k in ("contrast", "shadows", "highlights", "whites", "blacks", "clarity", "saturation", "vibrance", "temp", "tint"):
            setattr(p, k, float(np.clip(getattr(p, k), -1, 1)))
        p.strokes = list(p.strokes or [])
        p.angle = float(np.clip(p.angle or 0.0, -45, 45))
        p.crop = GEO.clamp_crop(p.crop)
        if p.aspect not in GEO.ASPECTS: p.aspect = "orig"
        return p

    def colour_key(self):
        return tuple(round(getattr(self, k), 3) for k in (
            "strength", "exposure", "contrast", "shadows", "highlights", "whites", "blacks",
            "saturation", "vibrance", "wide", "temp", "tint"))

@dataclass
class RefAnalysis:
    rgb: np.ndarray          # analysis-size reference
    fit_rgb: np.ndarray      # encoder-size reference
    faces: list
    mask: np.ndarray
    lab: np.ndarray
    vig_profile: np.ndarray
    embeddings: dict = field(default_factory=dict)  # model path -> embedding
    light_angle: float | None = None                 # estimated key-light direction of the reference

@dataclass
class Fit:
    glob: np.ndarray         # RGB->LAB LUT, global transfer
    skin: np.ndarray         # RGB->LAB LUT, skin transfer
    d_zone: list             # ab correction for shadows/mids/highlights
    faces_n: list            # faces normalised to image width (x, y, w, h)/W
    vig_diff: np.ndarray     # reference minus source radial L profile (<=0)
    offset: tuple            # closed-loop mean LAB residual
    method: str
    timings: dict = field(default_factory=dict)
    relight: object = None   # RL.Relight (3D data) or None
    guard_bg: tuple = None     # closed-loop background a/b correction (A, b), see _guard
    guard_skin: tuple = None   # closed-loop skin a/b correction (A, b)
    hue_k: object = None       # background per-hue chroma ratios (HUE_SECTORS)
    lms_n: list = None         # YuNet landmarks per face (5x2) normalised to image width; None = from the box

MASK_SIDE = 400   # skin mask working size (long side) for every resolution

def skin_mask_fixed(src, faces_n, qh, qw, subject=None):
    """Skin mask computed at a fixed working size and returned at (qh, qw). Resolution independent on
    purpose: with noisy input (RAW, no in-camera noise reduction) the colour test passed far more shadow
    pixels at full resolution than in the preview, so the export's shadows got the skin LUT."""
    H, W = src.shape[:2]; s = min(1.0, MASK_SIDE/max(H, W))
    mw, mh = max(2, round(W*s)), max(2, round(H*s))
    small = src if (mh, mw) == (H, W) else cv2.resize(src, (mw, mh), interpolation=cv2.INTER_AREA)
    m = F.skin_mask(small, [f*mw for f in faces_n], subject=subject).astype(np.float32)
    return m if (mh, mw) == (qh, qw) else cv2.resize(m, (qw, qh), interpolation=cv2.INTER_LINEAR)

def brush_map(strokes, h, w):
    """Rasterise brush strokes into a signed map (-1 burn .. +1 dodge) at h x w.
    Stroke = {"mode": "dodge"|"burn", "size": radius as fraction of width, "strength": flow 0..1, "softness": 0..1,
    "points": [[x, y], ...] normalised 0..1}."""
    out = np.zeros((h, w), np.float32)
    for s in strokes or []:
        pts = np.asarray(s.get("points", []), np.float32)
        if len(pts) == 0: continue
        r = max(1.0, float(s.get("size", 0.03))*w)
        layer = np.zeros((h, w), np.float32)
        p = np.round(pts*[w, h]).astype(np.int32)
        if len(p) == 1:
            cv2.circle(layer, tuple(int(v) for v in p[0]), int(r*0.6), 1.0, -1)
        else:
            cv2.polylines(layer, [p.reshape(-1, 1, 2)], False, 1.0, max(1, int(r*1.2)))
        soft = float(np.clip(s.get("softness", 0.5), 0, 1))
        layer = cv2.GaussianBlur(layer, (0, 0), max(0.5, r*(0.15+0.6*soft)))
        sign = 1.0 if s.get("mode", "dodge") == "dodge" else -1.0
        out += sign*float(s.get("strength", 0.5))*np.clip(layer*1.4, 0, 1)
    return np.clip(out, -1, 1)

def skin_even_gain(rgb_q, lab_q, mask, faces_q, amount):
    """Mixed-light skin evening: pull the low-frequency a/b of skin (face, neck, hands) towards the
    person's average skin tone, as per-channel RGB gains (quarter resolution)."""
    m = (mask > 0.3).astype(np.float32)
    if m.sum() < 50: return np.zeros(rgb_q.shape, np.float32)
    fw = float(np.median([f[2] for f in faces_q])) if faces_q else rgb_q.shape[1]/20
    ab = lab_q[..., 1:]; H, W = m.shape
    k = max(1.0, fw/32)                       # low-frequency field: compute where the face is ~32 px wide
    sw, sh = max(4, round(W/k)), max(4, round(H/k))
    ms = cv2.resize(m, (sw, sh), interpolation=cv2.INTER_AREA); abs_ = cv2.resize(np.ascontiguousarray(ab*m[..., None]), (sw, sh), interpolation=cv2.INTER_AREA)
    sg = max(1.0, fw/k*0.5)
    den = cv2.GaussianBlur(ms, (0, 0), sg) + 1e-4
    low = cv2.GaussianBlur(abs_, (0, 0), sg)/den[..., None]
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_LINEAR)
    w = mask > 0.5
    target = np.median(ab[w], axis=0) if w.sum() > 20 else (ab*m[..., None]).reshape(-1, 2).sum(0)/m.sum()
    corr = (target - low)
    mag = np.linalg.norm(corr, axis=2, keepdims=True); corr *= np.minimum(1, 12/(mag+1e-6))
    lab_t = lab_q.copy(); lab_t[..., 1:] += amount*corr*mask[..., None]
    rgb_t = cv2.cvtColor(lab_t.astype(np.float32), cv2.COLOR_LAB2RGB)
    cur = rgb_q.astype(np.float32)/255
    g = np.where(cur > 0.03, rgb_t/np.maximum(cur, 0.03) - 1, 0)
    return np.clip(g*mask[..., None], -0.35, 0.35).astype(np.float32)

# AdobeRGB (1998) -> sRGB, linear light, D65
_A2S = np.array([[1.39836, -0.39836, 0.0], [0.0, 1.0, 0.0], [0.0, -0.04293, 1.04293]], np.float32)

def wide_gamut_look(rgb):
    """What sRGB numbers look like when a wide-gamut (AdobeRGB-like) monitor shows them unmanaged,
    expressed back in sRGB: decode as AdobeRGB (gamma 2.2), convert to sRGB primaries, compress
    out-of-gamut chroma towards luminance (hue kept), encode sRGB. rgb, result: float 0..1."""
    lin = np.power(np.clip(rgb, 0, 1), 2.2) @ _A2S.T
    Y = (lin @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    d = lin - Y
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = np.where(d > 1e-6, (1 - Y)/d, np.inf); t_lo = np.where(d < -1e-6, -Y/d, np.inf)
    t = np.clip(np.minimum(t_hi, t_lo).min(axis=-1, keepdims=True), 0, 1)
    lin = np.clip(Y + d*t, 0, 1)
    return np.where(lin <= 0.0031308, 12.92*lin, 1.055*np.power(lin, 1/2.4) - 0.055).astype(np.float32)

def _guard(out_lab, out_m, ref_lab, ref_m, max_shift, c_lo, c_hi, mkl=0.0):
    """Closed-loop region guard. Returns an affine a/b correction (A 2x2, b 2): ab' = ab @ A + b, so that the rendered
    region matches the reference: mkl>0 blends in a linear Monge-Kantorovich map of the a/b distribution (Pitié 2007,
    removes casts that a mean shift cannot, e.g. cyan posters next to a warm wall); otherwise a Reinhard-style
    mean/chroma match. The result's mean chroma is kept within [c_lo, c_hi] x reference (no oversaturation)."""
    I = np.eye(2, dtype=np.float32)
    if out_m.sum() < 300 or ref_m.sum() < 300: return (I, np.zeros(2, np.float32))
    o = out_lab[..., 1:][out_m].astype(np.float64); r = ref_lab[..., 1:][ref_m].astype(np.float64)
    cr = np.hypot(r[:, 0], r[:, 1]).mean()
    if mkl > 0:
        mu_o, mu_r, T = C.mkl_fit(o, r); T = T.astype(np.float64)
        A = (1-mkl)*np.eye(2) + mkl*T.T; bb = mkl*(mu_r - mu_o@T.T)
        bb = bb + (1-mkl)*(mu_r - mu_o)*0.5
    else:
        co = np.hypot(o[:, 0], o[:, 1]).mean(); k = float(np.clip(cr/max(co, 1e-3), c_lo, c_hi))
        A = np.eye(2)*k; bb = r.mean(0) - o.mean(0)*k
    bb = np.clip(bb, -max_shift, max_shift)
    n = o@A + bb; cn = np.hypot(n[:, 0], n[:, 1]).mean()
    f = float(np.clip(cr*c_hi/max(cn, 1e-3), 0, 1)) if cn > cr*c_hi else float(max(1.0, cr*c_lo/max(cn, 1e-3)))
    return ((A*f).astype(np.float32), (bb*f).astype(np.float32))

def _dodge_burn_region(out, mq_raw, faces_px, strength, lms_px=None):
    """Auto D&B v2 (engine/dnb.py): one region for all faces, smooth luminance-only field on clean skin,
    feathered at the region border. The v1 band filter is kept as color.dodge_burn for the quality gate."""
    return DNB.dodge_burn_v2(out, mq_raw, faces_px, lms_px, strength)

HUE_SECTORS = 8

def _hue_chroma_match(out_lab, out_m, ref_lab, ref_m, lo=0.5, hi=1.05):
    """Per-hue chroma ratios (reference / result) in HUE_SECTORS sectors, from the 75th percentile chroma of each
    sector; a sector without enough pixels in both images stays 1. Like an automatic HSL 'saturation' panel."""
    k = np.ones(HUE_SECTORS, np.float32)
    o = out_lab[..., 1:][out_m]; r = ref_lab[..., 1:][ref_m]
    if len(o) < 500 or len(r) < 500: return k
    def sect(ab):
        c = np.hypot(ab[:, 0], ab[:, 1]); h = (np.degrees(np.arctan2(ab[:, 1], ab[:, 0])) % 360)
        return c, (h/(360/HUE_SECTORS)).astype(int) % HUE_SECTORS
    co, so = sect(o); cr, sr = sect(r)
    for i in range(HUE_SECTORS):
        a, b = co[(so == i) & (co > 4)], cr[(sr == i) & (cr > 4)]
        if len(a) > 0.01*len(o) and len(b) > 0.005*len(r):
            k[i] = float(np.clip(np.percentile(b, 75)/max(np.percentile(a, 75), 1e-3), lo, hi))
    return k

def _apply_hue_chroma(ab, k):
    h = (np.degrees(np.arctan2(ab[..., 1], ab[..., 0])) % 360)/(360/HUE_SECTORS) - 0.5
    i0 = np.floor(h).astype(int) % HUE_SECTORS; t = h - np.floor(h)
    f = k[i0]*(1-t) + k[(i0+1) % HUE_SECTORS]*t
    return ab*f[..., None]

BRUSH_GAIN = 0.25  # max +-25% exposure at strength 1 (subtle)

class Engine:
    """Holds model choice and caches. Thread-safe enough for one GUI worker + one batch thread
    (sessions are locked; caches are plain dicts of immutable results)."""

    def __init__(self, model_path=None, method="ai", depth_ai=True):
        self.depth_ai = depth_ai
        self.set_model(model_path, method)

    def set_model(self, model_path, method):
        self.model = str(model_path) if model_path else None
        self.method = method if self.model else "classic"

    def device(self):
        return MF.device_name(self.model) if self.model else "CPU"

    # ---------- stage 1 ----------
    def analyse_reference(self, ref_rgb) -> RefAnalysis:
        ref_rgb = C.crop_border(ref_rgb)
        r = C.small(ref_rgb, ANALYSIS_SIDE); fr = F.pick_faces(r); mr = F.skin_mask(r, fr)
        Rl = C.to_lab(r)
        bg = 1-cv2.dilate(mr, np.ones((15, 15)))
        ra = RefAnalysis(r, C.small(ref_rgb, FIT_SIDE), fr, mr, Rl, C.vignette_profile(Rl[..., 0], bg))
        ra.light_angle = RL.Relight(r, mr, [np.asarray(f, np.float32)/r.shape[1] for f in fr], self.depth_ai).auto_angle
        return ra

    # ---------- stage 2 ----------
    def fit(self, src_rgb, ref: RefAnalysis) -> Fit:
        T = {}; t = time.perf_counter()
        s = C.small(src_rgb, ANALYSIS_SIDE); fs, lms = F.pick_faces(s, landmarks=True); ms = F.skin_mask(s, fs)
        S = C.to_lab(s); T["analyse_s"] = time.perf_counter()-t; t = time.perf_counter()
        G, Glab = C.grid_lab(GRID)
        if self.model:
            if self.model not in ref.embeddings:
                ref.embeddings[self.model] = MF.embed(ref.fit_rgb, self.model)
            lut_rgb = MF.lut_from_embeddings(MF.embed(C.small(src_rgb, FIT_SIDE), self.model), ref.embeddings[self.model], N=GRID)
            glob = cv2.cvtColor(lut_rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(GRID, GRID, GRID, 3)
        else:
            glob = C.fit_idt_lut(C.small(src_rgb, FIT_SIDE), ref.fit_rgb, N=GRID)
        glob = C.smooth_lut(glob)
        sk_s = S.reshape(-1, 3)[ms.ravel() > 0.6]; sk_r = ref.lab.reshape(-1, 3)[ref.mask.ravel() > 0.6]
        if len(sk_s) > 500 and len(sk_r) > 500:
            mu_s, mu_r, Tm = C.mkl_fit(sk_s[:, 1:], sk_r[:, 1:])       # colour of skin: linear Monge-Kantorovich on a/b
            g = Glab.reshape(-1, 3); skin = np.empty_like(g)
            k = float(np.clip(sk_r[:, 0].std()/max(sk_s[:, 0].std(), 1e-3), 0.85, 1.05))   # never harden skin contrast much
            skin[:, 0] = (g[:, 0]-sk_s[:, 0].mean())*k + sk_r[:, 0].mean()
            skin[:, 1:] = (g[:, 1:]-mu_s)@Tm + mu_r
            skin = skin.reshape(Glab.shape).astype(np.float32)
        else:
            skin = glob
        T["fit_s"] = time.perf_counter()-t; t = time.perf_counter()
        # zone (shadows/mids/highlights) ab differences, measured after the transfer
        s6 = C.small(s, 600); ms6 = cv2.resize(ms, (s6.shape[1], s6.shape[0]), interpolation=cv2.INTER_AREA)
        Gs = C.apply_lut_fast(s6, glob); Ks = C.apply_lut_fast(s6, skin)   # statistics only: 600 px is enough
        Gs = Gs*(1-ms6[..., None]) + Ks*ms6[..., None]
        zg, zr = C.zone_w(Gs[..., 0]), C.zone_w(ref.lab[..., 0]); d = []
        for i in range(3):
            if zg[i].sum() < 100 or zr[i].sum() < 100:
                d.append(np.zeros(2, np.float32)); continue
            d.append(((ref.lab[..., 1:]*zr[i][..., None]).reshape(-1, 2).sum(0)/zr[i].sum()
                      - (Gs[..., 1:]*zg[i][..., None]).reshape(-1, 2).sum(0)/zg[i].sum()).astype(np.float32))
        bg_g = 1-cv2.dilate(ms6, np.ones((9, 9)))
        vig_diff = np.clip(ref.vig_profile - C.vignette_profile(Gs[..., 0], bg_g), -10, 0).astype(np.float32)
        W = s.shape[1]
        fit = Fit(glob, skin, d, [np.asarray(f, np.float32)/W for f in fs], vig_diff, (0.0, 0.0, 0.0), self.method or "classic", T)
        fit.lms_n = [np.asarray(l, np.float32)/W for l in lms] if len(lms) == len(fs) else None
        t3 = time.perf_counter()
        fit.relight = RL.Relight(s, ms, fit.faces_n, self.depth_ai)
        if ref.light_angle is not None: fit.relight.auto_angle = ref.light_angle   # light like in the reference
        T["depth_s"] = time.perf_counter()-t3
        # closed loop 1: per-region guard (skin / background): match mean a/b, never exceed the reference chroma
        loop = Params(plastic=0.0, skin_even=0.0, vignette=0.0, strokes=[])
        out = C.to_lab(self.render(s, fit, loop, db_enabled=False))
        fit.guard_skin = _guard(out, ms > 0.6, ref.lab, ref.mask > 0.6, max_shift=5.0, c_lo=0.75, c_hi=1.02)
        fit.guard_bg = _guard(out, ms < 0.1, ref.lab, ref.mask < 0.1, max_shift=6.0, c_lo=0.6, c_hi=1.03, mkl=0.7)
        fit._lut_cache = {}
        out = C.to_lab(self.render(s, fit, loop, db_enabled=False))
        fit.hue_k = _hue_chroma_match(out, ms < 0.1, ref.lab, ref.mask < 0.1)
        fit._lut_cache = {}
        # closed loop 2: global mean residual
        out = self.render(s, fit, Params(plastic=0.0, skin_even=0.0, strokes=[]), db_enabled=False)  # colour only
        res = ref.lab.reshape(-1, 3).mean(0) - C.to_lab(out).reshape(-1, 3).mean(0)
        fit.offset = (float(np.clip(res[0], -4, 4)), float(np.clip(res[1], -3, 3)), float(np.clip(res[2], -3, 3)))
        fit._lut_cache = {}
        T["prepare_s"] = time.perf_counter()-t
        return fit

    # ---------- stage 3 ----------
    @staticmethod
    def _bake_one(lab_in, d_zone, zone_amt, skin_protect, p: Params, offset, chroma_lim=1.2, is_skin=False, guard=None, hue_k=None):
        N = lab_in.shape[0]; _, S = C.grid_lab(N)
        O = lab_in.copy(); zw = C.zone_w(O[..., 0])
        if guard is not None: O[..., 1:] = O[..., 1:]@guard[0] + guard[1]
        if hue_k is not None: O[..., 1:] = _apply_hue_chroma(O[..., 1:], hue_k)
        for i in range(3): O[..., 1:] += zone_amt*0.7*d_zone[i]*zw[i][..., None]
        O[..., 0] += offset[0]*np.clip(O[..., 0]/50, 0, 1); O[..., 1] += offset[1]; O[..., 2] += offset[2]
        if skin_protect: O[..., 1:] -= skin_protect*0.5*(O[..., 1:]-S[..., 1:])
        Cs = np.hypot(S[..., 1], S[..., 2]); Cg = np.hypot(O[..., 1], O[..., 2])+1e-6
        f = np.minimum(1, (Cs*chroma_lim+6)/Cg); O[..., 1] *= f; O[..., 2] *= f
        O[..., 0] = C.rolloff(np.clip(O[..., 0], 0, None)); wf = np.clip((100-O[..., 0])/12, 0, 1); O[..., 1:] *= wf[..., None]**0.5
        O = S + p.strength*(O-S)
        rgb = cv2.cvtColor(O.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(N, N, N, 3)
        rgb = np.clip(rgb, 0, 1)
        # Photographic tone & colour edits in linear light + OKLab (work even when strength == 0)
        has_tone = any((p.exposure, p.contrast, p.highlights, p.shadows, p.whites, p.blacks, p.saturation, p.vibrance))
        if has_tone:
            lin = TN.srgb_to_lin(rgb)
            if p.exposure: lin = TN.apply_exposure_lin(lin, p.exposure)
            if p.contrast or p.highlights or p.shadows or p.whites or p.blacks:
                lin = TN.apply_tone_lin(lin, p.contrast, p.highlights, p.shadows, p.whites, p.blacks)
            if p.saturation or p.vibrance:
                lin = TN.apply_color_oklab(lin, p.saturation, p.vibrance, is_skin=is_skin)
            rgb = TN.lin_to_srgb(lin)
        if p.temp or p.tint: rgb = WB.apply_bradford01(rgb, p.temp, p.tint)
        if p.wide > 0: rgb = rgb + p.wide*(wide_gamut_look(rgb) - rgb)
        return rgb.astype(np.float32)

    def bake(self, fit: Fit, p: Params):
        cache = getattr(fit, "_lut_cache", None)
        key = p.colour_key()
        if cache is not None and key in cache: return cache[key]
        o = fit.offset
        luts = (self._bake_one(fit.glob, fit.d_zone, 1.0, 0, p, o, guard=fit.guard_bg, hue_k=fit.hue_k),
                self._bake_one(fit.skin, fit.d_zone, 0.3, 0.25, p, (o[0], o[1]*0.5, o[2]*0.5), is_skin=True, guard=fit.guard_skin))
        if cache is not None:
            if len(cache) > 16: cache.clear()
            cache[key] = luts
        return luts

    def render(self, src, fit: Fit, p: Params, db_enabled=True, return_mask=False, timings=None):
        """Apply a fit to src at its own resolution (preview or full-size)."""
        T = timings if timings is not None else {}
        t = time.perf_counter()
        H, W = src.shape[:2]; qh, qw = max(2, H//4), max(2, W//4)
        lut_bg, lut_sk = self.bake(fit, p)
        q = cv2.resize(src, (qw, qh), interpolation=cv2.INTER_AREA)
        subj = fit.relight.fg if (fit.relight is not None and getattr(fit.relight, "fg", None) is not None) else None
        mq_raw = skin_mask_fixed(src, fit.faces_n, qh, qw, subject=subj)
        mq = mq_raw*p.skin
        gain = np.zeros((qh, qw, 3), np.float32)
        need_vig = p.vignette > 0 and np.any(fit.vig_diff < 0)
        need_3d = p.plastic > 0 and fit.relight is not None
        need_even = p.skin_even > 0 and fit.faces_n and mq_raw.max() > 0.3
        need_clarity = abs(p.clarity) > 1e-4
        if need_vig or need_3d or need_even or need_clarity:
            base_q = R.render(q, lut_bg, lut_sk, mq, gain); lab_q = C.to_lab(base_q); Lq = lab_q[..., 0]
        if need_even:
            gain += skin_even_gain(base_q, lab_q, mq_raw, [f*qw for f in fit.faces_n], p.skin_even)
        if need_3d:
            gain += fit.relight.gain(qh, qw, Lq, p.plastic, p.light_angle, rgb_q=base_q)[..., None]
        if need_clarity:
            gain += TN.clarity_gain(Lq, mq_raw, p.clarity)[..., None]
        if need_vig:
            rad = C.radial(qh, qw); n = len(fit.vig_diff); idx = np.clip(rad*n-0.5, 0, n-1)
            dL = cv2.GaussianBlur(np.interp(idx, np.arange(n), fit.vig_diff).astype(np.float32), (0, 0), max(qh, qw)/80)
            gain += (p.vignette*dL/np.maximum(Lq, 8)*(1-mq_raw)).astype(np.float32)[..., None]
        if p.strokes:
            gain += (BRUSH_GAIN*brush_map(p.strokes, qh, qw))[..., None]
        T["masks_s"] = time.perf_counter()-t; t = time.perf_counter()
        out = R.render(src, lut_bg, lut_sk, mq, gain)
        T["render_s"] = time.perf_counter()-t; t = time.perf_counter()
        if db_enabled and p.db > 0 and fit.faces_n:
            lms = getattr(fit, "lms_n", None)
            out = _dodge_burn_region(out, mq_raw, [f*W for f in fit.faces_n], p.db, [l*W for l in lms] if lms else None)
        T["dodge_burn_s"] = time.perf_counter()-t
        return (out, mq_raw) if return_mask else out

    # ---------- convenience ----------
    def process(self, src_rgb, ref: RefAnalysis, p: Params, timings=None):
        fit = self.fit(src_rgb, ref)
        if timings is not None: timings.update(fit.timings)
        return self.render(src_rgb, fit, p, timings=timings)
