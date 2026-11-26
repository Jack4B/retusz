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
from . import wb as WB, geometry as GEO, tone as TN, colorchecker as CC, skin_physics as SP, detail as DET

GRID = 33
ANALYSIS_SIDE = 1200   # masks, zones, vignette, skin statistics
FIT_SIDE = 800         # AI encoder / IDT input

@dataclass
class Params:
    """User controls. Ranges: strength/skin/vignette/db/plastic/skin_even/skin_cast/deshine/blemish/eyes_teeth/sharpen_amount/denoise/wide 0..1,
    exposure -2..2 EV, contrast/brightness/dehaze/texture/shadows/highlights/whites/blacks/saturation/vibrance/clarity/skin_hue/temp/tint -1..1."""
    strength: float = 1.0
    skin: float = 1.0
    exposure: float = 0.0           # 'Ekspozycja' -2..2 EV (linear light + filmic shoulder)
    contrast: float = 0.0           # 'Kontrast' -1..1 (OKLab S-curve)
    brightness: float = 0.0         # 'Jasność (Półtony)' -1..1 (Capture One midtone transfer in OKLab L)
    dehaze: float = 0.0             # 'Zamglenie (Dehaze)' -1..1 (OKLab atmospheric veil removal)
    shadows: float = 0.0
    highlights: float = 0.0
    whites: float = 0.0             # 'Biele' -1..1 (OKLab upper endpoint)
    blacks: float = 0.0             # 'Czernie' -1..1 (OKLab black point)
    vignette: float = 0.3
    db: float = 0.4
    plastic: float = 0.3            # 'Plastyka 3D'
    clarity: float = 0.0            # 'Przejrzystość' -1..1 (Fast LLF local mid-tone contrast, spares skin)
    texture: float = 0.0            # 'Struktura' -1..1 (fine high-frequency detail band)
    sharpen_amount: float = 0.0     # 'Ostrość' 0..1 (live in-editor & export sharpening)
    sharpen_mode: str = "portrait"  # 'Tryb wyostrzania': portrait | rl | edges | soft
    denoise: float = 0.0            # 'Odszumianie' 0..1 (edge-preserving OKLab chroma + luma denoise)
    light_angle: float | None = None  # 'Kierunek światła' in degrees, None = auto (from the reference)
    light_x: float = 0.0            # 'Światło 3D: Lewo ↔ Prawo' -1..1
    light_y: float = 0.0            # 'Światło 3D: Dół ↔ Góra' -1..1
    bg_exposure: float = 0.0        # 'Separacja tła 3D' -1..1 (3D background exposure offset)
    saturation: float = 0.0         # 'Nasycenie' -1..1
    vibrance: float = 0.0           # 'Jaskrawość' -1..1 (OKLab, boosts dull colours, spares skin)
    shadow_tone: float = 0.0        # 'Ciepło cieni' -1..1 (OKLab Split Toning: cool blue <-> warm amber shadows)
    highlight_tone: float = 0.0     # 'Blask świateł' -1..1 (OKLab Split Toning: cool <-> golden hour highlights)
    hsl_green: float = 0.0          # 'Zieleń (Plener)' -1..1 (OKLCh selective foliage green control, spares skin)
    hsl_blue: float = 0.0           # 'Błękit (Niebo/Koszule)' -1..1 (OKLCh selective blue cast/sky control, spares skin)
    skin_even: float = 0.3          # 'Równanie skóry' 0..1 (chromophore hemoglobin/melanin + mixed-light evening)
    mixed_light: float = 0.0        # 'Światło mieszane (LED)' 0..1 (Dual-illuminant split-face white/red unification in OKLab)
    skin_cast: float = 0.0          # 'Czysta skóra' 0..1 (OKLCh skin-locus cast reduction)
    skin_hue: float = 0.0           # 'Odcień skóry' -1..1 (OKLCh skin hue angle rotation: - = Róż/Czerwień, + = Złoty/Żółty)
    deshine: float = 0.0            # 'Redukcja połysku' 0..1 (Dichromatic SUV skin specular removal)
    blemish: float = 0.0            # 'Niedoskonałości' 0..1 (Two-band pore-preserving blemish auto-heal)
    eyes_teeth: float = 0.0         # 'Oczy i zęby' 0..1 (Anatomical iris/sclera/teeth enhancement)
    mask_hue_tol: float = 0.0       # 'Zasięg odcienia maski AI' -1..1 (Mahalanobis skin colour & shadow range)
    mask_protect: float = 0.0       # 'Ochrona oczu i ust (AI)' 0..1 (Anatomical landmark eye/lip mask exclusion)
    wide: float = 0.0               # 'Pełne barwy' 0..1: colours as sRGB numbers look on a wide-gamut (AdobeRGB) monitor
    temp: float = 0.0               # 'Temperatura' -1..1 (white balance, Bradford CAT)
    tint: float = 0.0               # 'Odcień' -1..1 (+ = magenta)
    angle: float = 0.0              # straighten, degrees -45..45 (per photo, see geometry.py)
    crop: list | None = None        # [x0, y0, x1, y1] normalised in the straightened frame (per photo)
    aspect: str = "orig"            # crop preset key (geometry.ASPECTS)
    strokes: list = field(default_factory=list)  # manual dodge/burn, see brush_map()
    rating: int = 0                 # 0..5 star rating per photo (Library module)
    reference_path: str | None = None  # optional per-photo reference path override ('__none__' = standalone)
    ref_colorchecker: bool = False  # whether per-photo reference is a ColorChecker 24 chart

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        p = cls(**{k: v for k, v in (d or {}).items() if k in names})
        if p.light_angle is not None: p.light_angle = float(p.light_angle) % 360
        p.exposure = float(np.clip(p.exposure, -2.0, 2.0))
        for k in ("strength", "skin", "vignette", "db", "plastic", "skin_even", "mixed_light", "skin_cast",
                  "deshine", "blemish", "eyes_teeth", "mask_protect", "sharpen_amount", "denoise", "wide"):
            setattr(p, k, float(np.clip(getattr(p, k), 0, 1)))
        for k in ("contrast", "brightness", "dehaze", "texture", "shadows", "highlights", "whites", "blacks",
                  "clarity", "saturation", "vibrance", "shadow_tone", "highlight_tone", "hsl_green", "hsl_blue",
                  "skin_hue", "mask_hue_tol", "temp", "tint", "light_x", "light_y", "bg_exposure"):
            setattr(p, k, float(np.clip(getattr(p, k), -1, 1)))
        if p.sharpen_mode not in DET.SHARPEN_MODES: p.sharpen_mode = "portrait"
        p.rating = int(np.clip(int(p.rating or 0), 0, 5))
        p.reference_path = str(p.reference_path) if p.reference_path else None
        p.ref_colorchecker = bool(p.ref_colorchecker)
        p.strokes = list(p.strokes or [])
        p.angle = float(np.clip(p.angle or 0.0, -45, 45))
        p.crop = GEO.clamp_crop(p.crop)
        if p.aspect not in GEO.ASPECTS: p.aspect = "orig"
        return p

    def colour_key(self):
        return tuple(round(getattr(self, k), 3) for k in (
            "strength", "exposure", "contrast", "brightness", "dehaze", "shadows", "highlights", "whites", "blacks",
            "saturation", "vibrance", "shadow_tone", "highlight_tone", "hsl_green", "hsl_blue",
            "skin_cast", "skin_hue", "wide", "temp", "tint"))

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
    cc_cal: object = None    # CC.ChartCalibration if calibrated via ColorChecker

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
    cc_report: object = None   # CC.ChartReport if calibrated via ColorChecker
    cc_lut_srgb: np.ndarray | None = None  # 33^3x3 sRGB 0..1 LUT from ColorChecker calibration

MASK_SIDE = 400   # skin mask working size (long side) for every resolution

def skin_mask_fixed(src, faces_n, qh, qw, subject=None, hue_tol: float = 0.0, protect_features: float = 0.0, lms_n=None):
    """Skin mask computed at a fixed working size and returned at (qh, qw). Resolution independent on
    purpose: with noisy input (RAW, no in-camera noise reduction) the colour test passed far more shadow
    pixels at full resolution than in the preview, so the export's shadows got the skin LUT."""
    H, W = src.shape[:2]; s = min(1.0, MASK_SIDE/max(H, W))
    mw, mh = max(2, round(W*s)), max(2, round(H*s))
    small = src if (mh, mw) == (H, W) else cv2.resize(src, (mw, mh), interpolation=cv2.INTER_AREA)
    lms_scaled = [np.asarray(l, np.float32) * mw for l in lms_n] if lms_n else None
    m = F.skin_mask(small, [f*mw for f in faces_n], subject=subject,
                    hue_tol=hue_tol, protect_features=protect_features, lms=lms_scaled).astype(np.float32)
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
    def analyse_reference(self, ref_rgb, colorchecker: bool = False) -> RefAnalysis:
        ref_rgb = C.crop_border(ref_rgb)
        r = C.small(ref_rgb, ANALYSIS_SIDE); fr = F.pick_faces(r); mr = F.skin_mask(r, fr)
        Rl = C.to_lab(r)
        bg = 1-cv2.dilate(mr, np.ones((15, 15)))
        ra = RefAnalysis(r, C.small(ref_rgb, FIT_SIDE), fr, mr, Rl, C.vignette_profile(Rl[..., 0], bg))
        ra.light_angle = RL.Relight(r, mr, [np.asarray(f, np.float32)/r.shape[1] for f in fr], self.depth_ai).auto_angle
        if colorchecker:
            ra.cc_cal = CC.calibrate_chart(ref_rgb)
        return ra

    def prepare(self, src_rgb, reference=None, colorchecker: bool = False) -> Fit:
        """Convenience stage-1 + stage-2 entry point: accepts None, a raw RGB reference array, or a RefAnalysis."""
        if isinstance(reference, np.ndarray):
            ref = self.analyse_reference(reference, colorchecker=colorchecker)
        else:
            ref = reference
            if colorchecker and ref is not None and getattr(ref, "cc_cal", None) is None:
                ref.cc_cal = CC.calibrate_chart(ref.rgb)
        return self.fit(src_rgb, ref, colorchecker=colorchecker)

    # ---------- stage 2 ----------
    def fit(self, src_rgb, ref: RefAnalysis | None = None, colorchecker: bool = False) -> Fit:
        T = {}; t = time.perf_counter()
        s = C.small(src_rgb, ANALYSIS_SIDE); fs, lms = F.pick_faces(s, landmarks=True); ms = F.skin_mask(s, fs)
        S = C.to_lab(s); T["analyse_s"] = time.perf_counter()-t; t = time.perf_counter()
        G, Glab = C.grid_lab(GRID)
        W = s.shape[1]

        # ColorChecker calibration mode (from reference chart or chart directly inside src_rgb)
        cc_cal = getattr(ref, "cc_cal", None) if ref is not None else None
        if colorchecker and cc_cal is None:
            cc_cal = CC.calibrate_chart(ref.rgb if ref is not None else src_rgb)
        if cc_cal is not None:
            d = [np.zeros(2, np.float32) for _ in range(3)]
            vig_diff = np.linspace(0.0, -10.0, 32, dtype=np.float32)
            fit = Fit(cc_cal.lut_lab, cc_cal.lut_lab, d, [np.asarray(f, np.float32)/W for f in fs],
                      vig_diff, (0.0, 0.0, 0.0), "colorchecker", T)
            fit.cc_report = cc_cal.report
            fit.cc_lut_srgb = cc_cal.lut_srgb
            fit.lms_n = [np.asarray(l, np.float32)/W for l in lms] if len(lms) == len(fs) else None
            t3 = time.perf_counter()
            fit.relight = RL.Relight(s, ms, fit.faces_n, self.depth_ai)
            if ref is not None and ref.light_angle is not None:
                fit.relight.auto_angle = ref.light_angle
            T["depth_s"] = time.perf_counter()-t3
            fit._lut_cache = {}
            T["prepare_s"] = time.perf_counter()-t
            return fit

        if ref is None:
            glob = Glab.astype(np.float32)
            d = [np.zeros(2, np.float32) for _ in range(3)]
            vig_diff = np.linspace(0.0, -10.0, 32, dtype=np.float32)
            fit = Fit(glob, glob, d, [np.asarray(f, np.float32)/W for f in fs], vig_diff, (0.0, 0.0, 0.0), "standalone", T)
            fit.lms_n = [np.asarray(l, np.float32)/W for l in lms] if len(lms) == len(fs) else None
            t3 = time.perf_counter()
            fit.relight = RL.Relight(s, ms, fit.faces_n, self.depth_ai)
            T["depth_s"] = time.perf_counter()-t3
            T["prepare_s"] = time.perf_counter()-t
            return fit
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
    def _bake_one(lab_in, d_zone, zone_amt, skin_protect, p: Params, offset, chroma_lim=1.2, is_skin=False,
                  guard=None, hue_k=None, standalone=False, cc_lut_srgb=None):
        N = lab_in.shape[0]
        if cc_lut_srgb is not None:
            G = C.rgb_grid(N)
            rgb = np.clip(G + p.strength * (cc_lut_srgb - G), 0.0, 1.0).astype(np.float32)
        elif standalone:
            rgb = C.rgb_grid(N).copy()
        else:
            _, S = C.grid_lab(N)
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
        has_grading = any((p.shadow_tone, p.highlight_tone, p.hsl_green, p.hsl_blue))
        has_tone = any((p.exposure, p.contrast, p.brightness, p.dehaze, p.highlights, p.shadows, p.whites, p.blacks,
                        p.saturation, p.vibrance, has_grading, is_skin and (p.skin_cast or p.skin_hue)))
        if has_tone:
            lin = TN.srgb_to_lin(rgb)
            if p.exposure: lin = TN.apply_exposure_lin(lin, p.exposure)
            if p.contrast or p.brightness or p.dehaze or p.highlights or p.shadows or p.whites or p.blacks:
                lin = TN.apply_tone_lin(lin, p.contrast, p.highlights, p.shadows, p.whites, p.blacks,
                                        brightness=p.brightness, dehaze=p.dehaze)
            if is_skin and (p.skin_cast or p.skin_hue):
                lin = TN.apply_skin_oklch(lin, p.skin_cast, p.skin_hue)
            if has_grading:
                lin = TN.apply_color_grading_oklab(lin, shadow_tone=p.shadow_tone, highlight_tone=p.highlight_tone,
                                                   hsl_green=p.hsl_green, hsl_blue=p.hsl_blue, is_skin=is_skin)
            if p.saturation or p.vibrance:
                lin = TN.apply_color_oklab(lin, p.saturation, p.vibrance, is_skin=is_skin)
            rgb = TN.lin_to_srgb(lin)
        if p.temp or p.tint: rgb = WB.apply_bradford01(rgb, p.temp, p.tint)
        if p.wide > 0: rgb = rgb + p.wide*(wide_gamut_look(rgb) - rgb)
        return rgb.astype(np.float32)

    def bake(self, fit: Fit, p: Params):
        cache = getattr(fit, "_lut_cache", None)
        sa = fit.method == "standalone"
        cc_lut = getattr(fit, "cc_lut_srgb", None)
        key = (p.colour_key(), fit.method)
        if cache is not None and key in cache: return cache[key]
        o = fit.offset
        luts = (self._bake_one(fit.glob, fit.d_zone, 1.0, 0, p, o, guard=fit.guard_bg, hue_k=fit.hue_k, standalone=sa, cc_lut_srgb=cc_lut),
                self._bake_one(fit.skin, fit.d_zone, 0.3, 0.25, p, (o[0], o[1]*0.5, o[2]*0.5), is_skin=True, guard=fit.guard_skin, standalone=sa, cc_lut_srgb=cc_lut))
        if cache is not None:
            if len(cache) > 16: cache.clear()
            cache[key] = luts
        return luts

    def render(self, src, fit: Fit, p: Params, db_enabled=True, return_mask=False, timings=None, preview=False):
        """Apply a fit to src at its own resolution (preview or full-size)."""
        T = timings if timings is not None else {}
        t = time.perf_counter()
        H, W = src.shape[:2]; qh, qw = max(2, H//4), max(2, W//4)
        lut_bg, lut_sk = self.bake(fit, p)
        q = cv2.resize(src, (qw, qh), interpolation=cv2.INTER_AREA)
        subj = fit.relight.fg if (fit.relight is not None and getattr(fit.relight, "fg", None) is not None) else None
        lms_n = getattr(fit, "lms_n", None)
        mq_raw = skin_mask_fixed(src, fit.faces_n, qh, qw, subject=subj,
                                 hue_tol=p.mask_hue_tol, protect_features=p.mask_protect,
                                 lms_n=lms_n)
        lms_q = [np.asarray(l, np.float32) * qw for l in lms_n] if lms_n else None
        if p.mixed_light > 1e-4 and fit.faces_n:
            mq_raw = SP.mixed_light_face_mask(q, mq_raw, [f*qw for f in fit.faces_n], lms_q=lms_q)
        skin_w = max(p.skin, 1.0 if (p.skin_cast > 0 or abs(p.skin_hue) > 1e-4 or p.mixed_light > 1e-4) else 0.0)
        mq = mq_raw*skin_w
        gain = np.zeros((qh, qw, 3), np.float32)
        need_vig = p.vignette > 0 and np.any(fit.vig_diff < 0)
        need_3d = p.plastic > 0 and fit.relight is not None
        need_bg = abs(p.bg_exposure) > 1e-4 and fit.relight is not None
        need_mixed = p.mixed_light > 1e-4 and (fit.faces_n or mq_raw.max() > 0.25)
        need_even = p.skin_even > 0 and fit.faces_n and mq_raw.max() > 0.3
        need_deshine = p.deshine > 0 and mq_raw.max() > 0.25
        need_clarity = abs(p.clarity) > 1e-4
        if need_vig or need_3d or need_bg or need_mixed or need_even or need_deshine or need_clarity:
            base_q = R.render(q, lut_bg, lut_sk, mq, gain); lab_q = C.to_lab(base_q); Lq = lab_q[..., 0]
        if need_mixed:
            gain += SP.mixed_light_gain(base_q, mq_raw, [f*qw for f in fit.faces_n], p.mixed_light, lms_q=lms_q)
        if need_even:
            gain += skin_even_gain(base_q, lab_q, mq_raw, [f*qw for f in fit.faces_n], p.skin_even * (1.0 - 0.55 * p.mixed_light))
        if need_deshine:
            gain += SP.deshine_gain(base_q, mq_raw, p.deshine)
        if need_3d:
            gain += fit.relight.gain(qh, qw, Lq, p.plastic, p.light_angle, rgb_q=base_q,
                                     light_x=p.light_x, light_y=p.light_y)[..., None]
        if need_bg:
            gain += fit.relight.bg_gain(qh, qw, Lq, p.bg_exposure, skin_mask=mq_raw, rgb_q=base_q)[..., None]
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
        if db_enabled and (p.blemish > 0 or p.eyes_teeth > 0) and fit.faces_n:
            lms = getattr(fit, "lms_n", None)
            out = SP.apply_portrait_details(out, mq_raw, [f*W for f in fit.faces_n],
                                            [l*W for l in lms] if lms else None,
                                            blemish=p.blemish, eyes_teeth=p.eyes_teeth)
        if p.sharpen_amount > 1e-4 or abs(p.texture) > 1e-4 or p.denoise > 1e-4:
            out = DET.apply_detail_pass(out, mq_raw, sharpen_amount=p.sharpen_amount,
                                        sharpen_mode=p.sharpen_mode, texture=p.texture, denoise=p.denoise)
        T["dodge_burn_s"] = time.perf_counter()-t
        return (out, mq_raw) if return_mask else out

    # ---------- convenience ----------
    def process(self, src_rgb, ref: RefAnalysis, p: Params, timings=None):
        fit = self.fit(src_rgb, ref)
        if timings is not None: timings.update(fit.timings)
        return self.render(src_rgb, fit, p, timings=timings)
