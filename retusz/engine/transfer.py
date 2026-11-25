"""Transfer grading + relighting from a low-resolution AI version onto the full-resolution original.

prepare(): align (engine.align) -> AI resampled into the work frame -> map of places where the AI changed
the picture's structure (local normalised cross-correlation of high-pass luminance, invariant to the
AI's local gain) -> Bilateral Guided Upsampling fit original -> AI with those places and the area
outside the AI frame weighted 0 (they get the neighbours' / global transform, never AI content).
render(): slice the grid on the full-resolution original in strips, then mix with the original in
CIELAB: a*b* by `kolor`, L* by `jasnosc` x optional brush mask (Photoshop Color / Luminosity idea)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np, cv2
from . import align as AL, bgu

PREFILTER = 3.0     # work px
COARSE_CELL = 48.0  # work px

@dataclass
class TransferParams:
    kolor: float = 1.0          # 0..1, a*b* from the AI version
    jasnosc: float = 1.0        # 0..1, L* (light) from the AI version
    protect: bool = True        # keep the original where the AI changed content

    def key(self): return (round(self.kolor, 3), round(self.jasnosc, 3), self.protect)

@dataclass
class Prepared:
    al: AL.Alignment
    grid: bgu.Grid               # protected fit (changed areas weighted 0)
    grid_raw: bgu.Grid           # unprotected fit
    valid: np.ndarray            # work frame: 1 where the AI covers the original
    changed: np.ndarray          # work frame: 0..1, AI changed the structure here
    ai_work: np.ndarray          # AI resampled into the work frame, float32
    full_wh: tuple = (0, 0)      # size of the original the alignment was made on
    extra: dict = field(default_factory=dict)

    @property
    def changed_frac(self):
        v = self.valid.sum()
        return float((self.changed*self.valid).sum()/max(v, 1))

    @property
    def coverage_frac(self): return float(self.valid.mean())

    def coverage_full(self, shape):
        return cv2.resize(self.valid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

# ---------------------------------------------------------------- CIEDE2000 (Sharma, Wu, Dalal 2005)
def delta_e2000(lab1, lab2):
    L1, a1, b1 = [np.asarray(lab1, np.float64)[..., i] for i in range(3)]
    L2, a2, b2 = [np.asarray(lab2, np.float64)[..., i] for i in range(3)]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2); Cm = (C1+C2)/2
    G = 0.5*(1 - np.sqrt(Cm**7/(Cm**7 + 25.0**7)))
    a1p, a2p = (1+G)*a1, (1+G)*a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360; h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2-L1; dCp = C2p-C1p
    dh = h2p-h1p; dh = np.where(dh > 180, dh-360, np.where(dh < -180, dh+360, dh)); dh = np.where(C1p*C2p == 0, 0, dh)
    dHp = 2*np.sqrt(C1p*C2p)*np.sin(np.radians(dh/2))
    Lpm = (L1+L2)/2; Cpm = (C1p+C2p)/2
    hs = h1p+h2p
    hpm = np.where(C1p*C2p == 0, hs, np.where(np.abs(h1p-h2p) <= 180, hs/2, np.where(hs < 360, (hs+360)/2, (hs-360)/2)))
    T = 1 - 0.17*np.cos(np.radians(hpm-30)) + 0.24*np.cos(np.radians(2*hpm)) + 0.32*np.cos(np.radians(3*hpm+6)) - 0.20*np.cos(np.radians(4*hpm-63))
    dth = 30*np.exp(-((hpm-275)/25)**2)
    Rc = 2*np.sqrt(Cpm**7/(Cpm**7 + 25.0**7))
    Sl = 1 + 0.015*(Lpm-50)**2/np.sqrt(20 + (Lpm-50)**2); Sc = 1 + 0.045*Cpm; Sh = 1 + 0.015*Cpm*T
    Rt = -np.sin(np.radians(2*dth))*Rc
    return np.sqrt((dLp/Sl)**2 + (dCp/Sc)**2 + (dHp/Sh)**2 + Rt*(dCp/Sc)*(dHp/Sh))

def to_lab(rgb01):
    return cv2.cvtColor(np.ascontiguousarray(np.clip(rgb01, 0, 1), np.float32), cv2.COLOR_RGB2Lab)

# ---------------------------------------------------------------- changed-content map
def _ncc_score(work, ai_w, hp_sigma, win, valid):
    """Normalised convolution inside `valid` only: pixels outside the AI frame (replicated by remap) must not
    lower the correlation of the windows next to the frame edge."""
    v = valid.astype(np.float32)
    def nblur(x, s): return cv2.GaussianBlur(x*v, (0, 0), s)/(cv2.GaussianBlur(v, (0, 0), s) + 1e-6)
    def hp(x):
        g = AL.gray(x); return (g - nblur(g, hp_sigma))*v
    a, b = hp(work), hp(ai_w)
    va, vb, cab = nblur(a*a, win), nblur(b*b, win), nblur(a*b, win)
    ncc = cab/np.sqrt(va*vb + 0.004**4)
    energy = np.sqrt(np.maximum(va, vb))
    gate = np.clip((energy - 0.006)/0.010, 0, 1)
    return np.clip((0.6 - ncc)/0.35, 0, 1)*gate*v

def changed_map(work, ai_w, valid, scales=((1.5, 4.0), (5.0, 12.0))):
    """0..1 where the AI's structure does not follow the original's (removed / added / redrawn), at a
    fine and a medium scale. Local normalised cross-correlation ignores the AI's local gain and colour,
    so regrading and relighting are not flagged."""
    score = np.maximum.reduce([_ncc_score(work, ai_w, h, w, valid) for h, w in scales])
    score = cv2.GaussianBlur(score, (0, 0), 2.0)
    m = np.clip((score - 0.25)/0.25, 0, 1)
    m = cv2.dilate(m, np.ones((5, 5), np.uint8))
    return (cv2.GaussianBlur(m, (0, 0), 1.5)*valid).astype(np.float32)

def prepare(orig, ai, flow="auto", protect=True, progress=None) -> Prepared:
    """orig: full-resolution RGB (uint8/uint16/float), ai: AI version RGB in the same colour space."""
    al = AL.register(orig, ai, flow=flow)
    if progress: progress("dopasowanie", al)
    ai_w, valid = al.warp_to_work(ai)
    work = AL.to_float(al.work)
    changed = changed_map(work, ai_w, valid)
    # Equal pre-blur of both sides: the AI copy has lost the finest detail (resampling, JPEG); fitting a
    # sharp guide to a soft target would read that loss as lower contrast and flatten the result.
    bw, ba = cv2.GaussianBlur(work, (0, 0), PREFILTER), cv2.GaussianBlur(ai_w, (0, 0), PREFILTER)
    def two_level(w):           # coarse grid (smooth relight + tone curve) as the prior of the fine one
        coarse = bgu.fit(bw, ba, w, cell=COARSE_CELL, sigma_xy=1.0)
        return bgu.fit(bw, ba, w, prior_grid=coarse)
    raw = two_level(valid)
    grid = two_level(valid*(1 - changed)) if protect else raw
    return Prepared(al, grid, raw, valid, changed, ai_w, full_wh=(orig.shape[1], orig.shape[0]))

def _mask_rows(mask, y0, y1, W, H):
    mh, mw = mask.shape[:2]
    xs = ((np.arange(W)+0.5)*mw/W - 0.5).astype(np.float32); ys = ((np.arange(y0, y1)+0.5)*mh/H - 0.5).astype(np.float32)
    mx = np.broadcast_to(xs[None, :], (y1-y0, W)).copy(); my = np.broadcast_to(ys[:, None], (y1-y0, W)).copy()
    return cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def render_float(prep: Prepared, orig, params: TransferParams, mask=None, y0=0, y1=None):
    """Rows y0..y1 of the result as float32 0..1 (at the size of `orig`: full resolution or a preview)."""
    H, W = orig.shape[:2]; y1 = H if y1 is None else y1
    o = AL.to_float(orig[y0:y1])
    g = prep.grid if params.protect else prep.grid_raw
    al = prep.al
    fw, fh = prep.full_wh if prep.full_wh[0] else (W, H)
    sx, sy = al.sx*fw/W, al.sy*fh/H                         # a smaller preview of the same frame works too
    r = np.clip(bgu.slice_(o, g, sx, sy, y0=y0), 0, 1)
    k, j = float(params.kolor), float(params.jasnosc)
    if k >= 0.999 and j >= 0.999 and mask is None: return r
    Lo, Lr = to_lab(o), to_lab(r)
    jm = j if mask is None else j*np.clip(_mask_rows(mask, y0, y1, W, H), 0, 1)
    out = Lo.copy()
    out[..., 0] += jm*(Lr[..., 0]-Lo[..., 0]) if np.ndim(jm) == 0 else jm*(Lr[..., 0]-Lo[..., 0])
    out[..., 1:] += k*(Lr[..., 1:]-Lo[..., 1:])
    return np.clip(cv2.cvtColor(out, cv2.COLOR_Lab2RGB), 0, 1)

def render(prep: Prepared, orig, params: TransferParams, mask=None, rows=512, out_dtype=None):
    """Full-resolution result with the original's dtype (uint8 / uint16) unless out_dtype is given."""
    H, W = orig.shape[:2]
    dt = np.dtype(out_dtype or (orig.dtype if orig.dtype in (np.uint8, np.uint16) else np.float32))
    out = np.empty((H, W, 3), dt)
    mx = 255.0 if dt == np.uint8 else 65535.0 if dt == np.uint16 else 1.0
    for y in range(0, H, rows):
        f = render_float(prep, orig, params, mask, y, min(H, y+rows))
        out[y:y+rows] = (f*mx + (0.5 if mx > 1 else 0)).astype(dt) if mx > 1 else f
    return out
