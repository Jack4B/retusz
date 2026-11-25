"""'Plastyka 3D': relighting from a single image.

1. Relative depth from Depth Anything V2 Small (Apache-2.0, ONNX int8, 27 MB); fallback without the
   model: luminance-based pseudo-depth (shape-from-shading style, large-scale blurred L + face prior).
2. Normals from depth gradients (normalised per image so relative depth units do not matter).
3. Directional soft light: Lambert term relative to a camera-facing flat surface (flat areas unchanged).
4. Volume D&B from depth curvature (ridges up, hollows down), local contrast ("clarity"),
   all masked to the subjects (depth foreground ∪ skin). 0.1.6: no background darkening (with a soft,
   depth-derived outline it drew a dark glow around people and flattened the picture); the subject mask is
   snapped to the image's own edges with a guided filter, shading is spread only inside the subject and the
   depth curvature is faded out at the silhouette (the depth step there read as a ridge: a bright rim).
5. Light direction 'auto' = least-squares fit of L ≈ c0 + c·n on the reference subjects.
Everything ends up as one quarter-resolution multiplicative gain map, so the full-res cost is ~0."""
from __future__ import annotations
import threading
import numpy as np, cv2
from ..paths import bundled_model

DEPTH_MODEL = "depth_anything_v2_small_q8.onnx"
_sess = None; _lock = threading.Lock()
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)

def _session():
    global _sess
    if _sess is None:
        from . import device
        p = bundled_model(DEPTH_MODEL)
        if not p.exists(): return None
        _sess = device.make_session(p, lambda s: device.zeros_feed(s, (1, 3, 518, 784)))
    return _sess

def reset_session():
    global _sess
    _sess = None

def depth_ai(rgb, side=518):
    """Relative inverse depth (bigger = closer), normalised 0..1, at a size with long side ~= side."""
    with _lock:
        s = _session()
        if s is None: return None
        h, w = rgb.shape[:2]; k = side/max(h, w)
        nh, nw = max(14, int(round(h*k/14))*14), max(14, int(round(w*k/14))*14)
        x = (cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32)/255 - MEAN)/STD
        d = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0].astype(np.float32)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99.5)
    return np.clip((d-lo)/(hi-lo+1e-6), 0, 1)

def depth_luma(rgb, faces_n=(), side=518):
    """Fallback pseudo-depth: brighter large-scale areas read as closer, faces pushed forward."""
    h, w = rgb.shape[:2]; k = side/max(h, w); sm = cv2.resize(rgb, (int(w*k), int(h*k)), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(sm, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)/255
    d = cv2.GaussianBlur(L, (0, 0), sm.shape[1]/60)*0.6
    for f in faces_n:
        x, y, fw, fh = f*sm.shape[1]
        g = np.zeros_like(d); cv2.ellipse(g, (int(x+fw/2), int(y+fh*0.6)), (int(fw*0.7), int(fh*1.0)), 0, 0, 360, 1, -1)
        cv2.ellipse(g, (int(x+fw/2), int(y+fh*3.2)), (int(fw*1.6), int(fh*2.6)), 0, 0, 360, 0.7, -1)
        d = np.maximum(d, cv2.GaussianBlur(g, (0, 0), fw*0.35)*0.9)
    lo, hi = np.percentile(d, 1), np.percentile(d, 99.5)
    return np.clip((d-lo)/(hi-lo+1e-6), 0, 1)

def normals(depth, subject=None):
    sg = max(1.0, depth.shape[1]/400)
    if subject is not None and (subject > 0.5).sum() > 100:
        # smooth the subject's depth on its own: the step to the far background at the silhouette would give
        # near-horizontal normals there, i.e. a dark (or bright) outline around the person
        k = 2*max(1, depth.shape[1]//100) + 1                 # the depth model's outline is a few px too wide
        w = cv2.erode((subject > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(np.float32)
        wide = max(sg, depth.shape[1]/60)
        fill = cv2.GaussianBlur(depth*w, (0, 0), wide)/(cv2.GaussianBlur(w, (0, 0), wide) + 1e-6)
        ws = cv2.GaussianBlur(w, (0, 0), max(1.0, depth.shape[1]/150))*w      # smooth seam: no curvature valley
        depth = ws*depth + (1 - ws)*fill
    d = cv2.GaussianBlur(depth, (0, 0), sg)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = np.hypot(gx, gy); ref = mag[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else mag
    k = np.tan(np.radians(55))/(np.percentile(ref, 90)+1e-6)   # 90th pct slope -> 55 degrees
    n = np.dstack([-gx*k, -gy*k, np.ones_like(d)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    curv = -cv2.Laplacian(cv2.GaussianBlur(d, (0, 0), depth.shape[1]/150), cv2.CV_32F)
    c_ref = curv[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else curv
    curv = np.clip(curv/(np.percentile(np.abs(c_ref), 95)+1e-6), -1.5, 1.5)
    return n.astype(np.float32), curv.astype(np.float32)

def subject_mask(depth, skin=None, faces_n=()):
    """Soft subject mask at the depth map's size: depth foreground (Otsu) united with skin. With faces, only the
    foreground parts that hold a face are kept (the subjects are the people): background objects the depth model
    reads as near, and skin-coloured background patches, are dropped; an erosion first cuts thin bridges to them."""
    u8 = (depth*255).astype(np.uint8); t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fg = np.clip((depth - t/255)/0.08 + 0.5, 0, 1)
    sk = cv2.resize(skin, (depth.shape[1], depth.shape[0])) if skin is not None else None
    if sk is not None: fg = np.maximum(fg, sk)
    k = max(3, depth.shape[1]//80) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if len(faces_n):
        h, w = depth.shape; seed = np.zeros((h, w), np.uint8)
        for f in faces_n:
            x, y, fw, fh = np.asarray(f, np.float64)*w
            cv2.ellipse(seed, (int(x + fw/2), int(y + fh/2)), (max(1, int(fw/3)), max(1, int(fh/3))), 0, 0, 360, 1, -1)
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))
        core = cv2.erode((fg > 0.5).astype(np.uint8), ke) | seed
        n, lab = cv2.connectedComponents(core, connectivity=8)
        keep = np.zeros(n, np.uint8); keep[np.unique(lab[(seed > 0) & (lab > 0)])] = 1; keep[0] = 0
        if keep.any(): fg = fg*cv2.dilate(keep[lab], ke).astype(np.float32)
    return cv2.GaussianBlur(fg, (0, 0), depth.shape[1]/150).astype(np.float32)

def light_vec(angle_deg, elev_deg=45.0):
    """angle: 0 = light from the right, 90 = from the top, 180 = left (image coords, y down)."""
    a, e = np.radians(angle_deg), np.radians(elev_deg)
    return np.array([np.cos(a)*np.cos(e), -np.sin(a)*np.cos(e), np.sin(e)], np.float32)

def estimate_light(rgb, n, fg):
    """Least squares L ≈ c0 + c·n on subject pixels; returns angle in degrees or None."""
    L = cv2.cvtColor(cv2.resize(rgb, (n.shape[1], n.shape[0]), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    m = fg > 0.6
    if m.sum() < 200: return None
    A = np.column_stack([np.ones(m.sum()), n[..., 0][m], n[..., 1][m]]); c, *_ = np.linalg.lstsq(A, L[m], rcond=None)
    if np.hypot(c[1], c[2]) < 1e-3: return None
    return float(np.degrees(np.arctan2(-c[2], c[1])) % 360)

def guided_refine(I, p, r, eps):
    """Guided filter (He, Sun, Tang 2010) of p with guide I: a soft mask whose edges follow I's edges."""
    box = lambda x: cv2.boxFilter(x, -1, (2*r+1, 2*r+1))
    mI, mp = box(I), box(p)
    a = (box(I*p) - mI*mp)/(box(I*I) - mI*mI + eps); b = mp - a*mI
    return box(a)*I + box(b)

def guided_filter(I, r, eps):
    """Self-guided edge-preserving smoothing (He, Sun, Tang 2010), box filters only."""
    box = lambda x: cv2.boxFilter(x, -1, (2*r+1, 2*r+1))
    m = box(I); v = box(I*I) - m*m
    a = v/(v+eps); b = m - a*m
    return box(a)*I + box(b)

class Relight:
    """Per-image 3D data, computed once in fit() on the analysis image."""
    def __init__(self, rgb, skin=None, faces_n=(), use_ai=True):
        d = depth_ai(rgb) if use_ai else None
        self.method = "Depth Anything V2" if d is not None else "jasność (bez modelu)"
        self.depth = d if d is not None else depth_luma(rgb, faces_n)
        self.fg = subject_mask(self.depth, skin, faces_n)
        self.n, self.curv = normals(self.depth, self.fg)
        self.auto_angle = estimate_light(rgb, self.n, self.fg)

    def gain(self, qh, qw, Lq, amount, angle=None):
        """Multiplicative gain (relative) at qh x qw. Lq = rendered L (0..100) at the same size."""
        if amount <= 0: return np.zeros((qh, qw), np.float32)
        ang = angle if angle is not None else (self.auto_angle if self.auto_angle is not None else 120.0)
        l = light_vec(ang)
        rs = lambda a: cv2.resize(a, (qw, qh), interpolation=cv2.INTER_LINEAR)
        n = rs(self.n); curv = rs(self.curv)
        # subject mask with the image's own edges (the depth model's outline is soft and a few px off)
        # (binarised first: its soft tail reaches far into the background and would carry the sculpt there;
        # then snapped twice: a radius covering the depth model's error (~3 px at 518), then a fine one)
        I = Lq.astype(np.float32)/100
        fg = np.clip(guided_refine(I, (rs(self.fg) > 0.5).astype(np.float32), max(2, max(qw, qh)//60), 1e-3), 0, 1)
        fg = np.clip(guided_refine(I, (fg > 0.5).astype(np.float32), max(1, max(qw, qh)//300), 1e-3), 0, 1)
        shade = np.clip(n@l - l[2], -0.8, 0.5)                      # flat, camera-facing = 0
        sb = max(qw, qh)/300                                         # spread inside the subject only
        shade = cv2.GaussianBlur(shade*fg, (0, 0), sb)/(cv2.GaussianBlur(fg, (0, 0), sb) + 1e-4)
        # near the silhouette (~3 % of the frame) the depth model's step gives grazing normals and a curvature
        # valley: there the curvature fades out and the shading blends into the subject's broad shading, so the
        # volume falls off smoothly instead of drawing a dark line just inside the outline
        band = cv2.GaussianBlur(np.clip(2*fg - 1, 0, 1), (0, 0), max(1.0, max(qw, qh)/60))**2
        curv = curv*band
        wb = max(qw, qh)/40
        broad = cv2.GaussianBlur(shade*fg, (0, 0), wb)/(cv2.GaussianBlur(fg, (0, 0), wb) + 1e-4)
        shade = band*shade + (1 - band)*broad
        # local contrast on an edge-preserving base (guided filter): no halos around glasses, hair or shoulders
        Lb = guided_filter(Lq.astype(np.float32), max(2, int(max(qw, qh)/60)), 6.0**2)
        clarity = np.clip(Lq-Lb, -6, 6)/np.maximum(Lq, 25)
        g = fg*(0.38*shade + 0.06*curv + 0.15*clarity)              # subject sculpt only, background untouched
        g = np.where(g > 0, g*np.clip((97-Lq)/25, 0, 1), g)          # no blown highlights
        return (amount*g).astype(np.float32)
