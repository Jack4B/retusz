"""'Plastyka 3D': relighting from a single image.

1. Relative depth from Depth Anything V2 Small (Apache-2.0, ONNX int8, 27 MB); fallback without the
   model: luminance-based pseudo-depth (shape-from-shading style, large-scale blurred L + face prior).
2. Normals from depth gradients (normalised per image so relative depth units do not matter).
3. Directional soft light: Lambert term relative to a camera-facing flat surface (flat areas unchanged).
4. Volume D&B from depth curvature (ridges up, hollows down), local contrast ("clarity"),
   subject/background separation. All masked to the subjects (depth foreground ∪ skin).
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
    d = cv2.GaussianBlur(depth, (0, 0), max(1.0, depth.shape[1]/400))
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = np.hypot(gx, gy); ref = mag[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else mag
    k = np.tan(np.radians(55))/(np.percentile(ref, 90)+1e-6)   # 90th pct slope -> 55 degrees
    n = np.dstack([-gx*k, -gy*k, np.ones_like(d)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    curv = -cv2.Laplacian(cv2.GaussianBlur(d, (0, 0), depth.shape[1]/150), cv2.CV_32F)
    c_ref = curv[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else curv
    curv = np.clip(curv/(np.percentile(np.abs(c_ref), 95)+1e-6), -1.5, 1.5)
    return n.astype(np.float32), curv.astype(np.float32)

def subject_mask(depth, skin=None):
    u8 = (depth*255).astype(np.uint8); t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fg = np.clip((depth - t/255)/0.08 + 0.5, 0, 1)
    if skin is not None:
        fg = np.maximum(fg, cv2.resize(skin, (depth.shape[1], depth.shape[0])))
    k = max(3, depth.shape[1]//80) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
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
        self.fg = subject_mask(self.depth, skin)
        self.n, self.curv = normals(self.depth, self.fg)
        self.auto_angle = estimate_light(rgb, self.n, self.fg)

    def gain(self, qh, qw, Lq, amount, angle=None):
        """Multiplicative gain (relative) at qh x qw. Lq = rendered L (0..100) at the same size."""
        if amount <= 0: return np.zeros((qh, qw), np.float32)
        ang = angle if angle is not None else (self.auto_angle if self.auto_angle is not None else 120.0)
        l = light_vec(ang)
        rs = lambda a: cv2.resize(a, (qw, qh), interpolation=cv2.INTER_LINEAR)
        n = rs(self.n); fg = rs(self.fg); curv = rs(self.curv)
        shade = np.clip(n@l - l[2], -0.8, 0.5)                      # flat, camera-facing = 0
        shade = cv2.GaussianBlur(shade, (0, 0), max(qw, qh)/300)
        # local contrast on an edge-preserving base (guided filter): no halos around glasses, hair or shoulders
        Lb = guided_filter(Lq.astype(np.float32), max(2, int(max(qw, qh)/60)), 6.0**2)
        clarity = np.clip(Lq-Lb, -6, 6)/np.maximum(Lq, 25)
        g = fg*(0.38*shade + 0.06*curv + 0.15*clarity) - 0.08*(1-fg)  # subject sculpt + gentle separation
        g = np.where(g > 0, g*np.clip((97-Lq)/25, 0, 1), g)          # no blown highlights
        return (amount*g).astype(np.float32)
