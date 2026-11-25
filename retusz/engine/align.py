"""Registration of a low-resolution AI-edited copy onto the full-resolution original.

1. SIFT keypoints (OpenCV, Apache-2.0) + RANSAC affine: scale, crop, shift, small rotation, aspect.
2. ECC refinement (Evangelidis & Psarakis 2008, cv2.findTransformECC, MOTION_AFFINE).
3. Optional DIS optical flow (Kroeger et al. 2016, OpenCV) for small local drifts of generative tools,
   median-filtered, smoothed and clamped.
All matching runs on local-contrast-normalised luminance, which is insensitive to the AI's new grading
and relighting. The "work" frame is the original resized to the AI's pixel density."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, cv2

class AlignmentError(RuntimeError):
    pass

FLOW_MAX = 4.0          # work px
LCN_SIGMA = 6.0

def to_float(rgb):
    if rgb.dtype == np.uint8: return rgb.astype(np.float32)/255
    if rgb.dtype == np.uint16: return rgb.astype(np.float32)/65535
    return rgb.astype(np.float32)

def gray(rgb):
    f = to_float(rgb)
    return np.ascontiguousarray(f @ np.array([0.2126, 0.7152, 0.0722], np.float32)) if f.ndim == 3 else f

def lcn(g, sigma=LCN_SIGMA):
    m = cv2.GaussianBlur(g, (0, 0), sigma)
    d = g - m
    v = cv2.GaussianBlur(d*d, (0, 0), sigma)
    return (d/np.sqrt(v + 0.03**2)).astype(np.float32)

def to8(n):
    return np.clip(n*45 + 128, 0, 255).astype(np.uint8)

def resize_to(img, wh):
    return cv2.resize(img, wh, interpolation=cv2.INTER_AREA)

def _h(M):
    return np.vstack([M, [0, 0, 1]])

def _scale_map(sx, sy):
    """full -> resized frame, pixel-centre convention."""
    return np.array([[sx, 0, 0.5*sx-0.5], [0, sy, 0.5*sy-0.5]], np.float64)

@dataclass
class Alignment:
    M: np.ndarray            # 2x3, work -> ai
    sx: float                # work px per full px (x)
    sy: float
    work_wh: tuple
    ai_wh: tuple
    flow: np.ndarray | None  # work-frame (H, W, 2) offsets applied before M
    residual_px: float       # full-res px
    ncc: float
    inliers: int

    @property
    def scale(self):
        """AI pixels per original pixel."""
        return float(np.sqrt(abs(np.linalg.det(self.M[:, :2])))*np.sqrt(self.sx*self.sy))

    def full_to_work(self, P):
        P = np.asarray(P, np.float64)
        return np.c_[self.sx*(P[:, 0]+0.5)-0.5, self.sy*(P[:, 1]+0.5)-0.5]

    def full_to_ai(self, P):
        Q = self.full_to_work(P)
        if self.flow is not None:
            fx = cv2.remap(self.flow[..., 0], Q[:, 0:1].astype(np.float32), Q[:, 1:2].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fy = cv2.remap(self.flow[..., 1], Q[:, 0:1].astype(np.float32), Q[:, 1:2].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            Q = Q + np.c_[fx.ravel(), fy.ravel()]
        return (self.M @ np.c_[Q, np.ones(len(Q))].T).T

    def maps(self):
        """remap() maps from the work frame into AI pixel coordinates."""
        w, h = self.work_wh
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        if self.flow is not None: xx = xx + self.flow[..., 0]; yy = yy + self.flow[..., 1]
        M = self.M.astype(np.float32)
        return M[0, 0]*xx + M[0, 1]*yy + M[0, 2], M[1, 0]*xx + M[1, 1]*yy + M[1, 2]

    def warp_to_work(self, ai, interp=cv2.INTER_LINEAR):
        """AI image resampled into the work frame + validity mask (1 where the AI covers the original)."""
        mx, my = self.maps()
        out = cv2.remap(to_float(ai), mx, my, interp, borderMode=cv2.BORDER_REPLICATE)
        wa, ha = self.ai_wh
        valid = ((mx >= 0) & (mx <= wa-1) & (my >= 0) & (my <= ha-1)).astype(np.float32)
        valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
        return out, valid

def _sift_affine(a8, b8):
    sift = cv2.SIFT_create(nfeatures=5000)
    ka, da = sift.detectAndCompute(a8, None); kb, db = sift.detectAndCompute(b8, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        raise AlignmentError("Za mało szczegółów do dopasowania wersji AI do oryginału.")
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75*p[1].distance]
    if len(good) < 12:
        raise AlignmentError("Wersja AI nie pasuje do oryginału (za mało wspólnych punktów).")
    pa = np.float32([ka[g.queryIdx].pt for g in good]); pb = np.float32([kb[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffine2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.5, maxIters=8000, confidence=0.999)
    n = int(inl.sum()) if inl is not None else 0
    if M is None or n < 20 or n < 0.15*len(good):
        raise AlignmentError("Wersja AI nie pasuje do oryginału (dopasowanie niespójne).")
    sv = np.linalg.svd(M[:, :2], compute_uv=False)
    if sv[1] <= 0 or sv[0]/sv[1] > 1.3:
        raise AlignmentError("Wersja AI jest zniekształcona względem oryginału.")
    return M.astype(np.float64), n

def _ecc(t, i, M, mask):
    warp = M.astype(np.float32).copy()
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-7)
    try:
        cc, warp = cv2.findTransformECC(t, i, warp, cv2.MOTION_AFFINE, crit, mask, 5)
    except cv2.error:
        return M, 0.0
    return warp.astype(np.float64), float(cc)

def _residual(t, w, valid, tile=96):
    """Median sub-pixel shift between template and warped AI over textured tiles (work px)."""
    h, wd = t.shape; shifts = []
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F)
    for y in range(0, h-tile+1, tile):
        for x in range(0, wd-tile+1, tile):
            if valid[y:y+tile, x:x+tile].min() < 1: continue
            a = t[y:y+tile, x:x+tile]; b = w[y:y+tile, x:x+tile]
            if a.std() < 0.3: continue
            (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
            if resp > 0.05 and abs(dx) < 8 and abs(dy) < 8: shifts.append(np.hypot(dx, dy))
    return float(np.median(shifts)) if shifts else float("nan")

def _flow(t, w, valid):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    f = dis.calc(to8(t), to8(w), None)
    f = np.dstack([cv2.medianBlur(np.ascontiguousarray(f[..., c]), 5) for c in range(2)])
    wv = cv2.GaussianBlur(valid, (0, 0), 12) + 1e-4
    f = np.dstack([cv2.GaussianBlur(f[..., c]*valid, (0, 0), 12)/wv for c in range(2)])
    return np.clip(f, -FLOW_MAX, FLOW_MAX).astype(np.float32)

def register(orig, ai, flow=False, orig_small=None) -> Alignment:
    """Align `ai` (any size) to `orig` (full resolution). Raises AlignmentError when they do not match."""
    H, W = orig.shape[:2]; ha, wa = ai.shape[:2]
    s0 = min(1.0, 1.1*max(ha, wa)/max(H, W))
    w0, h0 = max(1, round(W*s0)), max(1, round(H*s0))
    o0 = resize_to(orig, (w0, h0)) if orig_small is None else orig_small
    ga, gb = gray(o0), gray(ai)
    M0, n = _sift_affine(to8(lcn(ga, 3)), to8(lcn(gb, 3)))
    sig = np.sqrt(abs(np.linalg.det(M0[:, :2])))
    s = min(1.0, s0*sig)
    ww, hw = max(16, round(W*s)), max(16, round(H*s))
    sx, sy = ww/W, hw/H
    work = resize_to(orig, (ww, hw))
    C = _h(_scale_map(w0/W, h0/H)) @ np.linalg.inv(_h(_scale_map(sx, sy)))   # work -> small0
    M = (_h(M0) @ C)[:2]
    t = lcn(gray(work)); i = lcn(gb)
    al = Alignment(M, sx, sy, (ww, hw), (wa, ha), None, 0.0, 0.0, n)
    al.M, cc = _ecc(t, i, al.M, None)       # inputMask is in AI coordinates; the AI is valid everywhere
    al.ncc = cc
    if cc < 0.2:
        raise AlignmentError("Wersja AI nie pasuje do oryginału (niska korelacja po dopasowaniu).")
    wi, valid = al.warp_to_work(i)
    if flow:
        al.flow = _flow(t, wi, valid)
        wi, valid = al.warp_to_work(i)
    al.residual_px = _residual(t, wi, valid)/np.sqrt(sx*sy)
    al.work = work
    return al
