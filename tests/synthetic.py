"""Repo-safe synthetic images for the AI grading-transfer tests and gate (no real photos)."""
from __future__ import annotations
import numpy as np, cv2

def textured(h=1200, w=1800, seed=0):
    """Photo-like test frame: multi-scale coloured texture, sharp-edged shapes, fine lines.
    Returns (rgb uint8, shape_mask float32 1 inside the big disc)."""
    r = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.float32)
    for sigma, amp in ((60, 0.35), (16, 0.18), (4, 0.10), (1.2, 0.06)):
        n = r.standard_normal((h, w, 3)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma); n /= n.std() + 1e-6
        img += amp*n
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img += np.dstack([0.15*x/w, 0.1*y/h, -0.1*x/w])
    img = 0.5 + 0.22*img
    for _ in range(14):                                    # hard-edged rectangles
        x0, y0 = int(r.uniform(0, w-200)), int(r.uniform(0, h-160)); ww, hh = int(r.uniform(40, 220)), int(r.uniform(30, 160))
        img[y0:y0+hh, x0:x0+ww] = img[y0:y0+hh, x0:x0+ww]*0.5 + r.uniform(0.1, 0.9, 3)*0.5
    for _ in range(40):                                    # thin lines (sharpness probes)
        p0 = (int(r.uniform(0, w)), int(r.uniform(0, h))); p1 = (int(r.uniform(0, w)), int(r.uniform(0, h)))
        c = tuple(float(v) for v in r.uniform(0, 1, 3)); cv2.line(img, p0, p1, c, 1, cv2.LINE_AA)
    cx, cy, rad = w*0.62, h*0.45, min(h, w)*0.22
    disc = (((x-cx)**2 + (y-cy)**2) < rad**2).astype(np.float32)
    img = img*(1-disc[..., None]*0.35) + disc[..., None]*np.array([0.75, 0.55, 0.45], np.float32)*0.35
    return (np.clip(img, 0, 1)*255 + 0.5).astype(np.uint8), disc

def guided_gray(I, p, r, eps):
    """He et al. 2010 guided filter, grey guide (box filters)."""
    B = lambda x: cv2.boxFilter(x, -1, (2*r+1, 2*r+1))
    mI, mp = B(I), B(p); a = (B(I*p) - mI*mp)/(B(I*I) - mI*mI + eps); b = mp - a*mI
    return B(a)*I + B(b)

def content_mask(rgb, q=70, hard=False):
    """Local-grade mask for real photos: bright regions (sky, windows, light objects), refined with a guided
    filter so it follows the photo's own edges like a real selection/AI mask does. hard=True keeps the raw
    threshold (its contour cuts through smooth gradients: an artificial edge no photo has - stress case)."""
    Y = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    m = (cv2.GaussianBlur(Y, (0, 0), 3) > np.percentile(Y, q)).astype(np.float32)
    return m if hard else np.clip(guided_gray(Y, m, 8, 1e-3), 0, 1)

def grade(rgb, mask=None, relight=True):
    """Ground-truth 'AI' grading: tone curve, teal/orange colour matrix, smooth relight, sharp local grade."""
    x = rgb.astype(np.float32)/255
    h, w = x.shape[:2]
    x = np.power(np.clip(x, 0, 1), 0.85)
    x = x + 0.18*x*(1-x)*(2*x-1)                           # S-curve
    M = np.array([[1.08, 0.03, -0.05], [-0.02, 1.00, 0.02], [-0.07, 0.05, 0.96]], np.float32)
    x = x @ M.T + np.array([0.012, 0.0, 0.02], np.float32)
    if relight:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        g = 0.78 + 0.45*np.exp(-(((xx-0.35*w)/(0.45*w))**2 + ((yy-0.4*h)/(0.6*h))**2))
        x = x*g[..., None]
    if mask is not None:
        m = mask[..., None]
        x = x*(1-m) + m*(x*np.array([0.86, 0.92, 1.06], np.float32) + 0.02)
    return (np.clip(x, 0, 1)*255 + 0.5).astype(np.uint8)

def make_ai(gt, k=3, angle=1.5, shift=(7.3, -4.6), crop=0.06, size=None, jpeg=90):
    """Simulate an AI tool's output: downscale by k, rotate/shift, crop the borders, JPEG.
    Returns (ai rgb uint8, T 2x3 float64 mapping full-res pixel coords -> ai pixel coords)."""
    H, W = gt.shape[:2]
    d = cv2.resize(gt, (round(W/k), round(H/k)), interpolation=cv2.INTER_AREA)
    sx, sy = d.shape[1]/W, d.shape[0]/H
    D = np.array([[sx, 0, 0.5*sx-0.5], [0, sy, 0.5*sy-0.5]], np.float64)   # full -> d (pixel centres)
    hd, wd = d.shape[:2]
    wa, ha = size or (round(wd*(1-2*crop)), round(hd*(1-2*crop)))
    R = cv2.getRotationMatrix2D((wd/2, hd/2), angle, 1.0)                   # d -> rotated d
    R[0, 2] += shift[0] - (wd-wa)/2; R[1, 2] += shift[1] - (hd-ha)/2        # then crop to the centre
    ai = cv2.warpAffine(d, R, (wa, ha), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", ai[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg]); ai = cv2.imdecode(buf, 1)[..., ::-1].copy()
    T = np.vstack([R, [0, 0, 1]]) @ np.vstack([D, [0, 0, 1]])
    return ai, T[:2]

def warp_error_px(T_true, T_est, full_shape, ai_shape, flow_fn=None, n=24):
    """Mean distance (in full-res pixels) between where the true and estimated maps send points of the
    full frame that land inside the AI image."""
    H, W = full_shape[:2]; ha, wa = ai_shape[:2]
    ys, xs = np.meshgrid(np.linspace(0, H-1, n), np.linspace(0, W-1, n), indexing="ij")
    P = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    a = T_true @ P; b = T_est(P[:2].T).T if callable(T_est) else T_est @ P
    inside = (a[0] > 2) & (a[0] < wa-3) & (a[1] > 2) & (a[1] < ha-3)
    s = np.sqrt(abs(np.linalg.det(T_true[:, :2])))
    return float(np.linalg.norm((a-b)[:, inside], axis=0).mean()/s)


def smooth_field(h, w, amp, sigma, seed):
    """Smooth random 2-channel displacement field (px), peak amplitude ~amp."""
    r = np.random.default_rng(seed)
    f = np.stack([cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (0, 0), sigma) for _ in range(2)], -1)
    return (f/(np.abs(f).max() + 1e-9)*amp).astype(np.float32)

def strong_grade(rgb, hue_deg=14.0, cool=0.10, gain=1.25, contrast=1.15):
    """A much stronger, Gemini-like regrade than `grade`: hue rotation, cooler cast, brighter, more contrast, poster-like bright areas."""
    x = rgb.astype(np.float32)/255
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV); hsv[..., 0] = (hsv[..., 0] + hue_deg) % 360; hsv[..., 1] = np.clip(hsv[..., 1]*1.25, 0, 1)
    x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    x = x*np.array([1 - cool, 1.0, 1 + cool], np.float32)*gain
    x = np.clip((x - 0.5)*contrast + 0.5, 0, 1)
    return (np.clip(x, 0, 1)*255 + 0.5).astype(np.uint8)

def make_generative(orig, out_wh, shift=(0.0, 0.0), zoom=1.0, drift_amp=0.012, drift_sigma=0.12, island=None,
                    regen=(), seed=7, grade_fn=strong_grade, soften=0.8, jpeg=88, fine=False):
    """Simulate a generative tool (Gemini / Nano Banana style) re-render of `orig`:
    - own output size (out_wh), a different framing (zoom about the centre, shift in output px),
    - smooth NON-rigid drift of everything (drift_amp = peak shift as a fraction of the output width),
    - `island`: (cx, cy, r, dx, dy) in output fractions, a 'person' that moved relative to the background,
    - `regen`: rects (x0, y0, x1, y1) in output fractions whose content is redrawn (different texture),
    - strong regrade and a bit of lost fine detail, JPEG; fine=True also redraws all fine detail (regen_fine).
    Returns (ai rgb uint8, changed mask float32 at the AI's size)."""
    W, H = out_wh; h, w = orig.shape[:2]
    big = cv2.resize(orig, (W, H), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    fld = smooth_field(H, W, drift_amp*W, drift_sigma*W, seed)
    if island is not None:
        cx, cy, r, dx, dy = island; g = np.exp(-(((xx-cx*W)**2 + (yy-cy*H)**2)/(2*(r*W)**2))).astype(np.float32)
        fld[..., 0] += g*dx*W; fld[..., 1] += g*dy*W
    z = 1.0/zoom
    mx = (xx - W/2)*z + W/2 - shift[0] + fld[..., 0]; my = (yy - H/2)*z + H/2 - shift[1] + fld[..., 1]
    warped = cv2.remap(big, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    ai = grade_fn(warped)
    changed = np.zeros((H, W), np.float32)
    rng = np.random.default_rng(seed + 1)
    for (x0, y0, x1, y1) in regen:
        a, b, c, d = int(x0*W), int(y0*H), int(x1*W), int(y1*H)
        patch, _ = textured(max(d-b, 260), max(c-a, 260), seed=int(rng.integers(1, 10**6))); patch = patch[:d-b, :c-a]
        ai[b:d, a:c] = (0.15*ai[b:d, a:c] + 0.85*grade_fn(patch)).astype(np.uint8); changed[b:d, a:c] = 1
    if fine: ai = regen_fine(ai, seed=seed + 2)
    if soften: ai = cv2.GaussianBlur(ai, (0, 0), soften)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", ai[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, jpeg]); ai = cv2.imdecode(buf, 1)[..., ::-1].copy()
    return ai, changed


def portrait_scene(h=1800, w=1200, seed=0, grain=0.0):
    """Repo-safe stand-in for a studio portrait (no real photo): smooth backdrop with a soft gradient and a
    textured 'poster', a person (skin face with soft modelling, mid-frequency blotches and pores, dark hair with
    fine strands, glasses frame, eyes, lips, textured shirt) and optional sensor grain (std in 8-bit levels).
    Returns (rgb uint8, info dict: masks person/skin/face/poster float32, face (x, y, w, h), lms 5x2 px
    [right eye, left eye, nose, mouth right, mouth left])."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    def blob(cx, cy, rx, ry): return ((xx-cx)/rx)**2 + ((yy-cy)/ry)**2
    def noise(sigma, ch=1):
        n = r.standard_normal((h, w, ch)).astype(np.float32)
        n = np.dstack([cv2.GaussianBlur(n[..., c], (0, 0), sigma) for c in range(ch)]); n = n/(n.std() + 1e-6)
        return n[..., 0] if ch == 1 else n
    S = lambda e: np.clip(1.0 - e, 0, 1)                        # 1 inside an ellipse (e < 1)
    bg = np.dstack([0.36 + 0.10*xx/w, 0.40 + 0.06*yy/h, 0.46 - 0.05*xx/w]) + 0.03*noise(90, 3)
    img = bg.astype(np.float32)
    # poster on the wall (top-left): coloured blocks + text-like strokes
    px0, py0, px1, py1 = int(0.06*w), int(0.08*h), int(0.34*w), int(0.40*h)
    poster = np.zeros((h, w), np.float32); poster[py0:py1, px0:px1] = 1
    pt = 0.55 + 0.12*noise(14, 3) + 0.06*noise(3, 3)
    for _ in range(26):
        y = int(r.uniform(py0+10, py1-10)); x0 = int(r.uniform(px0+8, px1-60)); x1 = x0 + int(r.uniform(20, 90))
        cv2.line(pt, (x0, y), (x1, y), (0.1, 0.1, 0.12), int(r.integers(2, 5)), cv2.LINE_AA)
    img = img*(1-poster[..., None]) + pt*poster[..., None]
    # person
    cx, cy, fw = 0.58*w, 0.36*h, 0.30*w
    torso = (S(blob(cx, 1.0*h, 0.38*w, 0.44*h)) > 0).astype(np.float32)
    neck = (S(blob(cx, cy + 0.75*fw, 0.22*fw, 0.40*fw)) > 0).astype(np.float32)
    head = (S(blob(cx, cy, 0.5*fw, 0.66*fw)) > 0).astype(np.float32)
    hair = np.clip((S(blob(cx, cy - 0.18*fw, 0.58*fw, 0.62*fw)) > 0).astype(np.float32) - head*(yy > cy - 0.36*fw), 0, 1)
    shirt = np.dstack([0.22, 0.30, 0.48]).astype(np.float32) + 0.05*noise(2.0, 3) + 0.06*noise(25, 3)
    img = img*(1-torso[..., None]) + shirt*torso[..., None]
    skin_base = np.array([0.80, 0.60, 0.50], np.float32)
    shade = 1.0 - 0.18*np.clip((xx-cx)/(0.5*fw), -1, 1)**2 - 0.10*np.clip((yy-cy)/(0.66*fw), -1, 1)**2
    blotch = 0.022*noise(0.03*fw)                               # uneven skin (mid frequency), ~2.5 L*
    pores = 0.018*noise(0.8)
    nose = cv2.GaussianBlur((blob(cx + 0.075*fw, cy + 0.05*fw, 0.035*fw, 0.17*fw) < 1).astype(np.float32), (0, 0), 0.012*fw)
    shade = shade - 0.18*nose                                    # a real shading edge (side of the nose) that must stay
    skin = (skin_base*(shade + blotch + pores)[..., None]).astype(np.float32)
    sk = np.clip(head + neck, 0, 1)
    img = img*(1-sk[..., None]) + skin*sk[..., None]
    strands = (noise(1.0)*0.5 + 1.5*cv2.GaussianBlur(r.standard_normal((h, w)).astype(np.float32), (1, 31), 0))[..., None]
    hairc = np.array([0.16, 0.11, 0.08], np.float32)*(1 + 0.25*strands)
    img = img*(1-hair[..., None]) + hairc*hair[..., None]
    # face features: eyes, brows, glasses, lips
    ex, ey, sep = cx, cy - 0.08*fw, 0.20*fw
    lms = np.array([[ex - sep, ey], [ex + sep, ey], [cx, cy + 0.12*fw], [cx - 0.14*fw, cy + 0.32*fw], [cx + 0.14*fw, cy + 0.32*fw]], np.float32)
    feat = np.zeros((h, w), np.float32)
    def put(mask, col):
        nonlocal img
        img = img*(1-mask[..., None]) + np.array(col, np.float32)*mask[..., None]
    for (x, y) in lms[:2]:
        e = cv2.GaussianBlur((blob(x, y, 0.07*fw, 0.035*fw) < 1).astype(np.float32), (0, 0), 1.2); put(e, (0.12, 0.08, 0.07)); feat = np.maximum(feat, e)
        b = cv2.GaussianBlur((blob(x, y - 0.09*fw, 0.09*fw, 0.018*fw) < 1).astype(np.float32), (0, 0), 1.5); put(b, (0.20, 0.13, 0.10)); feat = np.maximum(feat, b)
    lip = cv2.GaussianBlur((blob(cx, cy + 0.32*fw, 0.15*fw, 0.04*fw) < 1).astype(np.float32), (0, 0), 1.5); put(lip*0.8, (0.62, 0.30, 0.30)); feat = np.maximum(feat, lip)
    gl = np.zeros((h, w), np.float32)
    for (x, y) in lms[:2]: cv2.ellipse(gl, (int(x), int(y)), (int(0.11*fw), int(0.075*fw)), 0, 0, 360, 1.0, max(2, int(0.012*fw)), cv2.LINE_AA)
    cv2.line(gl, (int(lms[0, 0] + 0.11*fw), int(ey)), (int(lms[1, 0] - 0.11*fw), int(ey)), 1.0, max(2, int(0.01*fw)), cv2.LINE_AA)
    put(gl, (0.05, 0.05, 0.06)); feat = np.maximum(feat, gl)
    if grain:
        img = img + r.normal(0, grain/255, img.shape).astype(np.float32)
    rgb = (np.clip(img, 0, 1)*255 + 0.5).astype(np.uint8)
    person = np.clip(torso + sk + hair, 0, 1)
    skinm = sk*(1 - hair)*(1 - np.clip(cv2.dilate(feat, np.ones((5, 5), np.uint8))*2, 0, 1))
    face = (int(cx - 0.5*fw), int(cy - 0.66*fw), int(fw), int(1.32*fw))
    return rgb, dict(person=person, skin=skinm.astype(np.float32), face=face, lms=lms, poster=poster,
                     features=feat, blotch=blotch*sk, fw=fw, nose_shadow=nose)


def regen_fine(rgb, sigma=1.2, seed=0):
    """Replace the finest detail (high-pass below `sigma`) with new random detail of the same local amplitude:
    what a generative tool does to grain, pores, hair strands and fabric when it re-renders a photo."""
    x = rgb.astype(np.float32)
    base = cv2.GaussianBlur(x, (0, 0), sigma); hp = x - base
    amp = np.sqrt(cv2.GaussianBlur(hp*hp, (0, 0), 4*sigma))
    n = np.random.default_rng(seed).standard_normal(x.shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.6); n = n - cv2.GaussianBlur(n, (0, 0), sigma); n /= n.std() + 1e-6
    return np.clip(base + 0.8*amp*n, 0, 255).astype(np.uint8)


def relight(rgb, cx=0.62, cy=0.32, rx=0.30, ry=0.35, lift=0.35, floor=0.80):
    """Smooth local relight (a key light on the subject, darker surround), pointwise in the frame of `rgb`."""
    h, w = rgb.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = floor + lift*np.exp(-(((xx - cx*w)/(rx*w))**2 + ((yy - cy*h)/(ry*h))**2))
    return (np.clip(rgb.astype(np.float32)*g[..., None], 0, 255) + 0.5).astype(np.uint8)
