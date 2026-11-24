"""Non-destructive straighten + crop, applied after the colour render.

angle: degrees, positive = counter-clockwise (OpenCV convention). The straightened 'frame' has the
photo's own size: the rotated photo is scaled just enough to cover it, so there are never empty
corners. crop: [x0, y0, x1, y1] normalised in the frame, or None (whole frame). Everything is
resolution independent, so the 1600 px preview and the full-size export use the same numbers.
Normalised coordinate u maps to pixel u*W - 0.5 (pixel centres), the same as the canvas."""
from __future__ import annotations
import math
import numpy as np, cv2

MIN_CROP = 0.02
ASPECTS = {"free": "Dowolny", "orig": "Oryginał", "1:1": "1:1", "2:3": "2:3", "4:5": "4:5", "16:9": "16:9"}
_RATIO = {"1:1": 1.0, "2:3": 2/3, "4:5": 4/5, "16:9": 16/9}

def straighten_scale(w, h, angle):
    a = math.radians(abs(angle)); c, s = math.cos(a), math.sin(a)
    return max((w*c + h*s)/w, (w*s + h*c)/h)

def matrix(angle, w, h):
    """2x3 affine: source pixel -> frame pixel."""
    return cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), float(angle), straighten_scale(w, h, angle))

def _px(p, w, h): return np.array([p[0]*w - 0.5, p[1]*h - 0.5, 1.0])

def src_to_frame(p, angle, w, h):
    if not angle: return (float(p[0]), float(p[1]))
    q = matrix(angle, w, h) @ _px(p, w, h)
    return ((q[0]+0.5)/w, (q[1]+0.5)/h)

def frame_to_src(p, angle, w, h):
    if not angle: return (float(p[0]), float(p[1]))
    q = cv2.invertAffineTransform(matrix(angle, w, h)) @ _px(p, w, h)
    return ((q[0]+0.5)/w, (q[1]+0.5)/h)

def crop_px(crop, w, h):
    if crop is None: return 0, 0, w, h
    x0, y0 = int(round(crop[0]*w)), int(round(crop[1]*h))
    x1, y1 = max(x0+1, int(round(crop[2]*w))), max(y0+1, int(round(crop[3]*h)))
    return x0, y0, min(w, x1), min(h, y1)

def apply(img, angle, crop, interp=cv2.INTER_CUBIC):
    """Straighten then crop (one resampling). img: HxW or HxWxC, any dtype OpenCV warps."""
    h, w = img.shape[:2]; x0, y0, x1, y1 = crop_px(crop, w, h)
    if not angle:
        return img if crop is None else img[y0:y1, x0:x1]
    M = matrix(angle, w, h); M[:, 2] -= (x0, y0)
    return cv2.warpAffine(img, M, (x1-x0, y1-y0), flags=interp, borderMode=cv2.BORDER_REPLICATE)

def horizon_angle(a, b):
    """Angle change (deg) that makes the line a->b (frame pixels) level; steep lines are taken as verticals."""
    t = math.degrees(math.atan2(b[1]-a[1], b[0]-a[0]))
    if t > 90: t -= 180
    if t <= -90: t += 180
    if abs(t) > 45: t -= math.copysign(90, t)
    return t

def ratio(key, w, h, portrait):
    if key == "orig": r = w/h
    else:
        r = _RATIO.get(key)
        if r is None: return None
    if r != 1 and key != "orig": r = min(r, 1/r) if portrait else max(r, 1/r)
    elif key == "orig" and portrait != (h > w): r = 1/r
    return r

def aspect_rect(key, w, h, portrait):
    """Largest centred crop with the preset's ratio (w:h in pixels); None for 'free'."""
    r = ratio(key, w, h, portrait)
    if r is None: return None
    if w/h > r: cw, ch = h*r, h
    else: cw, ch = w, w/r
    fx, fy = cw/w, ch/h
    return [0.5-fx/2, 0.5-fy/2, 0.5+fx/2, 0.5+fy/2]

def fit_aspect(rect, r, w, h, anchor):
    """Adjust a dragged rect to ratio r (pixels) keeping the anchor corner fixed and staying inside."""
    ax, ay = anchor
    ox = rect[2] if abs(rect[0]-ax) < abs(rect[2]-ax) else rect[0]
    oy = rect[3] if abs(rect[1]-ay) < abs(rect[3]-ay) else rect[1]
    sx, sy = (1 if ox >= ax else -1), (1 if oy >= ay else -1)
    pw, ph = max(abs(ox-ax)*w, 1e-6), max(abs(oy-ay)*h, 1e-6)
    if pw/ph > r: ph = pw/r
    else: pw = ph*r
    mw = (1-ax)*w if sx > 0 else ax*w; mh = (1-ay)*h if sy > 0 else ay*h
    k = min(1.0, mw/pw, mh/ph); pw *= k; ph *= k
    bx, by = ax + sx*pw/w, ay + sy*ph/h
    return [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]

def clamp_crop(c):
    if c is None: return None
    x0, y0, x1, y1 = (float(np.clip(v, 0, 1)) for v in c)
    x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1)
    def grow(a, b):
        if b-a >= MIN_CROP: return a, b
        m = float(np.clip((a+b)/2, MIN_CROP/2, 1-MIN_CROP/2)); return m-MIN_CROP/2, m+MIN_CROP/2
    x0, x1 = grow(x0, x1); y0, y1 = grow(y0, y1)
    if x0 < 1e-4 and y0 < 1e-4 and x1 > 1-1e-4 and y1 > 1-1e-4: return None
    return [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]
