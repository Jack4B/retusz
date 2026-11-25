"""Reference copies of the 0.1.5 Dodge & Burn (v1) and Plastyka 3D gain, kept only so that the D&B quality gate
(tools/dnb_gate.py) can draw old-vs-new sheets and numbers. Not used by the app."""
import numpy as np, cv2
from retusz.engine import color as C, relight as RL

def dodge_burn_v1(out, mq_raw, faces_px, strength):
    """0.1.5 pipeline._dodge_burn_region with color.dodge_burn (Gaussian band of L*, a*b* kept)."""
    H, W = out.shape[:2]
    if mq_raw.shape[:2] != (H//4, W//4) and mq_raw.shape[:2] != (H, W):
        mq_raw = cv2.resize(mq_raw, (max(2, W//4), max(2, H//4)), interpolation=cv2.INTER_AREA)
    x0 = max(0, int(min(x - 0.6*w for x, y, w, h in faces_px))); y0 = max(0, int(min(y - 0.4*h for x, y, w, h in faces_px)))
    x1 = min(W, int(max(x + 1.6*w for x, y, w, h in faces_px))); y1 = min(H, int(max(y + 1.9*h for x, y, w, h in faces_px)))
    if x1-x0 < 8 or y1-y0 < 8: return out
    mc = cv2.resize(mq_raw[y0//4:(y1+3)//4, x0//4:(x1+3)//4], (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    crop = out[y0:y1, x0:x1]; lab = C.to_lab(crop)
    new = C.dodge_burn(lab, mc, [(x-x0, y-y0, w, h) for x, y, w, h in faces_px], strength)
    dL = new[..., 0] - lab[..., 0]
    fw = float(np.median([f[2] for f in faces_px])); m = max(4, int(0.15*fw))
    wy = np.clip(np.minimum(np.arange(y1-y0), np.arange(y1-y0)[::-1])/m, 0, 1)
    wx = np.clip(np.minimum(np.arange(x1-x0), np.arange(x1-x0)[::-1])/m, 0, 1)
    if y0 == 0: wy[:m] = 1
    if y1 == H: wy[-m:] = 1
    if x0 == 0: wx[:m] = 1
    if x1 == W: wx[-m:] = 1
    lab[..., 0] += dL*(wy[:, None]*wx[None, :]).astype(np.float32)
    out = out.copy(); out[y0:y1, x0:x1] = C.from_lab(lab)
    return out

def normals_v1(depth, subject=None):
    d = cv2.GaussianBlur(depth, (0, 0), max(1.0, depth.shape[1]/400))
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)/8; gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)/8
    mag = np.hypot(gx, gy); ref = mag[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else mag
    k = np.tan(np.radians(55))/(np.percentile(ref, 90)+1e-6)
    n = np.dstack([-gx*k, -gy*k, np.ones_like(d)]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    curv = -cv2.Laplacian(cv2.GaussianBlur(d, (0, 0), depth.shape[1]/150), cv2.CV_32F)
    c_ref = curv[subject > 0.5] if subject is not None and (subject > 0.5).sum() > 100 else curv
    curv = np.clip(curv/(np.percentile(np.abs(c_ref), 95)+1e-6), -1.5, 1.5)
    return n.astype(np.float32), curv.astype(np.float32)

def subject_mask_v1(depth, skin=None):
    u8 = (depth*255).astype(np.uint8); t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    fg = np.clip((depth - t/255)/0.08 + 0.5, 0, 1)
    if skin is not None:
        fg = np.maximum(fg, cv2.resize(skin, (depth.shape[1], depth.shape[0])))
    k = max(3, depth.shape[1]//80) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return cv2.GaussianBlur(fg, (0, 0), depth.shape[1]/150).astype(np.float32)

def relight_gain_v1(rl, skin, qh, qw, Lq, amount, angle):
    """0.1.5 Relight.gain (background darkening by a soft depth outline, bilinear quarter-res gain)."""
    if amount <= 0: return np.zeros((qh, qw), np.float32)
    fg0 = subject_mask_v1(rl.depth, skin); n0, curv0 = normals_v1(rl.depth, fg0)
    l = RL.light_vec(angle)
    rs = lambda a: cv2.resize(a, (qw, qh), interpolation=cv2.INTER_LINEAR)
    n = rs(n0); fg = rs(fg0); curv = rs(curv0)
    shade = np.clip(n@l - l[2], -0.8, 0.5)
    shade = cv2.GaussianBlur(shade, (0, 0), max(qw, qh)/300)
    Lb = RL.guided_filter(Lq.astype(np.float32), max(2, int(max(qw, qh)/60)), 6.0**2)
    clarity = np.clip(Lq-Lb, -6, 6)/np.maximum(Lq, 25)
    g = fg*(0.38*shade + 0.06*curv + 0.15*clarity) - 0.08*(1-fg)
    g = np.where(g > 0, g*np.clip((97-Lq)/25, 0, 1), g)
    return (amount*g).astype(np.float32)
