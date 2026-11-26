"""Full-resolution pass: trilinear lookup in two RGB->RGB LUTs (background, skin) mixed by a
quarter-resolution mask, times a quarter-resolution per-channel gain map (vignette, brush, 3D light,
skin-tone evening), hash dither."""
import os, sys
import numpy as np
from ..paths import data_dir

os.environ.setdefault("NUMBA_CACHE_DIR", str(data_dir() / "numba-cache"))
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:  # pragma: no cover - fallback when numba is missing
    HAVE_NUMBA = False

def _bilinear_up(q, H, W):
    """Align-corners bilinear upsampling (same mapping as the numba kernel)."""
    qh, qw = q.shape
    ys = np.arange(H)*((qh-1)/max(H-1, 1)); xs = np.arange(W)*((qw-1)/max(W-1, 1))
    y0 = ys.astype(int); x0 = xs.astype(int); y1 = np.minimum(y0+1, qh-1); x1 = np.minimum(x0+1, qw-1)
    wy = (ys-y0)[:, None]; wx = (xs-x0)[None, :]
    return ((q[y0][:, x0]*(1-wx) + q[y0][:, x1]*wx)*(1-wy) + (q[y1][:, x0]*(1-wx) + q[y1][:, x1]*wx)*wy).astype(np.float32)

def _apply_gain_soft_knee(acc, g):
    """Apply spatial gain `g` to `acc` in 0..1: linear for g <= 0 or below the 0.72 highlight knee,
    rational headroom roll-off above k = max(0.72, acc) so bright skin pores never flat-clip to 1.0."""
    v_lin = acc * (1.0 + g)
    k = np.maximum(0.72, acc)
    d = np.maximum(v_lin - k, 0.0)
    h = np.maximum(1.0 - k, 0.0)
    v_knee = k + (d * h) / (d + h + 1e-6)
    return np.where((g > 0.0) & (v_lin > 0.72), v_knee, v_lin)

def render_numpy(src, lut_bg, lut_sk, mask_q, gain_q, seed=12345):
    """Reference implementation (slow, used as fallback and in tests)."""
    from .color import apply_lut
    H, W = src.shape[:2]
    m = _bilinear_up(mask_q, H, W)[..., None]
    if gain_q.ndim == 2: gain_q = np.repeat(gain_q[..., None], 3, 2)
    g = np.dstack([_bilinear_up(gain_q[..., c], H, W) for c in range(3)])
    a = apply_lut(src, lut_bg); b = apply_lut(src, lut_sk)
    acc = a + m*(b-a)
    o = _apply_gain_soft_knee(acc, g)*255.0
    y, x = np.mgrid[0:H, 0:W].astype(np.int64)
    h = (x*73856093) ^ (y*19349663) ^ seed
    d = ((h & 1023)/1023.0 - 0.5)[..., None]
    return np.clip(o + d + 0.5, 0, 255).astype(np.uint8)

if HAVE_NUMBA:
    # frozen builds ship render.py next to the bytecode (see retusz.spec) so the JIT cache works there too
    _cache = (not getattr(sys, "frozen", False)) or os.path.exists(__file__)

    @nb.njit(inline="always")
    def _gain_channel(acc, g):
        v_lin = acc * (1.0 + g)
        if g <= 0.0 or v_lin <= 0.72:
            return v_lin
        k = 0.72 if acc < 0.72 else acc
        d = v_lin - k
        h = 1.0 - k
        if h <= 0.0:
            return acc
        return k + (d * h) / (d + h + 1e-6)

    @nb.njit(parallel=True, fastmath=True, cache=_cache)
    def _render(src, lut_bg, lut_sk, mask_q, gain_q, out, seed):
        H, W, _ = src.shape; N = lut_bg.shape[0]; s = (N-1)/255.0
        qh, qw = mask_q.shape; fy = (qh-1)/max(H-1, 1); fx = (qw-1)/max(W-1, 1)
        for y in nb.prange(H):
            qy = y*fy; y0 = int(qy); y1 = min(y0+1, qh-1); wy = qy-y0
            for x in range(W):
                qx = x*fx; x0 = int(qx); x1 = min(x0+1, qw-1); wx = qx-x0
                m = (mask_q[y0, x0]*(1-wx)+mask_q[y0, x1]*wx)*(1-wy) + (mask_q[y1, x0]*(1-wx)+mask_q[y1, x1]*wx)*wy
                r = src[y, x, 0]*s; gg = src[y, x, 1]*s; b = src[y, x, 2]*s
                ir = min(int(r), N-2); ig = min(int(gg), N-2); ib = min(int(b), N-2)
                fr = r-ir; fg = gg-ig; fb = b-ib
                # Tetrahedral interpolation (Sakamoto 1974 / Kasson 1995): 4 corners, exact neutral axis
                if fr >= fg:
                    if fg >= fb:
                        w0 = 1.0 - fr; wa = fr - fg; wb = fg - fb; w1 = fb
                        ar = 1; ag = 0; ab = 0; br = 1; bg_ = 1; bb = 0
                    elif fr >= fb:
                        w0 = 1.0 - fr; wa = fr - fb; wb = fb - fg; w1 = fg
                        ar = 1; ag = 0; ab = 0; br = 1; bg_ = 0; bb = 1
                    else:
                        w0 = 1.0 - fb; wa = fb - fr; wb = fr - fg; w1 = fg
                        ar = 0; ag = 0; ab = 1; br = 1; bg_ = 0; bb = 1
                else:
                    if fb > fg:
                        w0 = 1.0 - fb; wa = fb - fg; wb = fg - fr; w1 = fr
                        ar = 0; ag = 0; ab = 1; br = 0; bg_ = 1; bb = 1
                    elif fb > fr:
                        w0 = 1.0 - fg; wa = fg - fb; wb = fb - fr; w1 = fr
                        ar = 0; ag = 1; ab = 0; br = 0; bg_ = 1; bb = 1
                    else:
                        w0 = 1.0 - fg; wa = fg - fr; wb = fr - fb; w1 = fb
                        ar = 0; ag = 1; ab = 0; br = 1; bg_ = 1; bb = 0
                v00 = lut_bg[ir, ig, ib, 0]; v01 = lut_bg[ir, ig, ib, 1]; v02 = lut_bg[ir, ig, ib, 2]
                va0 = lut_bg[ir+ar, ig+ag, ib+ab, 0]; va1 = lut_bg[ir+ar, ig+ag, ib+ab, 1]; va2 = lut_bg[ir+ar, ig+ag, ib+ab, 2]
                vb0 = lut_bg[ir+br, ig+bg_, ib+bb, 0]; vb1 = lut_bg[ir+br, ig+bg_, ib+bb, 1]; vb2 = lut_bg[ir+br, ig+bg_, ib+bb, 2]
                v10 = lut_bg[ir+1, ig+1, ib+1, 0]; v11 = lut_bg[ir+1, ig+1, ib+1, 1]; v12 = lut_bg[ir+1, ig+1, ib+1, 2]
                acc0 = w0*v00 + wa*va0 + wb*vb0 + w1*v10
                acc1 = w0*v01 + wa*va1 + wb*vb1 + w1*v11
                acc2 = w0*v02 + wa*va2 + wb*vb2 + w1*v12
                if m > 0.002:
                    s00 = lut_sk[ir, ig, ib, 0]; s01 = lut_sk[ir, ig, ib, 1]; s02 = lut_sk[ir, ig, ib, 2]
                    sa0 = lut_sk[ir+ar, ig+ag, ib+ab, 0]; sa1 = lut_sk[ir+ar, ig+ag, ib+ab, 1]; sa2 = lut_sk[ir+ar, ig+ag, ib+ab, 2]
                    sb0 = lut_sk[ir+br, ig+bg_, ib+bb, 0]; sb1 = lut_sk[ir+br, ig+bg_, ib+bb, 1]; sb2 = lut_sk[ir+br, ig+bg_, ib+bb, 2]
                    s10 = lut_sk[ir+1, ig+1, ib+1, 0]; s11 = lut_sk[ir+1, ig+1, ib+1, 1]; s12 = lut_sk[ir+1, ig+1, ib+1, 2]
                    acc0 += m*(w0*s00 + wa*sa0 + wb*sb0 + w1*s10 - acc0)
                    acc1 += m*(w0*s01 + wa*sa1 + wb*sb1 + w1*s11 - acc1)
                    acc2 += m*(w0*s02 + wa*sa2 + wb*sb2 + w1*s12 - acc2)
                w00 = (1-wx)*(1-wy); w01 = wx*(1-wy); w10 = (1-wx)*wy; w11 = wx*wy
                g0 = gain_q[y0, x0, 0]*w00 + gain_q[y0, x1, 0]*w01 + gain_q[y1, x0, 0]*w10 + gain_q[y1, x1, 0]*w11
                g1 = gain_q[y0, x0, 1]*w00 + gain_q[y0, x1, 1]*w01 + gain_q[y1, x0, 1]*w10 + gain_q[y1, x1, 1]*w11
                g2 = gain_q[y0, x0, 2]*w00 + gain_q[y0, x1, 2]*w01 + gain_q[y1, x0, 2]*w10 + gain_q[y1, x1, 2]*w11
                h = (x*73856093) ^ (y*19349663) ^ seed
                d = ((h & 1023)/1023.0) - 0.5
                o0 = _gain_channel(acc0, g0)*255.0 + d + 0.5
                o1 = _gain_channel(acc1, g1)*255.0 + d + 0.5
                o2 = _gain_channel(acc2, g2)*255.0 + d + 0.5
                out[y, x, 0] = 0 if o0 < 0 else (255 if o0 > 255 else np.uint8(o0))
                out[y, x, 1] = 0 if o1 < 0 else (255 if o1 > 255 else np.uint8(o1))
                out[y, x, 2] = 0 if o2 < 0 else (255 if o2 > 255 else np.uint8(o2))

def render(src, lut_bg, lut_sk, mask_q, gain_q, seed=12345):
    src = np.ascontiguousarray(src, np.uint8)
    lut_bg = np.ascontiguousarray(lut_bg, np.float32); lut_sk = np.ascontiguousarray(lut_sk, np.float32)
    mask_q = np.ascontiguousarray(mask_q, np.float32)
    gain_q = np.asarray(gain_q, np.float32)
    if gain_q.ndim == 2: gain_q = np.repeat(gain_q[..., None], 3, 2)
    gain_q = np.ascontiguousarray(gain_q)
    if not HAVE_NUMBA:
        return render_numpy(src, lut_bg, lut_sk, mask_q, gain_q, seed)
    out = np.empty_like(src); _render(src, lut_bg, lut_sk, mask_q, gain_q, out, seed); return out

def warmup():
    """Compile the kernel once (takes ~1-2 s on first ever run, later loaded from cache)."""
    z = np.zeros((8, 8, 3), np.uint8); l = np.zeros((3, 3, 3, 3), np.float32); q = np.zeros((2, 2), np.float32)
    render(z, l, l, q, q)

