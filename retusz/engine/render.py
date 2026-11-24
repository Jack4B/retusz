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

def render_numpy(src, lut_bg, lut_sk, mask_q, gain_q, seed=12345):
    """Reference implementation (slow, used as fallback and in tests)."""
    from .color import apply_lut
    H, W = src.shape[:2]
    m = _bilinear_up(mask_q, H, W)[..., None]
    if gain_q.ndim == 2: gain_q = np.repeat(gain_q[..., None], 3, 2)
    g = np.dstack([_bilinear_up(gain_q[..., c], H, W) for c in range(3)])
    a = apply_lut(src, lut_bg); b = apply_lut(src, lut_sk)
    o = (a + m*(b-a))*255*(1+g)
    y, x = np.mgrid[0:H, 0:W].astype(np.int64)
    h = (x*73856093) ^ (y*19349663) ^ seed
    d = ((h & 1023)/1023.0 - 0.5)[..., None]
    return np.clip(o + d + 0.5, 0, 255).astype(np.uint8)

if HAVE_NUMBA:
    # frozen builds ship render.py next to the bytecode (see retusz.spec) so the JIT cache works there too
    _cache = (not getattr(sys, "frozen", False)) or os.path.exists(__file__)

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
                acc0 = 0.0; acc1 = 0.0; acc2 = 0.0
                for dr in range(2):
                    wr = fr if dr else 1-fr
                    for dg in range(2):
                        wg = fg if dg else 1-fg
                        for db in range(2):
                            w = wr*wg*(fb if db else 1-fb)
                            a0 = lut_bg[ir+dr, ig+dg, ib+db, 0]; a1 = lut_bg[ir+dr, ig+dg, ib+db, 1]; a2 = lut_bg[ir+dr, ig+dg, ib+db, 2]
                            if m > 0.002:
                                a0 += m*(lut_sk[ir+dr, ig+dg, ib+db, 0]-a0); a1 += m*(lut_sk[ir+dr, ig+dg, ib+db, 1]-a1); a2 += m*(lut_sk[ir+dr, ig+dg, ib+db, 2]-a2)
                            acc0 += w*a0; acc1 += w*a1; acc2 += w*a2
                w00 = (1-wx)*(1-wy); w01 = wx*(1-wy); w10 = (1-wx)*wy; w11 = wx*wy
                g0 = gain_q[y0, x0, 0]*w00 + gain_q[y0, x1, 0]*w01 + gain_q[y1, x0, 0]*w10 + gain_q[y1, x1, 0]*w11
                g1 = gain_q[y0, x0, 1]*w00 + gain_q[y0, x1, 1]*w01 + gain_q[y1, x0, 1]*w10 + gain_q[y1, x1, 1]*w11
                g2 = gain_q[y0, x0, 2]*w00 + gain_q[y0, x1, 2]*w01 + gain_q[y1, x0, 2]*w10 + gain_q[y1, x1, 2]*w11
                h = (x*73856093) ^ (y*19349663) ^ seed
                d = ((h & 1023)/1023.0) - 0.5
                o0 = acc0*255.0*(1.0+g0) + d + 0.5; o1 = acc1*255.0*(1.0+g1) + d + 0.5; o2 = acc2*255.0*(1.0+g2) + d + 0.5
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
