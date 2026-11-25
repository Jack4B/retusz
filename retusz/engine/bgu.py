"""Bilateral Guided Upsampling, fast local variant (Chen, Adams, Wadhwa, Hasinoff, SIGGRAPH Asia 2016).

Implemented from the paper (the authors' reference code google/bgu is Apache-2.0; nothing copied).
`fit` learns, in a bilateral grid over (x, y, luminance), 3x4 affine colour matrices that map a
low-resolution input to its edited version: trilinear splatting of the least-squares sufficient
statistics, Gaussian blur of those statistics in grid space (smoothness, no false edges), a ridge
towards one global affine fit (no noise gain in flat areas, sensible values where there is no data),
and a per-cell solve. `slice_` evaluates the grid on the full-resolution input, so all fine detail
comes from that input and only the smooth, edge-aware transform comes from the edit."""
from __future__ import annotations
from dataclasses import dataclass
import os, sys
import numpy as np
from ..paths import data_dir

os.environ.setdefault("NUMBA_CACHE_DIR", str(data_dir() / "numba-cache"))
try:
    import numba as nb
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

_LW = np.array([0.2126, 0.7152, 0.0722], np.float32)

@dataclass
class Grid:
    A: np.ndarray        # (gh, gw, bins, 3, 4) float32: out_c = A[c, :3] . rgb + A[c, 3]
    cell: float          # work pixels per grid step
    prior: np.ndarray    # (3, 4) global affine fit

    @property
    def bins(self): return self.A.shape[2]

def _z(rgb):
    return np.clip(rgb @ _LW, 0, 1)

# ---------------------------------------------------------------- splat (fit statistics)
def _splat_numpy(g, t, w, cell, bins, gh, gw):
    h, wd = g.shape[:2]
    S = np.zeros((gh, gw, bins, 28), np.float64)
    yy, xx = np.mgrid[0:h, 0:wd]
    gx, gy = xx/cell, yy/cell; gz = _z(g)*(bins-1)
    x0, y0, z0 = np.floor(gx).astype(int), np.floor(gy).astype(int), np.minimum(np.floor(gz).astype(int), bins-2)
    fx, fy, fz = gx-x0, gy-y0, gz-z0
    X = np.concatenate([g, np.ones((h, wd, 1), np.float32)], 2).astype(np.float64)
    feat = np.concatenate([(X[..., :, None]*X[..., None, :]).reshape(h, wd, 16), (X[..., :, None]*t[..., None, :]).reshape(h, wd, 12)], 2)
    feat *= w[..., None]
    for dz in (0, 1):
        wz = fz if dz else 1-fz
        for dy in (0, 1):
            wy = fy if dy else 1-fy
            for dx in (0, 1):
                wx = fx if dx else 1-fx
                ww = (wx*wy*wz)[..., None]
                np.add.at(S, (np.minimum(y0+dy, gh-1), np.minimum(x0+dx, gw-1), z0+dz), feat*ww)
    return S

if HAVE_NUMBA:
    _cache = (not getattr(sys, "frozen", False)) or os.path.exists(__file__)

    @nb.njit(cache=_cache, fastmath=True)
    def _splat_nb(g, t, w, cell, bins, gh, gw):
        h, wd = g.shape[0], g.shape[1]
        S = np.zeros((gh, gw, bins, 28), np.float64)
        f = np.empty(28, np.float64); X = np.empty(4, np.float64)
        for y in range(h):
            gy = y/cell; y0 = int(gy); fy = gy-y0
            for x in range(wd):
                wt = w[y, x]
                if wt <= 0: continue
                gx = x/cell; x0 = int(gx); fx = gx-x0
                X[0] = g[y, x, 0]; X[1] = g[y, x, 1]; X[2] = g[y, x, 2]; X[3] = 1.0
                zz = 0.2126*X[0] + 0.7152*X[1] + 0.0722*X[2]
                zz = min(max(zz, 0.0), 1.0)*(bins-1)
                z0 = min(int(zz), bins-2); fz = zz-z0
                for i in range(4):
                    for j in range(4): f[i*4+j] = X[i]*X[j]*wt
                    for c in range(3): f[16+i*3+c] = X[i]*t[y, x, c]*wt
                for dz in range(2):
                    wz = fz if dz else 1-fz
                    for dy in range(2):
                        wy = fy if dy else 1-fy
                        yy = min(y0+dy, gh-1)
                        for dx in range(2):
                            wx = fx if dx else 1-fx
                            xx = min(x0+dx, gw-1); ww = wx*wy*wz
                            if ww == 0: continue
                            for k in range(28): S[yy, xx, z0+dz, k] += f[k]*ww
        return S

    @nb.njit(parallel=True, cache=_cache, fastmath=True)
    def _slice_nb(img, A, inv_cell, sx, sy, y0f, out):
        h, wd = img.shape[0], img.shape[1]; gh, gw, bins = A.shape[0], A.shape[1], A.shape[2]
        for y in nb.prange(h):
            gy = (sy*(y + y0f + 0.5) - 0.5)*inv_cell
            gy = min(max(gy, 0.0), gh-1.0); yi = min(int(gy), gh-2) if gh > 1 else 0; fy = gy-yi
            for x in range(wd):
                gx = (sx*(x+0.5) - 0.5)*inv_cell
                gx = min(max(gx, 0.0), gw-1.0); xi = min(int(gx), gw-2) if gw > 1 else 0; fx = gx-xi
                r = img[y, x, 0]; g = img[y, x, 1]; b = img[y, x, 2]
                zz = min(max(0.2126*r + 0.7152*g + 0.0722*b, 0.0), 1.0)*(bins-1)
                zi = min(int(zz), bins-2); fz = zz-zi
                o0 = 0.0; o1 = 0.0; o2 = 0.0
                for dz in range(2):
                    wz = fz if dz else 1-fz
                    for dy in range(2):
                        wy = fy if dy else 1-fy
                        yy = min(yi+dy, gh-1)
                        for dx in range(2):
                            ww = (fx if dx else 1-fx)*wy*wz
                            xx = min(xi+dx, gw-1)
                            M = A[yy, xx, zi+dz]
                            o0 += ww*(M[0, 0]*r + M[0, 1]*g + M[0, 2]*b + M[0, 3])
                            o1 += ww*(M[1, 0]*r + M[1, 1]*g + M[1, 2]*b + M[1, 3])
                            o2 += ww*(M[2, 0]*r + M[2, 1]*g + M[2, 2]*b + M[2, 3])
                out[y, x, 0] = o0; out[y, x, 1] = o1; out[y, x, 2] = o2
        return out

def _blur_axis(S, axis, sigma):
    if sigma <= 0: return S
    r = max(1, int(np.ceil(2.5*sigma))); k = np.exp(-0.5*(np.arange(-r, r+1)/sigma)**2); k /= k.sum()
    pad = [(0, 0)]*S.ndim; pad[axis] = (r, r)
    P = np.pad(S, pad)                                      # zero padding: no evidence outside the frame
    n = S.shape[axis]; out = np.zeros_like(S)
    for i, kv in enumerate(k):
        out += kv*np.take(P, np.arange(i, i+n), axis=axis)
    return out

def _solve(XX, XY, lam, prior, mu=1e-7):
    """Ridge per cell: (XX + lam*N*D + mu*I) A = XY + (lam*N*D + mu*I) prior, D = diag(1,1,1,0)."""
    N = XX[..., 3, 3]
    D = np.diag([1.0, 1.0, 1.0, 0.0])
    R = lam*N[..., None, None]*D + mu*np.eye(4)
    P = prior.T                                             # (4, 3)
    return np.linalg.solve(XX + R, XY + R @ P)              # (..., 4, 3)

def fit(guide, target, weight=None, cell=8.0, bins=8, lam=1e-4, sigma_xy=1.0, sigma_z=0.7, use_numba=None) -> Grid:
    """guide/target: float32 HxWx3 in 0..1 at the same (work) resolution; weight HxW in 0..1."""
    g = np.ascontiguousarray(guide, np.float32); t = np.ascontiguousarray(target, np.float32)
    h, wd = g.shape[:2]
    w = np.ones((h, wd), np.float32) if weight is None else np.ascontiguousarray(weight, np.float32)
    gh, gw = int(np.ceil((h-1)/cell))+1, int(np.ceil((wd-1)/cell))+1
    use = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
    S = _splat_nb(g, t, w, float(cell), int(bins), gh, gw) if use else _splat_numpy(g, t, w, cell, bins, gh, gw)
    tot = S.reshape(-1, 28).sum(0)
    XXg, XYg = tot[:16].reshape(4, 4), tot[16:].reshape(4, 3)
    if XXg[3, 3] <= 0:
        prior = np.hstack([np.eye(3), np.zeros((3, 1))])
    else:
        ridge = 1e-6*XXg[3, 3]*np.diag([1.0, 1.0, 1.0, 0.0])
        prior = np.linalg.solve(XXg + ridge + 1e-9*np.eye(4), XYg + ridge @ np.vstack([np.eye(3), np.zeros((1, 3))])).T
    for ax, s in ((0, sigma_xy), (1, sigma_xy), (2, sigma_z)): S = _blur_axis(S, ax, s)
    XX = S[..., :16].reshape(gh, gw, bins, 4, 4); XY = S[..., 16:].reshape(gh, gw, bins, 4, 3)
    A = _solve(XX, XY, lam, prior)
    return Grid(np.ascontiguousarray(np.swapaxes(A, -1, -2), np.float32), float(cell), prior.astype(np.float32))

def _slice_numpy(img, A, inv_cell, sx, sy, y0f):
    h, wd = img.shape[:2]; gh, gw, bins = A.shape[:3]
    ys = (sy*(np.arange(h) + y0f + 0.5) - 0.5)*inv_cell; xs = (sx*(np.arange(wd)+0.5) - 0.5)*inv_cell
    gy = np.clip(ys, 0, gh-1)[:, None]*np.ones((1, wd)); gx = np.clip(xs, 0, gw-1)[None, :]*np.ones((h, 1))
    yi = np.minimum(gy.astype(int), max(gh-2, 0)); xi = np.minimum(gx.astype(int), max(gw-2, 0))
    fy, fx = gy-yi, gx-xi
    zz = _z(img)*(bins-1); zi = np.minimum(zz.astype(int), bins-2); fz = zz-zi
    X = np.concatenate([img, np.ones((h, wd, 1), np.float32)], 2)
    out = np.zeros((h, wd, 3), np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                ww = (fx if dx else 1-fx)*(fy if dy else 1-fy)*(fz if dz else 1-fz)
                M = A[np.minimum(yi+dy, gh-1), np.minimum(xi+dx, gw-1), zi+dz]   # h, w, 3, 4
                out += ww[..., None]*np.einsum("hwck,hwk->hwc", M, X)
    return out.astype(np.float32)

def slice_(img, grid: Grid, sx, sy, y0=0, use_numba=None):
    """Evaluate the grid on `img` (full-res float32 0..1, or a strip starting at full-res row y0).
    sx, sy: work pixels per full-res pixel."""
    img = np.ascontiguousarray(img, np.float32)
    use = HAVE_NUMBA if use_numba is None else (use_numba and HAVE_NUMBA)
    if use:
        out = np.empty(img.shape, np.float32)
        return _slice_nb(img, grid.A, 1.0/grid.cell, float(sx), float(sy), float(y0), out)
    return _slice_numpy(img, grid.A, 1.0/grid.cell, float(sx), float(sy), float(y0))
