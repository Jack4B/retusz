"""Colour maths from published methods: LAB helpers, 3D LUTs, MKL (Pitié & Kokaram 2007),
IDT sliced optimal transport (Pitié 2005/2007), tonal zones, vignette profile, highlight roll-off."""
import numpy as np, cv2

def small(rgb, side=1200):
    h, w = rgb.shape[:2]; s = side / max(h, w)
    if s >= 1: return rgb
    return cv2.resize(rgb, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)

def crop_border(rgb, thr=235, margin=4):
    """Remove a flat light frame (e.g. a white passe-partout around an exported reference)."""
    g = rgb.mean(2); rows = np.where(g.mean(1) < thr)[0]; cols = np.where(g.mean(0) < thr)[0]
    if len(rows) < 10 or len(cols) < 10: return rgb
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    if (r1-r0) > 0.97*rgb.shape[0] and (c1-c0) > 0.97*rgb.shape[1]: return rgb
    return rgb[r0+margin:r1-margin, c0+margin:c1-margin]

def to_lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32)/255, cv2.COLOR_RGB2LAB)

def from_lab(lab):
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)*255+0.5, 0, 255).astype(np.uint8)

def rgb_grid(N=33):
    g = np.linspace(0, 1, N, dtype=np.float32)
    return np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)

def grid_lab(N=33):
    G = rgb_grid(N); return G, cv2.cvtColor(G.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(N, N, N, 3)

def apply_lut(rgb, lut):
    """Trilinear lookup of uint8 RGB (any shape [...,3]) in an N^3 LUT; returns float32 LUT values."""
    N = lut.shape[0]; x = rgb.astype(np.float32)/255*(N-1)
    i0 = np.clip(np.floor(x).astype(np.int32), 0, N-2); f = x-i0
    out = np.zeros(rgb.shape[:-1] + (lut.shape[-1],), np.float32)
    for dr in (0, 1):
        wr = f[..., 0] if dr else 1-f[..., 0]
        for dg in (0, 1):
            wg = f[..., 1] if dg else 1-f[..., 1]
            for db in (0, 1):
                wb = f[..., 2] if db else 1-f[..., 2]
                out += (wr*wg*wb)[..., None]*lut[i0[..., 0]+dr, i0[..., 1]+dg, i0[..., 2]+db]
    return out

def apply_lut_fast(rgb, lut):
    """Same as apply_lut, evaluated only on distinct colours (fast on previews)."""
    packed = (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]
    u, inv = np.unique(packed.ravel(), return_inverse=True)
    cols = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], -1).astype(np.uint8)
    return apply_lut(cols, lut)[inv.ravel()].reshape(rgb.shape[:-1] + (lut.shape[-1],))

def smooth_lut(lut):
    """Separable 1-2-1 smoothing on every grid axis (removes CDF steps, keeps monotonicity)."""
    out = lut; k = (0.25, 0.5, 0.25)
    for ax in range(3):
        pad = [(0, 0)]*4; pad[ax] = (1, 1); p = np.pad(out, pad, mode="edge"); n = out.shape[ax]
        sl = lambda a: tuple(slice(a, a+n) if i == ax else slice(None) for i in range(4))
        out = k[0]*p[sl(0)] + k[1]*p[sl(1)] + k[2]*p[sl(2)]
    return out.astype(np.float32)

def mkl_fit(S, R):
    """Monge-Kantorovich linear map between two LAB point clouds. Returns (mu_s, mu_r, T)."""
    S = S.astype(np.float64); R = R.astype(np.float64)
    A, B = np.cov(S.T)+np.eye(3)*1e-3, np.cov(R.T)+np.eye(3)*1e-3
    Da2, Ua = np.linalg.eigh(A); Da = np.diag(np.sqrt(np.clip(Da2, 1e-9, None)))
    C = Da@Ua.T@B@Ua@Da; Dc2, Uc = np.linalg.eigh(C); Dc = np.diag(np.sqrt(np.clip(Dc2, 1e-9, None)))
    Di = np.diag(1/np.diag(Da)); T = Ua@Di@Uc@Dc@Uc.T@Di@Ua.T
    return S.mean(0), R.mean(0), T.astype(np.float32)

def _cdf_map(src, ref, n=256):
    q = np.linspace(0, 1, n); xs = np.quantile(src, q); xr = np.quantile(ref, q)
    xs, idx = np.unique(xs, return_index=True); return xs, xr[idx]

def fit_idt_lut(s, r, iters=12, N=33, seed=0):
    """Iterative distribution transfer (sliced 1-D OT in random rotations), applied jointly to
    sample points and the LUT grid. Returns an RGB->LAB LUT (N^3x3)."""
    rng = np.random.default_rng(seed)
    S = to_lab(s).reshape(-1, 3).astype(np.float64); R = to_lab(r).reshape(-1, 3).astype(np.float64)
    S = S[rng.choice(len(S), min(len(S), 150000), replace=False)]
    R = R[rng.choice(len(R), min(len(R), 150000), replace=False)]
    _, Glab = grid_lab(N); X = S.copy(); Gx = Glab.reshape(-1, 3).astype(np.float64)
    for _ in range(iters):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        Xp, Rp, Gp = X@Q, R@Q, Gx@Q
        for c in range(3):
            xs, xr = _cdf_map(Xp[:, c], Rp[:, c])
            Xp[:, c] = np.interp(Xp[:, c], xs, xr); Gp[:, c] = np.interp(Gp[:, c], xs, xr)
        X, Gx = Xp@Q.T, Gp@Q.T
    return Gx.reshape(N, N, N, 3).astype(np.float32)

def zone_w(L):
    sh = np.clip((50-L)/30, 0, 1); hi = np.clip((L-55)/30, 0, 1); mid = np.clip(1-sh-hi, 0, 1)
    return [sh, mid, hi]

def radial(h, w):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.sqrt(((x-w/2)/(w/2))**2+((y-h/2)/(h/2))**2)/np.sqrt(2)

def vignette_profile(L, mask_bg, bins=12):
    r = radial(*L.shape); prof = []
    for i in range(bins):
        m = (r >= i/bins) & (r < (i+1)/bins) & (mask_bg > 0.5)
        prof.append(np.median(L[m]) if m.sum() > 50 else np.nan)
    prof = np.array(prof); idx = np.arange(bins); ok = ~np.isnan(prof)
    if ok.sum() == 0: return np.zeros(bins, np.float32)
    prof = np.interp(idx, idx[ok], prof[ok])
    return (prof - prof[:3].mean()).astype(np.float32)

def rolloff(L, knee=88.0):
    over = np.clip(L-knee, 0, None); span = 100-knee
    return np.where(L > knee, knee + span*(1-np.exp(-over/span)), L)

DB_FACE_PX = 240   # D&B is computed at a scale where the face is ~240 px wide: same result for preview and export

def dodge_burn(lab, mask, fcs, strength=0.4):
    """Auto D&B: flatten the mid-frequency band of L inside skin (between fw/60 and fw/8),
    so blotches and uneven light patches even out while pores and face shape stay.
    The band is measured at a face-normalised scale (fast and resolution independent)."""
    if not len(fcs) or strength <= 0: return lab
    fw = float(np.median([f[2] for f in fcs]))
    L = lab[..., 0]; H, W = L.shape
    k = max(1.0, fw/DB_FACE_PX)
    if k > 1.05:
        sw, sh = max(8, int(round(W/k))), max(8, int(round(H/k)))
        Ls = cv2.resize(L, (sw, sh), interpolation=cv2.INTER_AREA); ms = cv2.resize(mask, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        Ls, ms, k = L, mask, 1.0
    f = fw/k; mb = (ms > 0.3).astype(np.float32)
    nb = lambda X, sg: cv2.GaussianBlur(X*mb, (0, 0), sg)/(cv2.GaussianBlur(mb, (0, 0), sg)+1e-4)
    mid = nb(Ls, max(f/60, 0.5)) - nb(Ls, max(f/8, 1.0))
    if k > 1.0: mid = cv2.resize(mid, (W, H), interpolation=cv2.INTER_CUBIC)
    out = lab.copy(); out[..., 0] = L - strength*0.8*mid*mask
    return out

# ---- metrics used by tests and benchmarks ----
def de_mean(a, b):
    A = to_lab(small(a, 400)).reshape(-1, 3).mean(0); B = to_lab(small(b, 400)).reshape(-1, 3).mean(0)
    return float(np.linalg.norm(A-B))

def swd(a, b, n=64, seed=0):
    rng = np.random.default_rng(seed)
    A = to_lab(small(a, 400)).reshape(-1, 3); B = to_lab(small(b, 400)).reshape(-1, 3)
    q = np.linspace(0.005, 0.995, 200); d = []
    for _ in range(n):
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        d.append(np.abs(np.quantile(A@v, q)-np.quantile(B@v, q)).mean())
    return float(np.mean(d))
