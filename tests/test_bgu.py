import numpy as np, cv2, pytest
from synthetic import textured, grade
from retusz.engine import bgu

def _small(img, k):
    h, w = img.shape[:2]
    return cv2.resize(img, (w//k, h//k), interpolation=cv2.INTER_AREA)

def _hf(img):
    g = img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

def test_global_affine_colour_change_reproduced_exactly():
    o, _ = textured(600, 900, seed=2); f = o.astype(np.float32)/255
    M = np.array([[1.1, 0.05, -0.05], [0.0, 0.9, 0.05], [-0.05, 0.1, 0.95]], np.float32); b = np.array([0.02, -0.01, 0.03], np.float32)
    gt = f @ M.T + b
    k = 3; g = bgu.fit(_small(f, k), _small(gt, k))
    out = bgu.slice_(f, g, (900//k)/900, (600//k)/600)
    assert out.shape == f.shape
    assert np.abs(out - gt).max() < 2/255

def test_slice_keeps_full_resolution_detail():
    o, disc = textured(900, 1350, seed=3); gt = grade(o, None, relight=True)
    k = 3; f = o.astype(np.float32)/255; t = gt.astype(np.float32)/255
    g = bgu.fit(_small(f, k), _small(t, k))
    out = bgu.slice_(f, g, (1350//k)/1350, (900//k)/900)
    naive = cv2.resize(_small(t, k), (1350, 900), interpolation=cv2.INTER_CUBIC)
    assert _hf(out)/_hf(t) > 0.97
    assert _hf(naive)/_hf(t) < 0.6
    assert np.abs(out - t).mean() < 2.5/255

def test_numba_matches_numpy():
    o, _ = textured(240, 360, seed=4); f = o.astype(np.float32)/255; t = grade(o, None).astype(np.float32)/255
    g1 = bgu.fit(_small(f, 2), _small(t, 2), use_numba=True); g2 = bgu.fit(_small(f, 2), _small(t, 2), use_numba=False)
    assert np.allclose(g1.A, g2.A, atol=1e-4)
    a = bgu.slice_(f, g1, 0.5, 0.5, use_numba=True); b = bgu.slice_(f, g1, 0.5, 0.5, use_numba=False)
    assert np.abs(a - b).max() < 1e-4

def test_flat_region_no_noise_gain():
    r = np.random.default_rng(5)
    f = np.full((300, 450, 3), 0.5, np.float32) + r.normal(0, 0.004, (300, 450, 3)).astype(np.float32)
    f[:, :150] = np.linspace(0, 1, 150, dtype=np.float32)[None, :, None]    # gradient gives the global fit signal
    t = np.clip(f*1.2 - 0.05, 0, 1)
    g = bgu.fit(_small(f, 3), _small(t, 3))
    out = bgu.slice_(f, g, 1/3, 1/3)
    assert out[:, 200:].std() < 1.5*t[:, 200:].std()

def test_zero_weight_falls_back_to_prior():
    o, _ = textured(300, 450, seed=6); f = o.astype(np.float32)/255
    t = f*0.8 + 0.1
    w = np.ones(f.shape[:2], np.float32); w[:, 75:] = 0                  # only the left third is known
    t_bad = t.copy(); t_bad[:, 75:] = 0                                   # garbage where weight is 0
    g = bgu.fit(_small(f, 3), _small(t_bad, 3), _small(w, 3))
    out = bgu.slice_(f, g, 1/3, 1/3)
    assert np.abs(out[:, 120:] - t[:, 120:]).mean() < 3/255
