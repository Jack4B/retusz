import numpy as np
from retusz.engine import render as R, color as C
from conftest import synth

def test_identity_render_is_lossless():
    img = synth(); G = C.rgb_grid(33); q = np.zeros((60, 90), np.float32)
    out = R.render(img, G, G, q, q)
    assert np.abs(out.astype(int)-img).max() <= 1

def test_numba_matches_numpy_reference():
    img = synth(123, 201); G = C.rgb_grid(17)
    lut_b = np.clip(G**1.2, 0, 1).astype(np.float32); lut_s = np.clip(G*0.9+0.05, 0, 1).astype(np.float32)
    rng = np.random.default_rng(0)
    m = rng.uniform(0, 1, (30, 50)).astype(np.float32); g = rng.uniform(-0.2, 0.2, (30, 50)).astype(np.float32)
    a = R.render(img, lut_b, lut_s, m, g); b = R.render_numpy(img, lut_b, lut_s, m, g)
    assert np.abs(a.astype(int)-b).max() <= 2 and np.abs(a.astype(int)-b).mean() < 0.3

def test_gain_brightens():
    img = np.full((40, 40, 3), 100, np.uint8); G = C.rgb_grid(9)
    out = R.render(img, G, G, np.zeros((10, 10), np.float32), np.full((10, 10), 0.2, np.float32))
    assert abs(out.mean()-120) < 1.5
