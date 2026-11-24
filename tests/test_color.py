import numpy as np
from retusz.engine import color as C
from conftest import synth

def test_identity_lut_roundtrip():
    G = C.rgb_grid(17); img = synth()
    out = C.apply_lut(img, G)*255
    assert np.abs(out-img).max() < 0.6
    assert np.abs(C.apply_lut_fast(img, G)*255-img).max() < 0.6

def test_smooth_lut_keeps_identity():
    G = C.rgb_grid(33); S = C.smooth_lut(G)
    assert np.abs(S[1:-1, 1:-1, 1:-1]-G[1:-1, 1:-1, 1:-1]).max() < 1e-5

def test_lab_roundtrip():
    img = synth(); assert np.abs(C.from_lab(C.to_lab(img)).astype(int)-img).max() <= 1

def test_mkl_matches_mean_and_cov(rng):
    S = rng.normal([50, 0, 0], [10, 5, 3], (5000, 3)); R = rng.normal([60, 10, -5], [8, 3, 6], (5000, 3))
    ms, mr, T = C.mkl_fit(S, R); X = (S-ms)@T + mr
    assert np.allclose(X.mean(0), R.mean(0), atol=0.5)
    assert np.allclose(np.cov(X.T), np.cov(R.T), atol=2.0)

def test_idt_brings_distribution_closer():
    s, r = synth(seed=1), synth(seed=2, tint=(1.0, 0.7, 0.5))
    lut = C.fit_idt_lut(s, r, N=17)
    out = C.from_lab(C.apply_lut(s, lut))
    assert C.swd(out, r) < 0.5*C.swd(s, r)

def test_crop_border_removes_white_frame():
    img = np.full((200, 300, 3), 250, np.uint8); img[20:180, 30:270] = synth(160, 240)
    c = C.crop_border(img); assert c.shape[0] < 165 and c.shape[1] < 245
    assert C.crop_border(synth()).shape == synth().shape  # no frame -> unchanged

def test_rolloff_monotone_and_bounded():
    L = np.linspace(0, 120, 500); R = C.rolloff(L)
    assert np.all(np.diff(R) >= -1e-9) and R.max() < 100 and np.allclose(R[L < 88], L[L < 88])

def test_dodge_burn_reduces_mid_band_but_keeps_mean():
    rng = np.random.default_rng(3); L = 60 + cv_blobs(rng)
    lab = np.dstack([L, np.zeros_like(L), np.zeros_like(L)]).astype(np.float32)
    mask = np.ones(L.shape, np.float32); faces = [(50, 50, 120, 140)]
    out = C.dodge_burn(lab, mask, faces, 0.8)
    assert out[..., 0].std() < lab[..., 0].std()
    assert abs(out[..., 0].mean()-lab[..., 0].mean()) < 0.5

def cv_blobs(rng, h=240, w=240):
    import cv2
    return cv2.GaussianBlur(rng.normal(0, 30, (h, w)).astype(np.float32), (0, 0), 6)
