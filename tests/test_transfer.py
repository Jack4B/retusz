import numpy as np, cv2, pytest
from synthetic import textured, grade, make_ai
from retusz.engine import transfer as TR

def _hf(img):
    g = img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())

def _lab(x):
    return cv2.cvtColor(np.ascontiguousarray(x, np.float32), cv2.COLOR_RGB2Lab)

@pytest.fixture(scope="module")
def case():
    o, disc = textured(1200, 1800, seed=11); gt = grade(o, disc)
    ai, T = make_ai(gt, 3, 1.2, (4.4, -2.1), 0.05)
    return o, gt, ai, T, TR.prepare(o, ai)

def test_delta_e2000_reference_pair():
    # Sharma, Wu, Dalal (2005) test data, pair 1
    a = np.array([[50.0, 2.6772, -79.7751]]); b = np.array([[50.0, 0.0, -82.7485]])
    assert abs(TR.delta_e2000(a, b)[0] - 2.0425) < 1e-3

def test_result_matches_ground_truth_and_keeps_sharpness(case):
    o, gt, ai, T, prep = case
    out = TR.render(prep, o, TR.TransferParams())
    assert out.shape == o.shape and out.dtype == np.uint8
    inside = prep.coverage_full(o.shape) > 0.99
    de = TR.delta_e2000(_lab(out/255.)[inside], _lab(gt/255.)[inside])
    assert de.mean() < 2.0, de.mean()
    assert _hf(out)/_hf(gt) >= 0.95

def test_zero_strengths_return_original(case):
    o, gt, ai, T, prep = case
    out = TR.render(prep, o, TR.TransferParams(kolor=0, jasnosc=0))
    assert np.abs(out.astype(int) - o).max() <= 1

def test_kolor_only_keeps_original_lightness(case):
    o, gt, ai, T, prep = case
    out = TR.render(prep, o, TR.TransferParams(kolor=1, jasnosc=0))
    full = TR.render(prep, o, TR.TransferParams())
    Lo, Lc, Lf = _lab(o/255.), _lab(out/255.), _lab(full/255.)
    assert np.abs(Lc[..., 0] - Lo[..., 0]).mean() < 0.6
    assert np.abs(Lc[..., 1:] - Lf[..., 1:]).mean() < 1.5

def test_mask_zero_disables_luminosity(case):
    o, gt, ai, T, prep = case
    m = np.zeros((300, 450), np.float32)
    a = TR.render(prep, o, TR.TransferParams(), mask=m)
    b = TR.render(prep, o, TR.TransferParams(jasnosc=0))
    assert np.abs(a.astype(int) - b).max() <= 1

def test_identity_ai_is_downscaled_original():
    o, _ = textured(900, 1350, seed=12)
    ai = cv2.resize(o, (450, 300), interpolation=cv2.INTER_AREA)
    prep = TR.prepare(o, ai)
    out = TR.render(prep, o, TR.TransferParams())
    assert TR.delta_e2000(_lab(out/255.).reshape(-1, 3), _lab(o/255.).reshape(-1, 3)).mean() < 0.6
    assert prep.changed_frac < 0.01

def test_removed_object_does_not_ghost():
    o, disc = textured(1200, 1800, seed=13); gt = grade(o, None)
    ai, T = make_ai(gt, 3, 0.0, (0, 0), 0.0, jpeg=95)
    y0, y1, x0, x1 = 120, 190, 200, 300                     # AI "removes" a patch: flat fill
    ai[y0:y1, x0:x1] = ai[y0-20:y0-10, x0:x1].reshape(-1, 3).mean(0).astype(np.uint8)
    prep = TR.prepare(o, ai)
    assert prep.changed_frac > 0.005
    out = TR.render(prep, o, TR.TransferParams())
    Y0, Y1, X0, X1 = y0*3+15, y1*3-15, x0*3+15, x1*3-15
    de = TR.delta_e2000(_lab(out[Y0:Y1, X0:X1]/255.).reshape(-1, 3), _lab(gt[Y0:Y1, X0:X1]/255.).reshape(-1, 3))
    assert de.mean() < 4.0, de.mean()
