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

def test_preview_render_on_smaller_image_matches_full(case):
    o, gt, ai, T, prep = case
    full = TR.render(prep, o, TR.TransferParams(kolor=0.8, jasnosc=0.6))
    small = cv2.resize(o, (600, 400), interpolation=cv2.INTER_AREA)
    prev = TR.render(prep, small, TR.TransferParams(kolor=0.8, jasnosc=0.6))
    ref = cv2.resize(full, (600, 400), interpolation=cv2.INTER_AREA)
    assert prev.shape == (400, 600, 3)
    assert np.abs(prev.astype(int) - ref).mean() < 1.5

def test_ai_frame_border_is_not_flagged_as_changed():
    o, _ = textured(400, 600, seed=14); w = o.astype(np.float32)/255
    valid = np.zeros((400, 600), np.float32); valid[30:-30, 40:-40] = 1      # AI covers the centre only
    a = w.copy()                                                               # outside: replicated edge, as remap does
    a[:30] = a[30]; a[-30:] = a[-31]; a[:, :40] = a[:, 40:41]; a[:, -40:] = a[:, -41:-40]
    ch = TR.changed_map(w, a, valid)
    assert float((ch*valid).sum()/valid.sum()) < 0.005


# ---------------------------------------------------------------- generative AI (Gemini / Nano Banana style)
@pytest.fixture(scope="module")
def generative():
    """Grainy portrait original; the AI copy is re-rendered: other framing, non-rigid drift, a moved 'person',
    local relight + strong regrade, all fine detail redrawn and smoothed, and a redrawn poster (the only
    real content change, ~7 % of the frame)."""
    from synthetic import portrait_scene, make_generative, strong_grade, relight
    o, _ = portrait_scene(1800, 1200, seed=1, grain=8)
    lit = relight(o)
    ai, ch = make_generative(lit, (700, 1050), shift=(6, -5), zoom=1.04, drift_amp=0.004,
                             island=(0.58, 0.40, 0.14, 0.005, -0.003), regen=((0.08, 0.10, 0.33, 0.38),), fine=True)
    prep = TR.prepare(o, ai)
    ch_work = cv2.remap(ch, *prep.al.maps(), cv2.INTER_LINEAR)
    return o, strong_grade(lit), ch_work, prep

def test_generative_change_map_flags_only_the_redrawn_region(generative):
    """0.1.4 flagged ~78 % of such a frame (fine-scale NCC reads smoothed/redrawn grain as changed content),
    so almost nothing was transferred."""
    o, gt, ch_work, prep = generative
    core = cv2.erode(ch_work, np.ones((9, 9), np.uint8)) > 0.5
    assert prep.changed_frac < 0.12, prep.changed_frac
    assert prep.changed[core].mean() > 0.8, prep.changed[core].mean()

def test_generative_light_and_colour_are_transferred(generative):
    o, gt, ch_work, prep = generative
    out = TR.render(prep, o, TR.TransferParams())
    H, W = o.shape[:2]
    away = cv2.dilate((cv2.resize(ch_work, (W, H)) > 0.01).astype(np.uint8), np.ones((31, 31), np.uint8)) == 0
    m = (prep.coverage_full(o.shape) > 0.99) & away
    de = TR.delta_e2000(_lab(out/255.)[m], _lab(gt/255.)[m]).mean()
    de0 = TR.delta_e2000(_lab(o/255.)[m], _lab(gt/255.)[m]).mean()
    assert de < 2.0 and de < 0.2*de0, (de, de0)

def test_smoothed_ai_with_redrawn_fine_detail_is_not_flagged():
    """Same picture, but the AI copy has no grain and its finest detail is redrawn: nothing changed."""
    from synthetic import portrait_scene, regen_fine
    o, _ = portrait_scene(900, 600, seed=2, grain=10)
    ai = cv2.GaussianBlur(regen_fine(cv2.resize(o, (360, 540), interpolation=cv2.INTER_AREA), seed=5), (0, 0), 0.8)
    assert TR.prepare(o, ai).changed_frac < 0.02


def test_preview_cache_matches_render_for_all_controls(case):
    o, gt, ai, T, prep = case
    prev = cv2.resize(o, (600, 400), interpolation=cv2.INTER_AREA)
    cache = TR.make_preview_cache(prep, prev)
    yy, xx = np.mgrid[0:200, 0:300].astype(np.float32)
    mask = np.clip(xx / 300.0, 0.0, 1.0)
    for prm, m in (
        (TR.TransferParams(1.0, 1.0, True), None),
        (TR.TransferParams(0.65, 0.35, True), None),
        (TR.TransferParams(0.8, 0.9, False), None),
        (TR.TransferParams(0.0, 0.0, True), None),
        (TR.TransferParams(0.9, 0.8, True), mask),
    ):
        fast = TR.mix_cached(cache, prm, mask=m)
        ref = TR.render(prep, prev, prm, mask=m, out_dtype=np.uint8)
        assert fast.shape == ref.shape and fast.dtype == np.uint8
        assert int(np.abs(fast.astype(int) - ref.astype(int)).max()) <= 1


def test_prepare_reports_progress_stages():
    o, _ = textured(600, 900, seed=15)
    ai = cv2.resize(o, (300, 200), interpolation=cv2.INTER_AREA)
    stages = []
    TR.prepare(o, ai, progress=lambda s, info=None: stages.append(s))
    assert stages == ["rejestracja", "dopasowanie", "mapa_zmian", "siatka", "gotowe"]

