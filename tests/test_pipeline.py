import time
import numpy as np, pytest
from retusz.engine.pipeline import Engine, Params, brush_map
from retusz.engine import color as C, relight as RL
from retusz import models
from conftest import synth, REAL_SRC, REAL_REF

def test_params_from_dict_clamps_and_ignores_unknown():
    p = Params.from_dict({"strength": 3, "shadows": -5, "db": -1, "light_angle": 400, "bogus": 1})
    assert p.strength == 1 and p.shadows == -1 and p.db == 0 and p.light_angle == 40
    assert Params.from_dict(None) == Params()
    assert Params().db == 0.4 and Params().plastic == 0.3 and Params().light_angle is None

def test_brush_map_signs():
    m = brush_map([{"mode": "dodge", "size": 0.05, "strength": 1, "points": [[0.25, 0.5], [0.3, 0.5]]},
                   {"mode": "burn", "size": 0.05, "strength": 0.5, "points": [[0.75, 0.5]]}], 100, 200)
    assert m[50, 55] > 0.5 and m[50, 150] < -0.2 and abs(m[5, 100]) < 1e-3 and m.max() <= 1

@pytest.fixture(scope="module")
def classic():
    e = Engine(None, "classic", depth_ai=False); src, ref = synth(seed=1), synth(seed=2, tint=(1, .75, .55))
    ra = e.analyse_reference(ref); return e, src, ref, ra, e.fit(src, ra)

def test_classic_transfer_moves_towards_reference(classic):
    e, src, ref, ra, fit = classic
    out = e.render(src, fit, Params(plastic=0, vignette=0, db=0))
    assert out.shape == src.shape and out.dtype == np.uint8
    assert C.swd(out, ref) < 0.6*C.swd(src, ref)

def test_strength_zero_keeps_source(classic):
    e, src, ref, ra, fit = classic
    out = e.render(src, fit, Params(strength=0, plastic=0, vignette=0, db=0))
    assert np.abs(out.astype(int)-src).mean() < 1.5

def test_fit_is_resolution_independent(classic):
    e, src, ref, ra, fit = classic
    big = np.ascontiguousarray(np.repeat(np.repeat(src, 3, 0), 3, 1))
    a = C.small(e.render(big, fit, Params(db=0)), src.shape[1]); b = e.render(src, fit, Params(db=0))
    assert np.abs(a.astype(int)-b).mean() < 4

def test_shadows_highlights_controls(classic):
    e, src, ref, ra, fit = classic
    base = C.to_lab(e.render(src, fit, Params(plastic=0, db=0)))[..., 0].mean()
    up = C.to_lab(e.render(src, fit, Params(plastic=0, db=0, shadows=1, highlights=1)))[..., 0].mean()
    assert up > base + 2

def test_light_vec_and_estimate_on_sphere():
    h = w = 200; y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x-100)**2+(y-100)**2)/80**2; z = np.sqrt(np.clip(1-r2, 0, 1))
    n = np.dstack([(x-100)/80, (y-100)/80, z]); n /= np.linalg.norm(n, axis=2, keepdims=True)
    for ang in (30, 120, 250):
        l = RL.light_vec(ang); shade = np.clip(n@l, 0, 1)
        img = (np.repeat((40+180*shade)[..., None], 3, 2)).astype(np.uint8)
        est = RL.estimate_light(img, n.astype(np.float32), (r2 < 0.8).astype(np.float32))
        assert min(abs(est-ang), 360-abs(est-ang)) < 15

def test_relight_gain_zero_and_bounded():
    img = synth(); rl = RL.Relight(img, use_ai=False)
    Lq = C.to_lab(img)[..., 0]
    assert np.all(rl.gain(60, 90, C.small(Lq[..., None].repeat(3, 2).astype(np.uint8), 90)[..., 0].astype(np.float32), 0) == 0)
    g = rl.gain(img.shape[0], img.shape[1], Lq, 1.0, 45)
    assert np.isfinite(g).all() and np.abs(g).max() < 0.6

def test_relight_with_depth_model_if_bundled():
    if RL._session() is None: pytest.skip("depth model not bundled")
    d = RL.depth_ai(synth()); assert d is not None and d.min() >= 0 and d.max() <= 1

@pytest.mark.skipif(not (REAL_SRC.exists() and REAL_REF.exists()), reason="Jack4B photos not on this machine")
def test_real_photos_light_model_quality_and_speed():
    from retusz import imageio
    a, _ = imageio.load(REAL_SRC); b, _ = imageio.load(REAL_REF); bc = C.crop_border(b)
    path, eff = models.resolve("ai_light"); e = Engine(path, eff)
    ra = e.analyse_reference(b); e.process(C.small(a, 800), ra, Params())   # warm-up
    t = time.perf_counter(); out = e.process(a, ra, Params(plastic=0)); dt = time.perf_counter()-t
    assert C.swd(out, bc) < 0.2*C.swd(a, bc)
    assert C.de_mean(out, bc) < 2.0
    assert dt < 5.0, f"24 Mpx took {dt:.1f} s"

def test_dodge_burn_has_no_seams_between_close_faces():
    """Regression (0.1.0 export): overlapping per-face crops left hard vertical/horizontal seams on faces."""
    import cv2
    from retusz.engine.pipeline import _dodge_burn_region
    rng = np.random.default_rng(1); H, W = 600, 900
    base = np.dstack([np.full((H, W), v, np.float32) for v in (200, 150, 120)])
    blot = cv2.GaussianBlur(rng.normal(0, 25, (H, W)).astype(np.float32), (0, 0), 6)[..., None]
    img = np.clip(base + blot, 0, 255).astype(np.uint8)
    mq = np.ones((H//4, W//4), np.float32)
    faces = [(250, 200, 160, 200), (390, 180, 160, 200), (520, 230, 150, 190)]      # overlapping crops
    out = _dodge_burn_region(img, mq, faces, 0.8)
    d = out.astype(np.float32).mean(2) - img.astype(np.float32).mean(2)
    col = np.abs(np.diff(d, axis=1)).mean(0); row = np.abs(np.diff(d, axis=0)).mean(1)
    assert np.abs(d[200:400, 300:600]).mean() > 0.3                 # D&B did something
    assert col.max() < 3*np.median(col) + 0.1 and row.max() < 3*np.median(row) + 0.1   # no line-like steps

def test_skin_mask_does_not_depend_on_resolution_with_noisy_raw():
    """Regression (0.1.3, his D700 NEF): unprocessed sensor noise let dark pixels pass the skin-colour test
    far more often at full resolution than in the 1600 px preview (shadow mask 0.48 vs 0.27), so the skin
    LUT lifted the export's shadows by +1.2 L* compared with the preview."""
    import cv2
    rng = np.random.default_rng(3); H, W = 1600, 2400
    y, x = np.mgrid[0:H, 0:W]
    img = np.empty((H, W, 3), np.float32); img[:] = (60, 55, 50)                   # dark shadow, Cr 130.6: just outside the skin range
    face = ((x-1200)/260)**2 + ((y-700)/330)**2 < 1; img[face] = (205, 150, 125)   # a face
    full = np.clip(img + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)    # high-ISO noise
    prev = cv2.resize(full, (W//3*2//2, H//3*2//2), interpolation=cv2.INTER_AREA)   # 'preview' at 1/3
    e = Engine(None, "classic", depth_ai=False)
    fit = e.fit(prev, e.analyse_reference(synth(seed=2)))
    fit.faces_n = [np.array([940, 370, 520, 660], np.float64)/W]            # faces are normalised to image width
    _, mp = e.render(prev, fit, Params(db=0, plastic=0, skin_even=0, vignette=0), return_mask=True)
    _, mf = e.render(full, fit, Params(db=0, plastic=0, skin_even=0, vignette=0), return_mask=True)
    mf = cv2.resize(mf, (mp.shape[1], mp.shape[0]), interpolation=cv2.INTER_AREA)
    shadow = cv2.resize((~face).astype(np.uint8), (mp.shape[1], mp.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    assert abs(float(mf[shadow].mean()) - float(mp[shadow].mean())) < 0.02
    assert abs(float(mf[~shadow].mean()) - float(mp[~shadow].mean())) < 0.05


def test_standalone_fit_without_reference_works_and_exports(tmp_path):
    """Studio editor mode (0.2.0): Engine.fit(src, None) and batch.export_one(e, None, ...) work without a reference."""
    from PIL import Image
    from retusz import batch, export
    src = synth(seed=11)
    e = Engine(None, "classic", depth_ai=False)
    fit = e.fit(src, None)
    assert fit.method == "standalone"
    # Zero adjustments + no reference -> identity colour
    out0 = e.render(src, fit, Params(strength=1.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0))
    assert np.abs(out0.astype(int) - src.astype(int)).max() <= 1
    # Exposure +0.5 EV brightens the image without needing a reference
    out_ev = e.render(src, fit, Params(exposure=0.5, db=0.0, plastic=0.0))
    assert out_ev.mean() > src.mean() + 8.0
    # Standalone export_one with reference=None
    ph = tmp_path / "portret.jpg"
    Image.fromarray(src).save(ph, quality=95)
    dst = batch.export_one(e, None, ph, Params(exposure=0.25, contrast=0.1), tmp_path / "Retusz", export.ExportOptions())
    assert dst.exists()

