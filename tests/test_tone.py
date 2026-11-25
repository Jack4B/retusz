"""TDD tests for engine/tone.py (OKLab, Exposure EV, Contrast, Highlights/Shadows, Whites/Blacks,
OKLab Vibrance/Saturation, Guided Clarity), Bradford WB, tetrahedral LUTs, and pipeline integration."""
import numpy as np, cv2
from conftest import synth
from synthetic import portrait_scene
from retusz.engine import tone as T, wb as WB, color as C, render as R
from retusz.engine.pipeline import Engine, Params


def test_oklab_roundtrip_accurate():
    rng = np.random.default_rng(0)
    lin = rng.uniform(0.0, 1.0, (200, 3)).astype(np.float32)
    back = T.oklab_to_lin(T.lin_to_oklab(lin))
    assert np.abs(back - lin).max() < 1e-4


def test_exposure_identity_monotonic_and_filmic_shoulder():
    lin = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None].repeat(3, 1)
    assert np.array_equal(T.apply_exposure_lin(lin, 0.0), lin)
    # midtone scaling ~ 2^EV
    mid = np.array([[0.18, 0.18, 0.18]], np.float32)
    assert abs(float(T.apply_exposure_lin(mid, 1.0)[0, 0]) - 0.36) < 0.01
    assert abs(float(T.apply_exposure_lin(mid, -1.0)[0, 0]) - 0.09) < 0.005
    # monotonic ramp at +2 EV and -2 EV
    for ev in (-2.0, -1.0, 1.0, 2.0):
        out = T.apply_exposure_lin(lin, ev)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert np.all(np.diff(out[:, 0]) >= -1e-6)
    # filmic shoulder preserves skin hue under strong +2 EV push (unlike per-channel clip)
    skin = np.array([[0.65, 0.38, 0.25]], np.float32)
    pushed = T.apply_exposure_lin(skin, 2.0)
    ok0 = T.lin_to_oklab(skin)[0]; ok1 = T.lin_to_oklab(pushed)[0]
    h0 = np.degrees(np.arctan2(ok0[2], ok0[1])); h1 = np.degrees(np.arctan2(ok1[2], ok1[1]))
    assert abs(h1 - h0) < 4.0, f"hue shifted under +2 EV shoulder: {h0:.1f} -> {h1:.1f}"
    assert pushed.max() < 1.0  # smooth shoulder, not hard-clipped to 1.0


def test_tone_controls_identity_direction_separation_and_monotonicity():
    ramp = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None].repeat(3, 1)
    assert np.allclose(T.apply_tone_lin(ramp), ramp, atol=1e-5)

    # Contrast increases spread around mid-grey without shifting endpoints wildly
    c_pos = T.apply_tone_lin(ramp, contrast=0.8)[:, 0]
    c_neg = T.apply_tone_lin(ramp, contrast=-0.8)[:, 0]
    assert c_pos[200] > ramp[200, 0] and c_pos[40] < ramp[40, 0]
    assert c_neg[200] < ramp[200, 0] and c_neg[40] > ramp[40, 0]
    assert np.all(np.diff(c_pos) >= -1e-5) and np.all(np.diff(c_neg) >= -1e-5)

    # Shadows move darks much more than highlights; highlights move brights much more than darks
    sh_up = T.apply_tone_lin(ramp, shadows=0.8)[:, 0]
    hi_dn = T.apply_tone_lin(ramp, highlights=-0.8)[:, 0]
    assert (sh_up[20] - ramp[20, 0]) > 4.0 * abs(sh_up[210] - ramp[210, 0])
    assert (ramp[200, 0] - hi_dn[200]) > 4.0 * abs(hi_dn[20] - ramp[20, 0])

    # Blacks lift pure black cleanly; whites move upper endpoint cleanly
    bl_up = T.apply_tone_lin(ramp, blacks=0.8)[:, 0]
    wh_up = T.apply_tone_lin(ramp, whites=0.8)[:, 0]
    assert bl_up[0] > 0.001 and abs(bl_up[220] - ramp[220, 0]) < 0.01
    assert wh_up[210] > ramp[210, 0] + 0.02 and abs(wh_up[15] - ramp[15, 0]) < 0.005

    # Monotonicity under extreme combined pushes
    for kw in (dict(contrast=1.0), dict(contrast=-1.0), dict(shadows=1.0, highlights=-1.0), dict(blacks=-1.0, whites=1.0)):
        out = T.apply_tone_lin(ramp, **kw)[:, 0]
        assert np.all(np.diff(out) >= -1e-4), kw


def test_oklab_vibrance_spares_skin_and_keeps_blue_hue():
    # Blue sky should keep its OKLab hue when saturation increases (no CIELAB purple shift)
    blue = np.array([[0.03, 0.16, 0.72]], np.float32)
    b1 = T.apply_color_oklab(blue, saturation=0.6, vibrance=0.0)
    ok0, ok1 = T.lin_to_oklab(blue)[0], T.lin_to_oklab(b1)[0]
    h0 = np.degrees(np.arctan2(ok0[2], ok0[1])); h1 = np.degrees(np.arctan2(ok1[2], ok1[1]))
    assert abs(h1 - h0) < 2.0, f"blue hue shifted: {h0:.1f} -> {h1:.1f}"

    # Vibrance boosts dull non-skin much more than skin tones
    dull_teal = np.array([[0.18, 0.24, 0.25]], np.float32)
    skin = np.array([[0.52, 0.28, 0.20]], np.float32)
    dt1 = T.apply_color_oklab(dull_teal, vibrance=0.8, is_skin=False)
    sk1 = T.apply_color_oklab(skin, vibrance=0.8, is_skin=True)
    c = lambda x: float(np.hypot(*T.lin_to_oklab(x)[0, 1:]))
    assert (c(dt1) / c(dull_teal)) > 1.25 * (c(sk1) / c(skin))


def test_clarity_gain_skin_protected_and_signed():
    rgb, info = portrait_scene(600, 400, seed=3)
    Lq = C.to_lab(cv2.resize(rgb, (100, 150), interpolation=cv2.INTER_AREA))[..., 0]
    sk = cv2.resize(info["skin"], (100, 150), interpolation=cv2.INTER_AREA)
    assert np.all(T.clarity_gain(Lq, sk, 0.0) == 0)
    gp = T.clarity_gain(Lq, sk, 1.0)
    gn = T.clarity_gain(Lq, sk, -1.0)
    assert np.isfinite(gp).all() and np.abs(gp).max() <= 0.26
    # Positive clarity boosts texture on non-skin strongly while sparing skin
    non_skin = (sk < 0.1) & (Lq > 15) & (Lq < 85)
    skin_m = (sk > 0.8)
    assert np.abs(gp[non_skin]).mean() > 1.5 * np.abs(gp[skin_m]).mean()
    # Positive and negative clarity act in opposite directions
    assert float((gp * gn).mean()) < 0


def test_bradford_wb_matches_neutrals_and_warms():
    grey = np.full((5, 5, 3), 0.5, np.float32)
    b_out = WB.apply_bradford01(grey, 0.35, -0.2)
    assert b_out[0, 0, 0] > b_out[0, 0, 2]
    assert abs(float(WB.to_lin(b_out[0, 0]) @ WB.Y709) - float(WB.to_lin(grey[0, 0]) @ WB.Y709)) < 1e-4
    # Exact one-click eyedropper round-trip on a saturated colour under Bradford CAT
    col = np.array([[[0.25, 0.45, 0.70]]], np.float32)
    shifted = WB.apply_bradford01(col, 0.3, -0.2)
    d = WB.solve(np.clip(shifted*255 + 0.5, 0, 255).astype(np.uint8))
    assert d is not None
    neutral = WB.apply_bradford01(col, 0.3 + d[0], -0.2 + d[1])
    assert abs(float(neutral[0, 0, 0] - neutral[0, 0, 1])) < 0.015
    assert abs(float(neutral[0, 0, 1] - neutral[0, 0, 2])) < 0.015
    assert np.array_equal(WB.apply_bradford01(grey, 0.0, 0.0), grey)


def test_tetrahedral_lut_preserves_neutral_diagonal_and_matches_numba():
    G = C.rgb_grid(17)
    # Non-linear curve on diagonal + asymmetric off-diagonal perturbation
    lut = np.clip(G**1.15, 0, 1).astype(np.float32)
    lut[1:, 0, 0, 0] = np.clip(lut[1:, 0, 0, 0] + 0.1, 0, 1)
    # Neutral gray ramp must remain strictly neutral (R == G == B) under tetrahedral interpolation
    ramp = np.arange(0, 256, dtype=np.uint8)[:, None, None].repeat(3, 2)
    out = C.apply_lut(ramp, lut)
    assert np.abs(out[..., 0] - out[..., 1]).max() < 1e-5
    assert np.abs(out[..., 1] - out[..., 2]).max() < 1e-5
    # Numba render matches render_numpy
    img = synth(120, 180)
    q = np.zeros((30, 45), np.float32)
    a = R.render(img, lut, lut, q, q)
    b = R.render_numpy(img, lut, lut, q, q)
    assert np.abs(a.astype(int) - b).max() <= 1


def test_pipeline_new_params_render_and_preview_equals_export():
    e = Engine(None, "classic", depth_ai=False)
    src, _ = portrait_scene(1200, 800, seed=2)
    ref = synth(300, 450, seed=9)
    fit = e.fit(src, e.analyse_reference(ref))
    p = Params(exposure=0.75, contrast=0.4, highlights=-0.3, shadows=0.4, whites=0.2, blacks=-0.2, clarity=0.5, db=0, plastic=0)
    # Clamping & colour_key
    p2 = Params.from_dict({"exposure": 9.0, "contrast": -5.0, "whites": 3.0, "blacks": -3.0, "clarity": 2.0})
    assert p2.exposure == 2.0 and p2.contrast == -1.0 and p2.whites == 1.0 and p2.blacks == -1.0 and p2.clarity == 1.0
    assert p.colour_key() != Params().colour_key()
    # Direction check: +exposure makes image clearly brighter
    base = C.to_lab(e.render(src, fit, Params(db=0, plastic=0)))[..., 0].mean()
    bright = C.to_lab(e.render(src, fit, Params(exposure=1.0, db=0, plastic=0)))[..., 0].mean()
    assert bright > base + 6.0
    # Preview (1/2) vs full-res export ΔE < 1.0
    full = e.render(src, fit, p)
    prev_src = cv2.resize(src, (400, 600), interpolation=cv2.INTER_AREA)
    prev = e.render(prev_src, fit, p)
    ref_small = cv2.resize(full, (400, 600), interpolation=cv2.INTER_AREA)
    de = float(np.linalg.norm(C.to_lab(prev) - C.to_lab(ref_small), axis=2).mean())
    assert de < 1.0, f"preview vs export dE={de:.3f}"
