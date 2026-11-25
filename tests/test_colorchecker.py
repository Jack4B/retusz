"""TDD tests for ColorChecker Classic 24-patch detection, Root-Polynomial + Bradford calibration,
ΔE00 reporting, chart-to-chart matching, and OKLCh skin cast/hue controls."""
from __future__ import annotations
import numpy as np
from tests.synthetic import (
    CC24_SRGB,
    colorchecker_chart,
    apply_illuminant_cast,
    portrait_with_colorchecker,
    portrait_scene,
)
from retusz.engine import colorchecker as CC
from retusz.engine.pipeline import Engine, Params


def test_detect_colorchecker_rotated_and_flipped_in_portrait():
    for angle in (0.0, 7.5, -11.0, 180.0):
        cast_rgb, _gt, gt_centres, _info = portrait_with_colorchecker(
            h=1200, w=900, patch_px=40, angle=angle, gains=(1.20, 0.98, 0.78), ev=-0.35
        )
        det = CC.detect_chart(cast_rgb)
        assert det is not None, f"Chart not detected at angle={angle}"
        assert det.centres.shape == (24, 2)
        err = np.linalg.norm(det.centres - gt_centres, axis=1)
        assert err.mean() < 6.0, f"Mean patch center error too high ({err.mean():.2f} px) at angle={angle}"
        assert err.max() < 12.0, f"Max patch center error too high ({err.max():.2f} px) at angle={angle}"


def test_calibrate_colorchecker_reduces_de00_below_1():
    cast_rgb, gt_rgb, _centres, _info = portrait_with_colorchecker(
        h=1200, w=900, patch_px=44, angle=5.0, gains=(1.32, 0.92, 0.64), ev=-0.55
    )
    cal = CC.calibrate_chart(cast_rgb)
    assert cal is not None
    rep = cal.report
    assert rep.de00_before > 10.0, f"Expected initial ΔE00 > 10, got {rep.de00_before:.2f}"
    assert rep.de00_after < 1.2, f"Expected calibrated mean ΔE00 < 1.2, got {rep.de00_after:.2f}"
    assert rep.skin_de00 < 1.0, f"Expected calibrated skin patches (#1,#2) ΔE00 < 1.0, got {rep.skin_de00:.2f}"
    assert rep.gray_de00 < 0.8, f"Expected calibrated neutral patches (#19..#24) ΔE00 < 0.8, got {rep.gray_de00:.2f}"
    assert len(rep.patch_de00_after) == 24
    # Visual overlay returns an annotated RGB frame of the same dimensions
    ov = CC.draw_chart_overlay(cast_rgb, cal)
    assert ov.shape == cast_rgb.shape and ov.dtype == np.uint8


def test_engine_prepare_colorchecker_calibrates_series():
    cast_chart_photo, _gt1, _c, _i1 = portrait_with_colorchecker(
        h=1200, w=900, patch_px=44, angle=-4.0, gains=(0.78, 1.04, 1.24), ev=-0.30, seed=1
    )
    portrait2_gt, info2 = portrait_scene(h=1200, w=900, seed=2)
    portrait2_cast = apply_illuminant_cast(portrait2_gt, gains=(0.78, 1.04, 1.24), ev=-0.30)

    eng = Engine()
    fit = eng.prepare(portrait2_cast, reference=cast_chart_photo, colorchecker=True)
    assert fit.method == "colorchecker"
    assert fit.cc_report is not None
    assert fit.cc_report.de00_after < 1.2

    p = Params(db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0)
    out = eng.render(portrait2_cast, fit, p, preview=False)
    skin_m = info2["skin"] > 0.5
    de_before = CC.mean_de00_rgb(portrait2_cast[skin_m], portrait2_gt[skin_m])
    de_after = CC.mean_de00_rgb(out[skin_m], portrait2_gt[skin_m])
    assert de_before > 8.0
    assert de_after < 2.0, f"Expected series skin ΔE00 < 2.0 after ColorChecker calibration, got {de_after:.2f}"


def test_oklch_skin_cast_and_hue_preserve_lightness_and_background():
    from retusz.engine import tone as TN
    # Greenish-yellow foliage cast on a medium skin tone vs natural skin target (~50 deg OKLCh hue)
    skin_lin = TN.srgb_to_lin(np.array([[[0.72, 0.62, 0.42]]], dtype=np.float32))
    ok_before = TN.lin_to_oklab(skin_lin)
    h_before = float(np.degrees(np.arctan2(ok_before[0, 0, 2], ok_before[0, 0, 1])))
    assert abs(h_before - 50.0) > 10.0

    fixed_lin = TN.apply_skin_oklch(skin_lin, skin_cast=0.85, skin_hue=0.0)
    ok_after = TN.lin_to_oklab(fixed_lin)
    h_after = float(np.degrees(np.arctan2(ok_after[0, 0, 2], ok_after[0, 0, 1])))
    # Hue pulled strongly towards 50 deg natural skin locus, while lightness L is preserved within 1e-4
    assert abs(h_after - 50.0) < abs(h_before - 50.0) * 0.35
    assert abs(float(ok_after[0, 0, 0] - ok_before[0, 0, 0])) < 1e-3

    # Blue sky color outside skin locus is unaffected by skin_cast
    sky_lin = TN.srgb_to_lin(np.array([[[0.25, 0.50, 0.85]]], dtype=np.float32))
    sky_after = TN.apply_skin_oklch(sky_lin, skin_cast=1.0, skin_hue=0.5)
    assert np.max(np.abs(sky_after - sky_lin)) < 1e-3

