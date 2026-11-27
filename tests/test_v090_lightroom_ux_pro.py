"""TDD test suite for Retusz v0.9.0 Lightroom Studio Pro:
1. MIT license consistency & SHA-256 verification gate for all bundled ONNX models
2. Intel iGPU acceleration: OpenCV OpenCL (cv2.ocl / UMat) + DirectML session configuration
3. Auto Ton AI v2 (ITA° skin phototype angle Chardon 1991 + low-key/high-key intent) & Auto WB v2
4. Two-stage YuNet anatomical landmark verification & cross-tile IoU deduplication
5. Non-destructive Frequency-Separated Pore-Preserving Healing Brush ('Usuń / Popraw', heal_strokes)
6. Full 8-channel OKLCh HSL Mixer (8 hues x H/S/L) + 3-way Color Grading (midtone_tone, grading_balance)
7. Lightroom Classic UI mockup alignment (5 clean tabs, neutral slider tracks, flags P/X/U, selective copy)

Author: Jack4B.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import tomllib
import numpy as np
import cv2

from retusz.engine import device as DEV, tone as TN, wb as WB, faces as F, skin_physics as SP, pipeline as P


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_consistency_and_model_sha256_gate():
    lic = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in lic and "Jack4B" in lic

    import retusz
    pyproj = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    assert pyproj["project"]["version"] == retusz.__version__
    assert cargo["workspace"]["package"]["version"] == retusz.__version__
    lic_val = pyproj["project"]["license"]
    lic_str = lic_val.get("text", "") if isinstance(lic_val, dict) else str(lic_val)
    assert "MIT" in lic_str and "Proprietary" not in lic_str

    tpl = (REPO_ROOT / "THIRD_PARTY_LICENSES.md").read_text(encoding="utf-8")
    onnx_files = sorted((REPO_ROOT / "retusz" / "assets" / "models").glob("*.onnx"))
    assert len(onnx_files) >= 3
    for mf in onnx_files:
        digest = hashlib.sha256(mf.read_bytes()).hexdigest()
        assert mf.name in tpl, f"Bundled model {mf.name} missing from THIRD_PARTY_LICENSES.md"
        assert digest in tpl, f"SHA-256 {digest} for {mf.name} missing from THIRD_PARTY_LICENSES.md"


def test_intel_igpu_opencl_and_directml_device_configuration():
    from retusz import settings as S
    assert "ocl" in S.DEVICES and "gpu" in S.DEVICES and "auto" in S.DEVICES and "cpu" in S.DEVICES

    info = DEV.ocl_configure("ocl")
    assert isinstance(info, dict) and "available" in info and "enabled" in info
    # Large frame (>= 0.4 Mpx) blur & resize must return exact numpy uint8/float32 arrays with or without OpenCL ICD
    big = np.random.default_rng(7).uniform(0, 255, size=(700, 800, 3)).astype(np.float32)
    blurred = DEV.ocl_gaussian_blur(big, 1.5)
    assert isinstance(blurred, np.ndarray) and blurred.shape == big.shape and blurred.dtype == np.float32
    resized = DEV.ocl_resize(big.astype(np.uint8), (400, 350))
    assert isinstance(resized, np.ndarray) and resized.shape == (350, 400, 3)

    DEV.ocl_configure("cpu")
    assert not DEV.ocl_info()["enabled"]


def test_auto_tone_v2_ita_skin_typology_protects_dark_skin_and_low_key():
    # Deep dark complexion portrait (Fitzpatrick V/VI: sRGB ~ [98, 66, 48], CIELAB L* ~ 31, ITA° < -20°)
    # on a dark low-key background: Auto Ton v2 must NOT bleach dark skin toward L*=67!
    H, W = 240, 320
    low_key_dark_skin = np.full((H, W, 3), [24, 24, 28], np.uint8)
    low_key_dark_skin[50:190, 95:225] = [104, 70, 52]
    faces_n = [np.array([95 / W, 50 / W, 130 / W, 140 / W], np.float32)]

    ita_dark = TN.skin_ita_deg(low_key_dark_skin, faces_n)
    assert ita_dark is not None and ita_dark < 10.0, f"Expected dark phototype ITA° < 10°, got {ita_dark}"

    auto_dark_skin = TN.auto_tone_params(low_key_dark_skin, faces_n=faces_n)
    assert auto_dark_skin["exposure"] <= 0.70, f"Auto Ton v2 must cap exposure on dark phototype/low-key: {auto_dark_skin}"

    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(low_key_dark_skin, None)
    out = eng.render(low_key_dark_skin, fit, P.Params.from_dict(auto_dark_skin), db_enabled=False)
    lab_out = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    skin_L_star = float(lab_out[80:160, 120:200, 0].mean())
    assert 30.0 <= skin_L_star <= 56.0, f"Dark complexion L* was washed out: L*={skin_L_star:.1f}"


def test_auto_wb_v2_shades_of_gray_neutralizes_cast_while_sparing_skin():
    H, W = 200, 260
    # Neutral grey studio backdrop + warm skin face in center, then apply strong tungsten warm-green cast
    scene = np.full((H, W, 3), [155, 155, 155], np.float32)
    scene[40:160, 80:180] = [205, 158, 132]
    # Simulate warm/green ambient cast: boost R & G, drop B
    cast_scene = np.clip(scene * np.array([1.18, 1.06, 0.78], np.float32), 0, 255).astype(np.uint8)
    faces_n = [np.array([80 / W, 40 / W, 100 / W, 120 / W], np.float32)]

    t_corr, ti_corr = WB.auto_wb_scene(cast_scene, faces_n=faces_n)
    assert t_corr < -0.12, f"Expected cooling temperature correction for warm tungsten cast, got temp={t_corr}"
    fixed = (WB.apply_bradford01(cast_scene.astype(np.float32) / 255.0, t_corr, ti_corr) * 255.0).astype(np.uint8)
    bg_before = cast_scene[:30, :60].mean(axis=(0, 1))
    bg_after = fixed[:30, :60].mean(axis=(0, 1))
    assert (bg_after.max() - bg_after.min()) < (bg_before.max() - bg_before.min()) * 0.45


def test_two_stage_yunet_verification_and_group_tile_iou():
    H, W = 200, 200
    rgb = np.full((H, W, 3), [80, 85, 90], np.uint8)
    rgb[40:160, 50:150] = [202, 158, 134]  # valid skin face region
    box = np.array([50.0, 40.0, 100.0, 120.0], np.float32)
    valid_lm = np.array([
        [78.0, 82.0],   # right eye
        [122.0, 83.0],  # left eye
        [100.0, 106.0], # nose
        [84.0, 130.0],  # mouth right
        [116.0, 130.0], # mouth left
    ], np.float32)
    assert F.verify_face_landmarks(rgb, box, valid_lm) is True

    # Implausible landmarks: eyes stacked vertically or mouth above nose
    bad_lm = np.array([
        [100.0, 60.0],
        [102.0, 110.0],
        [100.0, 130.0],
        [85.0, 75.0],
        [115.0, 75.0],
    ], np.float32)
    assert F.verify_face_landmarks(rgb, box, bad_lm) is False

    # Cross-tile IoU deduplication
    b1 = np.array([50.0, 40.0, 100.0, 120.0], np.float32)
    b2 = np.array([54.0, 43.0, 98.0, 118.0], np.float32)   # duplicate of b1 across tile seam
    b3 = np.array([220.0, 45.0, 95.0, 115.0], np.float32)  # second person in couple shot
    kept_b, kept_l = F.nms_boxes_iou([b1, b2, b3], [valid_lm, valid_lm, valid_lm + [170, 0]], [0.85, 0.72, 0.68])
    assert len(kept_b) == 2 and len(kept_l) == 2


def test_frequency_separated_healing_brush_removes_spot_preserving_pores():
    H, W = 220, 220
    rng = np.random.default_rng(123)
    pores = rng.normal(0.0, 5.2, size=(H, W, 1)).astype(np.float32)
    skin = np.clip(np.array([204.0, 160.0, 136.0], np.float32) + pores, 0, 255)
    # Inject an inflamed dark-red blemish/spot at center (110, 110), radius 6 px
    yy, xx = np.ogrid[:H, :W]
    spot_m = np.exp(-((xx - 110) ** 2 + (yy - 110) ** 2) / (2.0 * 4.2 ** 2)).astype(np.float32)[..., None]
    img = np.clip(skin * (1.0 - spot_m) + np.array([135.0, 58.0, 52.0], np.float32) * spot_m, 0, 255).astype(np.uint8)

    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)
    p = P.Params(db=0.0, plastic=0.0, skin_even=0.0,
                 heal_strokes=[{"points": [[110 / W, 110 / H]], "size": 12.0 / W}])
    healed = eng.render(img, fit, p, db_enabled=False)

    # Blemish green-channel depression at center must be >75% eliminated
    ref_mean_g = float(img[40:80, 40:80, 1].mean())
    err_before = abs(float(img[106:115, 106:115, 1].mean()) - ref_mean_g)
    err_after = abs(float(healed[106:115, 106:115, 1].mean()) - ref_mean_g)
    assert err_after < err_before * 0.25, f"Expected spot to be healed: err {err_before:.1f} -> {err_after:.1f}"

    # Pore high-frequency texture inside healed patch must not be blurred into flat plastic
    hp_ref = (healed[40:80, 40:80, 1].astype(np.float32)
              - cv2.GaussianBlur(healed[40:80, 40:80, 1].astype(np.float32), (0, 0), 1.5)).std()
    hp_healed = (healed[102:118, 102:118, 1].astype(np.float32)
                 - cv2.GaussianBlur(healed[102:118, 102:118, 1].astype(np.float32), (0, 0), 1.5)).std()
    assert hp_healed >= 0.72 * hp_ref, f"Expected pore micro-texture inside healed patch: {hp_healed:.2f} vs {hp_ref:.2f}"


def test_full_8_channel_oklch_hsl_mixer_and_3way_color_grading():
    p0 = P.Params()
    p1 = P.Params(
        midtone_tone=0.45,
        grading_balance=-0.35,
        hsl_hue=[0.2, 0.0, -0.3, 0.4, 0.0, -0.2, 0.1, 0.0],
        hsl_sat=[-0.2, 0.0, 0.3, -0.5, 0.2, 0.4, 0.0, -0.3],
        hsl_lum=[0.0, 0.0, 0.1, -0.2, 0.0, -0.3, 0.0, 0.2],
        flag=1,
    )
    assert p0.colour_key() != p1.colour_key()
    rt = P.Params.from_dict(p1.to_dict())
    assert rt.flag == 1
    assert abs(rt.midtone_tone - 0.45) < 1e-5
    assert abs(rt.grading_balance - (-0.35)) < 1e-5
    assert np.allclose(rt.hsl_sat, p1.hsl_sat, atol=1e-5)

    # Red dress patch vs skin patch: shifting Red saturation (channel 0) & Purple saturation (channel 6)
    red_lin = TN.srgb_to_lin(np.full((6, 6, 3), [0.82, 0.14, 0.16], np.float32))
    skin_lin = TN.srgb_to_lin(np.full((6, 6, 3), [0.78, 0.59, 0.50], np.float32))
    red_out = TN.apply_color_grading_oklab(red_lin, hsl_sat=[-0.8, 0, 0, 0, 0, 0, 0, 0], is_skin=False)
    skin_out = TN.apply_color_grading_oklab(skin_lin, hsl_sat=[-0.8, 0, 0, 0, 0, 0, 0, 0], is_skin=True)
    assert np.abs(red_out - red_lin).mean() > 0.03
    assert np.abs(skin_out - skin_lin).max() < 0.012


def test_lightroom_mockup_ui_tabs_neutral_sliders_and_ux_helpers():
    from retusz.gui import app as APP
    assert APP.SLIDER_TABS == ["Podstawowe", "Portret 3D", "Kolor HSL", "Detal", "Wzorzec"]
    basic_keys = [row[0] for row in APP.SLIDER_GROUPS["Podstawowe"]]
    assert basic_keys == [
        "temp", "tint",
        "exposure", "contrast", "brightness",
        "highlights", "shadows", "whites", "blacks",
        "vibrance", "saturation",
    ]
    for alias in ("Światło i Kolor", "Portret i Plastyka", "Detal i Ostrość", "▸ Wzorzec (AI)"):
        assert alias in APP._TAB_ALIASES and APP._TAB_ALIASES[alias] in APP.SLIDER_TABS
