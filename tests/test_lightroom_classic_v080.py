"""TDD tests for Retusz v0.8.0 Lightroom Classic Pro:
- Auto Ton AI (scene + face-aware exposure/tone solver)
- OKLab Color Grading (shadow_tone, highlight_tone) & Selective OKLCh HSL Mixer (hsl_green, hsl_blue)
- Multi-face detection with glasses/occlusion & dark frame/beard protection in skin_mask
- Authentic Single-Column Lightroom Classic UI slider definitions
Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
from retusz.engine import tone as TN, faces as F, pipeline as P


def _synthetic_portrait(h: int = 240, w: int = 320, scale: float = 1.0) -> np.ndarray:
    """Synthetic portrait with dark shadow region, mid-tone skin region, foliage green, sky blue, and bright highlight."""
    img = np.zeros((h, w, 3), np.float32)
    # Dark background shadows on left
    img[:, :w // 4] = [28.0, 30.0, 36.0]
    # Foliage green region
    img[:h // 2, w // 4:w // 2] = [65.0, 150.0, 55.0]
    # Sky / shirt blue region
    img[h // 2:, w // 4:w // 2] = [70.0, 130.0, 205.0]
    # Bright highlight region on right
    img[:, 3 * w // 4:] = [225.0, 222.0, 215.0]
    # Central face skin region (x: 0.3..0.7, y: 0.2..0.7)
    y0, y1 = int(0.2 * h), int(0.7 * h)
    x0, x1 = int(0.3 * w), int(0.7 * w)
    img[y0:y1, x0:x1] = [198.0, 152.0, 128.0]
    img = np.clip(img * scale, 0.0, 255.0).astype(np.uint8)
    return img


def test_auto_tone_ai_balances_underexposed_and_bright_portraits_without_clipping():
    dark = _synthetic_portrait(scale=0.45)
    faces_n = [np.array([0.30, 0.20, 0.40, 0.50], np.float32)]
    auto_dark = TN.auto_tone_params(dark, faces_n=faces_n)
    assert isinstance(auto_dark, dict)
    assert auto_dark["exposure"] > 0.25, f"Expected positive exposure boost for dark portrait, got {auto_dark}"
    assert auto_dark["shadows"] >= 0.05, f"Expected shadow lift for dark portrait, got {auto_dark}"

    # Apply via Engine in standalone mode and verify skin gets brighter without highlight clipping
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(dark, None)
    p = P.Params.from_dict(auto_dark)
    out = eng.render(dark, fit, p, db_enabled=False)
    H, W = dark.shape[:2]
    skin_before = dark[int(0.3 * H):int(0.6 * H), int(0.4 * W):int(0.6 * W)].mean()
    skin_after = out[int(0.3 * H):int(0.6 * H), int(0.4 * W):int(0.6 * W)].mean()
    assert skin_after > skin_before + 22.0
    clipped_frac = float((out.max(axis=-1) >= 253).mean())
    assert clipped_frac < 0.05

    # Overexposed portrait should receive negative exposure or negative highlights recovery
    bright = _synthetic_portrait(scale=1.32)
    auto_bright = TN.auto_tone_params(bright, faces_n=faces_n)
    assert auto_bright["exposure"] < -0.05 or auto_bright["highlights"] < -0.10


def test_color_grading_shadow_tone_highlight_tone_and_selective_hsl_green_blue():
    # Params serialization & colour_key
    p0 = P.Params()
    p1 = P.Params(shadow_tone=-0.5, highlight_tone=0.6, hsl_green=-0.5, hsl_blue=0.4)
    assert p0.colour_key() != p1.colour_key()
    p_round = P.Params.from_dict(p1.to_dict())
    assert abs(p_round.shadow_tone - (-0.5)) < 1e-5
    assert abs(p_round.highlight_tone - 0.6) < 1e-5
    assert abs(p_round.hsl_green - (-0.5)) < 1e-5
    assert abs(p_round.hsl_blue - 0.4) < 1e-5

    # Verify shadow_tone cools/warms dark pixels while keeping highlights stable
    sh_lin = TN.srgb_to_lin(np.full((8, 8, 3), 0.18, np.float32))
    hi_lin = TN.srgb_to_lin(np.full((8, 8, 3), 0.85, np.float32))
    sh_warm = TN.apply_color_grading_oklab(sh_lin, shadow_tone=0.7)
    sh_cool = TN.apply_color_grading_oklab(sh_lin, shadow_tone=-0.7)
    assert sh_warm[..., 0].mean() > sh_cool[..., 0].mean()  # warmer shadows have more red
    assert sh_warm[..., 2].mean() < sh_cool[..., 2].mean()  # cooler shadows have more blue

    hi_warm = TN.apply_color_grading_oklab(hi_lin, highlight_tone=0.7)
    hi_cool = TN.apply_color_grading_oklab(hi_lin, highlight_tone=-0.7)
    assert hi_warm[..., 0].mean() > hi_cool[..., 0].mean()
    assert hi_warm[..., 2].mean() < hi_cool[..., 2].mean()

    # Verify hsl_green and hsl_blue modify foliage green and sky blue while protecting skin tones
    skin_lin = TN.srgb_to_lin(np.full((8, 8, 3), [0.78, 0.59, 0.50], np.float32))
    green_lin = TN.srgb_to_lin(np.full((8, 8, 3), [0.25, 0.62, 0.22], np.float32))
    blue_lin = TN.srgb_to_lin(np.full((8, 8, 3), [0.24, 0.48, 0.82], np.float32))

    green_shifted = TN.apply_color_grading_oklab(green_lin, hsl_green=-0.8, hsl_blue=-0.8)
    blue_shifted = TN.apply_color_grading_oklab(blue_lin, hsl_green=-0.8, hsl_blue=-0.8)
    skin_shifted = TN.apply_color_grading_oklab(skin_lin, hsl_green=-0.8, hsl_blue=-0.8, is_skin=True)

    assert np.abs(green_shifted - green_lin).mean() > 0.015
    assert np.abs(blue_shifted - blue_lin).mean() > 0.015
    # Skin tone must remain virtually untouched by green/blue HSL sliders
    assert np.abs(skin_shifted - skin_lin).max() < 0.008


def test_multi_face_detection_glasses_threshold_and_dark_frame_beard_attenuation():
    import inspect
    sig = inspect.signature(F.detect)
    assert sig.parameters["score"].default <= 0.48, "Expected YuNet score threshold <= 0.48 for glasses/occluded faces"

    # Create synthetic face image with smooth cheek skin and a warm-tinted dark tortoiseshell glasses frame / beard
    H, W = 200, 200
    rgb = np.full((H, W, 3), [60, 65, 70], np.uint8)
    rgb[30:170, 50:150] = [205, 160, 136]  # skin core (Y ~ 170)
    # Warm-tinted dark tortoiseshell glasses frame and dark brown beard (passes YCrCb chroma gate if not attenuated)
    rgb[66:74, 58:142] = [108, 76, 62]      # warm dark glasses frame (Y ~ 84, within skin_rng)
    rgb[126:136, 75:125] = [104, 74, 60]    # warm dark brown beard/mustache

    fcs = [np.array([50.0, 30.0, 100.0, 140.0], np.float32)]
    m = F.skin_mask(rgb, fcs, hands=False)
    cheek_val = float(m[95:110, 65:135].mean())
    frame_val = float(m[68:72, 70:130].mean())
    beard_val = float(m[129:133, 85:115].mean())
    assert cheek_val > 0.75
    assert frame_val < cheek_val * 0.55, f"Expected dark glasses frame to be attenuated in skin_mask, got {frame_val} vs {cheek_val}"
    assert beard_val < cheek_val * 0.55, f"Expected dark beard/mustache to be attenuated in skin_mask, got {beard_val} vs {cheek_val}"


def test_single_column_lightroom_slider_groups_and_pro_tips():
    from retusz.gui import app as APP
    defaults = P.Params()
    for tab, items in APP.SLIDER_GROUPS.items():
        rows = []
        for entry in items:
            key, label, lo, hi, kind, r_, c_ = entry
            assert c_ == 0, f"Slider {key} in tab {tab} must be in single column (col=0), got col={c_}"
            assert hasattr(defaults, key), f"Slider {key} missing on Params"
            assert key in APP.PRO_TIPS, f"Slider {key} missing PRO_TIPS entry"
            rows.append(r_)
        assert rows == sorted(rows) and len(rows) == len(set(rows)), f"Rows in tab {tab} must be strictly sequential 1-column rows: {rows}"
    assert "mixed_light" in APP.SLIDER_HINTS and "mixed_light" in APP.PRO_TIPS


def test_mixed_light_half_white_half_red_face_unification():
    """Verify that a portrait face split between white flash (left half) and red ambient LED (right half)
    is unified into a natural golden-peach skin tone by mixed_light without losing pore texture."""
    import cv2
    from retusz.engine import skin_physics as SP

    H, W = 240, 240
    rng = np.random.default_rng(42)
    pores = rng.normal(0.0, 4.5, size=(H, W, 1)).astype(np.float32)

    img = np.full((H, W, 3), [45.0, 48.0, 54.0], np.float32)
    # Face box: x in 60..180, y in 40..200 (center x=120)
    # Left half of face (x: 60..120): chalk-white washed-out flash
    img[40:200, 60:120] = np.clip(np.array([218.0, 215.0, 214.0], np.float32) + pores[40:200, 60:120], 0, 255)
    # Right half of face (x: 120..180): strong red ambient LED cast
    img[40:200, 120:180] = np.clip(np.array([228.0, 92.0, 88.0], np.float32) + pores[40:200, 120:180], 0, 255)
    src = img.astype(np.uint8)

    faces_n = [np.array([60.0 / W, 40.0 / W, 120.0 / W, 160.0 / W], np.float32)]

    # Measure LAB difference between left cheek (y:90..150, x:75..108) and right cheek (y:90..150, x:132..165)
    lab_before = cv2.cvtColor(src.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    left_before = lab_before[90:150, 75:108].mean(axis=(0, 1))
    right_before = lab_before[90:150, 132:165].mean(axis=(0, 1))
    de_before = float(np.linalg.norm(left_before - right_before))
    assert de_before > 40.0, f"Expected large initial split-light ΔE between white and red halves, got {de_before}"

    # 1. Auto Ton AI should automatically detect the mixed-light split face and recommend mixed_light > 0.35
    auto = TN.auto_tone_params(src, faces_n=faces_n)
    assert auto.get("mixed_light", 0.0) > 0.35, f"Expected auto_tone_params to detect split mixed light, got {auto}"

    # 2. Render with mixed_light=0.95 and verify both halves converge to natural skin tone
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(src, None)
    fit.faces_n = faces_n
    p = P.Params(mixed_light=0.95, skin_even=0.5, db=0.0, plastic=0.0)
    out = eng.render(src, fit, p, db_enabled=False)

    lab_after = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    left_after = lab_after[90:150, 75:108].mean(axis=(0, 1))
    right_after = lab_after[90:150, 132:165].mean(axis=(0, 1))
    de_after = float(np.linalg.norm(left_after - right_after))

    # Left-right mismatch must drop by > 65%
    assert de_after < de_before * 0.35, f"Expected split-light ΔE to drop by >65%, got {de_before:.1f} -> {de_after:.1f}"
    # White half must gain warm skin chroma (a* > 5, b* > 8)
    assert left_after[1] > 5.0 and left_after[2] > 8.0, f"Expected white half of face to gain warm skin tone, got LAB={left_after}"
    # Red half must lose extreme redness (a* reduced from >45 down to <24)
    assert right_after[1] < 24.0, f"Expected red half of face to be neutralized to natural skin, got LAB={right_after}"

    # High-frequency pore detail on both cheeks must remain intact
    hp_before = src[90:150, 75:165, 1].astype(np.float32) - cv2.GaussianBlur(src[90:150, 75:165, 1].astype(np.float32), (0, 0), 2.0)
    hp_after = out[90:150, 75:165, 1].astype(np.float32) - cv2.GaussianBlur(out[90:150, 75:165, 1].astype(np.float32), (0, 0), 2.0)
    assert hp_after.std() >= 0.75 * hp_before.std(), "Expected pore micro-texture to be preserved during mixed-light correction"

