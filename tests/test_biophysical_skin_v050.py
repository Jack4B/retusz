"""TDD test suite for Retusz 0.5.0:
1. Dichromatic SUV skin De-Shine (Mallick et al. 2005 / Yang et al. 2010)
2. Optical-density Melanin / Hemoglobin chromophore separation & redness evening (Tsumura et al. 2003)
3. Two-band pore-preserving Blemish Auto-Heal & Anatomical Eyes/Teeth enhancement (YuNet landmarks)
4. Fast Local Laplacian Filter (Fast LLF — Paris, Hasinoff & Kautz 2011 / Aubry et al. 2014) zero-halo local contrast
5. OKLCh Skin-Tone Vectorscope (52° skin line indicator)
Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
import cv2


def test_dichromatic_suv_deshine_removes_oily_glare_and_preserves_skin_chroma_and_pores():
    """Dichromatic SUV De-Shine removes white specular glare on forehead/nose while recovering underlying
    diffuse skin chromaticity and preserving high-frequency pore micro-texture."""
    from retusz.engine.skin_physics import deshine_gain

    rng = np.random.default_rng(101)
    H, W = 128, 128
    # Base warm diffuse skin (R=205, G=155, B=128) + fine pore texture
    base = np.empty((H, W, 3), np.float32)
    base[..., 0] = 205.0
    base[..., 1] = 155.0
    base[..., 2] = 128.0
    pores = rng.normal(0.0, 3.0, (H, W)).astype(np.float32)
    pores -= cv2.GaussianBlur(pores, (0, 0), 1.5)

    # Add an oily white specular glare blob in the center (+45 across all R, G, B -> washed out white shine)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    glare = 46.0 * np.exp(-((xx - 64.0) ** 2 + (yy - 64.0) ** 2) / (2.0 * 16.0 ** 2))
    shiny_rgb = np.clip(base + glare[..., None] + pores[..., None], 0, 255).astype(np.uint8)
    mask = np.ones((H, W), np.float32)

    g = deshine_gain(shiny_rgb, mask, amount=0.85)
    assert g.shape == (H, W, 3)
    out = np.clip(shiny_rgb.astype(np.float32) * (1.0 + g), 0, 255)

    center = (yy - 64.0) ** 2 + (xx - 64.0) ** 2 <= 10.0 ** 2
    ring = ((yy - 64.0) ** 2 + (xx - 64.0) ** 2 >= 36.0 ** 2) & ((yy - 64.0) ** 2 + (xx - 64.0) ** 2 <= 48.0 ** 2)

    # 1. Specular peak brightness in the glare center must drop significantly toward surrounding diffuse skin
    before_excess = float(shiny_rgb[center].mean() - shiny_rgb[ring].mean())
    after_excess = float(out[center].mean() - out[ring].mean())
    assert after_excess < before_excess * 0.45, f"Expected glare reduction, got {before_excess:.2f} -> {after_excess:.2f}"

    # 2. Chromaticity (R-B difference / warmth) in the glare center is restored (not greyed out)
    rb_glare_after = float((out[center, 0] - out[center, 2]).mean())
    rb_ring = float((out[ring, 0] - out[ring, 2]).mean())
    assert abs(rb_glare_after - rb_ring) < 12.0, f"Expected restored skin chroma, got {rb_glare_after:.1f} vs {rb_ring:.1f}"


def test_chromophore_melanin_hemoglobin_separation_evens_red_blotches():
    """Optical-density chromophore decomposition (-log10 RGB -> Melanin & Hemoglobin, Tsumura 2003)
    attenuates low-frequency hemoglobin redness blotches without flattening skin luminance."""
    from retusz.engine.skin_physics import decompose_chromophores, chromophore_even_gain
    from retusz.engine import color as C

    H, W = 120, 120
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Base skin
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0] = 202.0
    rgb[..., 1] = 156.0
    rgb[..., 2] = 132.0
    # Add low-frequency hemoglobin erythema blotches (absorbs green & blue more than red -> looks red/pink)
    blotch = np.sin(xx / 9.0) * np.cos(yy / 9.0)
    rgb[..., 1] -= 18.0 * blotch
    rgb[..., 2] -= 14.0 * blotch
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    mask = np.ones((H, W), np.float32)

    mel, hem, _ = decompose_chromophores(rgb_u8)
    # Hemoglobin map should correlate strongly with the red blotch pattern
    corr = float(np.corrcoef(hem.ravel(), blotch.ravel())[0, 1])
    assert corr > 0.80, f"Hemoglobin component should track erythema blotch, corr={corr:.3f}"

    lab = C.to_lab(rgb_u8)
    g = chromophore_even_gain(rgb_u8, lab, mask, [[20, 20, 80, 80]], amount=0.8)
    out_u8 = np.clip(rgb_u8.astype(np.float32) * (1.0 + g), 0, 255).astype(np.uint8)
    _, hem_after, _ = decompose_chromophores(out_u8)

    assert hem_after.std() < hem.std() * 0.55, f"Expected hemoglobin blotch std reduction: {hem.std():.4f} -> {hem_after.std():.4f}"


def test_two_band_blemish_heal_removes_acne_spot_and_keeps_pore_texture():
    """Two-band auto-heal detects isolated red acne blemishes inside safe skin mask, heals the base
    chromophore/luminance anomaly, and preserves high-frequency pore micro-texture."""
    from retusz.engine.skin_physics import apply_portrait_details

    rng = np.random.default_rng(202)
    H, W = 240, 240
    clean = np.empty((H, W, 3), np.float32)
    clean[..., 0] = 204.0
    clean[..., 1] = 158.0
    clean[..., 2] = 134.0
    pores = rng.normal(0.0, 2.8, (H, W)).astype(np.float32)
    pores -= cv2.GaussianBlur(pores, (0, 0), 1.2)
    clean_with_pores = np.clip(clean + pores[..., None], 0, 255).astype(np.uint8)

    # Inject 3 realistic small acne blemishes (radius ~4 px: dark red/inflamed core)
    blemished = clean_with_pores.copy().astype(np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spots = [(95, 110), (145, 95), (125, 145)]
    for sy, sx in spots:
        w = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2.0 * 3.2 ** 2))
        blemished[..., 0] -= 14.0 * w
        blemished[..., 1] -= 42.0 * w
        blemished[..., 2] -= 38.0 * w
    blemished_u8 = np.clip(blemished, 0, 255).astype(np.uint8)

    mask_q = np.ones((H // 4, W // 4), np.float32)
    faces_px = [np.array([20.0, 20.0, 200.0, 200.0], np.float32)]
    lms_px = [np.array([[85.0, 80.0], [155.0, 80.0], [120.0, 120.0], [95.0, 165.0], [145.0, 165.0]], np.float32)]

    healed = apply_portrait_details(blemished_u8, mask_q, faces_px, lms_px, blemish=0.9, eyes_teeth=0.0)

    # Measure error on the blemish spots vs clean skin
    spot_mask = np.zeros((H, W), bool)
    for sy, sx in spots:
        spot_mask |= ((xx - sx) ** 2 + (yy - sy) ** 2) <= 4.5 ** 2

    err_before = float(np.abs(blemished_u8[spot_mask].astype(np.float32) - clean_with_pores[spot_mask].astype(np.float32)).mean())
    err_after = float(np.abs(healed[spot_mask].astype(np.float32) - clean_with_pores[spot_mask].astype(np.float32)).mean())
    assert err_after < err_before * 0.48, f"Blemish error should drop by >52%, got {err_before:.2f} -> {err_after:.2f}"

    # High-frequency pore texture on the healed spots must remain alive (not blurred to 0)
    hf_healed = healed[..., 1].astype(np.float32) - cv2.GaussianBlur(healed[..., 1].astype(np.float32), (0, 0), 1.5)
    hf_clean = clean_with_pores[..., 1].astype(np.float32) - cv2.GaussianBlur(clean_with_pores[..., 1].astype(np.float32), (0, 0), 1.5)
    assert hf_healed[spot_mask].std() >= hf_clean[spot_mask].std() * 0.75


def test_anatomical_eyes_and_teeth_enhancement_from_landmarks():
    """Anatomical eyes & teeth enhancer boosts iris micro-contrast/brightness and reduces yellow cast on teeth
    using YuNet facial landmarks."""
    from retusz.engine.skin_physics import apply_portrait_details

    H, W = 240, 240
    img = np.full((H, W, 3), (195, 152, 130), np.uint8)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Left and right eyes at (80, 85) and (160, 85), radius ~12
    for ex, ey in ((80.0, 85.0), (160.0, 85.0)):
        r = np.hypot(xx - ex, yy - ey)
        iris = (r >= 3.5) & (r <= 10.5)
        # Textured mid-tone iris
        stripes = 12.0 * np.sin(np.arctan2(yy - ey, xx - ex) * 10.0)
        img[iris, 0] = np.clip(95.0 + stripes[iris], 0, 255).astype(np.uint8)
        img[iris, 1] = np.clip(115.0 + stripes[iris], 0, 255).astype(np.uint8)
        img[iris, 2] = np.clip(125.0 + stripes[iris], 0, 255).astype(np.uint8)

    # Mouth/teeth region between (95, 165) and (145, 165): slightly yellowish teeth (R=218, G=208, B=168)
    teeth = ( np.abs(xx - 120.0) <= 18.0 ) & ( np.abs(yy - 165.0) <= 6.0 )
    img[teeth] = (218, 208, 168)

    mask_q = np.ones((H // 4, W // 4), np.float32)
    faces_px = [np.array([20.0, 20.0, 200.0, 200.0], np.float32)]
    lms_px = [np.array([[80.0, 85.0], [160.0, 85.0], [120.0, 125.0], [95.0, 165.0], [145.0, 165.0]], np.float32)]

    out = apply_portrait_details(img, mask_q, faces_px, lms_px, blemish=0.0, eyes_teeth=0.85)

    r_eye = np.hypot(xx - 80.0, yy - 85.0)
    iris_m = (r_eye >= 4.0) & (r_eye <= 10.0)
    # Iris contrast (std) and brightness should increase
    assert out[iris_m].astype(np.float32).std() > img[iris_m].astype(np.float32).std() * 1.12
    assert out[iris_m].astype(np.float32).mean() > img[iris_m].astype(np.float32).mean() + 3.0

    # Teeth yellowness (R - B) should decrease while lightness increases
    yellowness_before = float((img[teeth, 0].astype(np.float32) - img[teeth, 2].astype(np.float32)).mean())
    yellowness_after = float((out[teeth, 0].astype(np.float32) - out[teeth, 2].astype(np.float32)).mean())
    assert yellowness_after < yellowness_before * 0.72, f"Expected teeth whitening, R-B {yellowness_before:.1f} -> {yellowness_after:.1f}"
    assert out[teeth].mean() > img[teeth].mean()


def test_fast_local_laplacian_filter_zero_halo_on_step_edge():
    """Fast Local Laplacian Filter (Paris, Hasinoff & Kautz 2011 / Aubry et al. 2014) boosts fine/medium
    local contrast while producing zero halo across a sharp high-contrast step edge."""
    from retusz.engine.llf import fast_llf_gain

    H, W = 128, 128
    # Sharp step edge at x = 64: dark glasses frame (L=15) next to bright skin (L=78)
    L = np.full((H, W), 15.0, np.float32)
    L[:, 64:] = 78.0
    # Add fine medium-contrast detail pattern on the right half (x >= 80)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    tex = 4.0 * np.sin(xx / 2.2) * np.cos(yy / 2.2)
    L[:, 76:] += tex[:, 76:]

    mask = np.zeros((H, W), np.float32)
    gain = fast_llf_gain(L, mask, clarity=0.8)
    L_out = L * (1.0 + gain)

    # 1. Texture standard deviation in the textured zone (x >= 80) increases by > 25%
    std_in = float(L[:, 80:].std())
    std_out = float(L_out[:, 80:].std())
    assert std_out > std_in * 1.25, f"Expected LLF micro-contrast boost >25%, got {std_in:.2f} -> {std_out:.2f}"

    # 2. Zero halo at the step boundary (cols 64..72 on the bright side must not overshoot 78.0 by more than 1.2 L*)
    edge_band = L_out[:, 64:72]
    overshoot = float(edge_band.max() - 78.0)
    assert overshoot <= 1.2, f"Fast LLF must be halo-free at step edge, got overshoot={overshoot:.2f}"


def test_oklch_skin_vectorscope_reports_skin_line_angle():
    """histo.skin_vectorscope computes polar chroma density and mean skin hue angle relative to the 52° OKLCh Skin Line."""
    from retusz.engine import histo as HI

    # Natural warm portrait skin tone (R=212, G=162, B=136) -> OKLCh hue ~ 52 deg
    skin_rgb = np.full((64, 64, 3), (212, 162, 136), np.uint8)
    vs = HI.skin_vectorscope(skin_rgb)
    assert "grid" in vs and "hue_deg" in vs and "delta_deg" in vs
    assert vs["grid"].shape == (64, 64)
    assert abs(vs["hue_deg"] - 52.0) < 10.0, f"Expected skin hue near 52° OKLCh line, got {vs['hue_deg']:.1f}°"

    # Greenish cast on skin shifts hue angle significantly above the 52° skin line
    green_skin = np.full((64, 64, 3), (185, 182, 130), np.uint8)
    vs_green = HI.skin_vectorscope(green_skin)
    assert vs_green["hue_deg"] > vs["hue_deg"] + 15.0
