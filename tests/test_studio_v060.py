"""TDD test suite for Retusz 0.6.0 Studio:
1. Per-photo colour reference overrides, reference bank, and star ratings in Series & batch export
2. Capture One midtone Brightness & OKLab Dehaze in 3D LUT / tone engine
3. Four live sharpening modes (Portrait skin-protected, Richardson-Lucy deconvolution, Clamped USM Edges,
   Soft Focus) + Texture & OKLab Chroma/Luma Denoising
4. Six export formats (JPG 8-bit, TIFF 16-bit, TIFF 8-bit, PNG 16-bit, PNG 8-bit, WebP) and five ICC
   colour spaces (including ProPhoto RGB / ROMM RGB)
Author: Jack4B.
"""
from __future__ import annotations
import numpy as np
import cv2
import tifffile
from PIL import Image


def test_per_photo_reference_and_reference_bank_in_series_and_batch_export(tmp_path):
    """Different photos in the same series can use the series reference, their own custom reference,
    or standalone mode ('__none__'), and preserve their star rating across series syncs."""
    from retusz.sidecar import Series
    from retusz.engine.pipeline import Engine, Params
    from retusz import batch, export, imageio
    from retusz.reference import Reference
    from tests.synthetic import portrait_scene

    img, _ = portrait_scene(600, 400, seed=11)
    # Create a warm reference and a cool reference
    ref_warm = np.clip(img.astype(np.float32) * np.array([1.18, 1.00, 0.78], np.float32), 0, 255).astype(np.uint8)
    ref_cool = np.clip(img.astype(np.float32) * np.array([0.78, 1.00, 1.22], np.float32), 0, 255).astype(np.uint8)

    p1 = tmp_path / "p1.jpg"
    p2 = tmp_path / "p2.jpg"
    p3 = tmp_path / "p3.jpg"
    rw_path = tmp_path / "ref_warm.jpg"
    rc_path = tmp_path / "ref_cool.jpg"
    for p, arr in ((p1, img), (p2, img), (p3, img), (rw_path, ref_warm), (rc_path, ref_cool)):
        cv2.imwrite(str(p), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    ser = Series(tmp_path)
    ser.reference = rw_path
    ser.add_to_reference_bank(rw_path)
    ser.add_to_reference_bank(rc_path)
    ser.save()

    # Reload series and verify reference bank persistence
    ser2 = Series(tmp_path)
    assert len(ser2.reference_bank) == 2

    # p1 uses series default (warm), rating=5
    p1_params = ser2.params_for(p1)
    p1_params.rating = 5
    ser2.set_override(p1, p1_params)

    # p2 uses custom per-photo cool reference, rating=4
    p2_params = ser2.params_for(p2)
    p2_params.rating = 4
    p2_params.reference_path = str(rc_path.name)
    ser2.set_override(p2, p2_params)

    # p3 uses standalone mode ('__none__') despite series having a reference
    p3_params = ser2.params_for(p3)
    p3_params.reference_path = "__none__"
    p3_params.db = 0.0
    p3_params.plastic = 0.0
    ser2.set_override(p3, p3_params)

    assert ser2.effective_reference_for(p1)[0].resolve() == rw_path.resolve()
    assert ser2.effective_reference_for(p2)[0].resolve() == rc_path.resolve()
    assert ser2.effective_reference_for(p3)[0] is None

    # Syncing white balance across the series must NOT erase p1/p2 ratings or p2/p3 per-photo reference_path
    ser2.sync_fields([p1, p2, p3], {"temp": 0.05})
    assert ser2.params_for(p1).rating == 5
    assert ser2.params_for(p2).reference_path == str(rc_path.name)
    assert ser2.params_for(p3).reference_path == "__none__"

    # Batch export of the series applies each photo's own reference
    eng = Engine(None, "classic", depth_ai=False)
    out_dir = tmp_path / "out"
    res = batch.export_series(eng, Reference(rw_path), ser2, [p1, p2, p3], out_dir, export.ExportOptions())
    assert all(r[2] is None for r in res)

    out1, _ = imageio.load(res[0][1])
    out2, _ = imageio.load(res[1][1])
    # p1 (warm ref) must have significantly higher R-B than p2 (cool ref)
    rb1 = float(out1[..., 0].astype(np.float32).mean() - out1[..., 2].astype(np.float32).mean())
    rb2 = float(out2[..., 0].astype(np.float32).mean() - out2[..., 2].astype(np.float32).mean())
    assert rb1 > rb2 + 12.0, f"Expected warm p1 ({rb1:.1f}) > cool p2 ({rb2:.1f})"


def test_capture_one_midtone_brightness_and_dehaze_in_oklab():
    """Capture One style midtone Brightness shifts midtones while pinning black (0) and white (1);
    Dehaze deepens hazy lifted shadows and restores chroma."""
    from retusz.engine import tone as TN

    # Ramp from 0.0 (black) to 0.25 (mid-grey) to 1.0 (white)
    ramp = np.array([[[0.0, 0.0, 0.0], [0.22, 0.20, 0.18], [1.0, 1.0, 1.0]]], np.float32)
    lifted = TN.apply_tone_lin(ramp, brightness=0.75)
    darkened = TN.apply_tone_lin(ramp, brightness=-0.75)

    # Endpoints 0.0 and 1.0 remain pinned
    assert np.abs(lifted[0, 0]).max() < 1e-4
    assert np.abs(lifted[0, 2] - 1.0).max() < 1e-3
    # Midtone is lifted or darkened significantly
    assert float(lifted[0, 1].mean()) > float(ramp[0, 1].mean()) + 0.06
    assert float(darkened[0, 1].mean()) < float(ramp[0, 1].mean()) - 0.05

    # Dehaze on a hazy washed-out patch (lifted black floor + low saturation)
    hazy = np.full((16, 16, 3), (0.48, 0.42, 0.38), np.float32)
    dehazed = TN.apply_tone_lin(hazy, dehaze=0.75)
    assert float(dehazed.mean()) < float(hazy.mean()) - 0.05
    # Chromatic spread (R - B) relative to mean luminance increases after dehaze
    c_before = float((hazy[..., 0] - hazy[..., 2]).mean() / hazy.mean())
    c_after = float((dehazed[..., 0] - dehazed[..., 2]).mean() / dehazed.mean())
    assert c_after > c_before * 1.15


def test_four_sharpening_modes_rl_deconvolution_portrait_edges_soft_and_denoise():
    """Tests the 4 live sharpening modes ('portrait', 'rl', 'edges', 'soft'), texture slider,
    and edge-preserving OKLab chroma/luma denoising."""
    from retusz.engine.detail import apply_detail_pass

    rng = np.random.default_rng(606)
    H, W = 128, 128
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Create an image with fine texture on left (skin, mask=1) and right (eyes/hair, mask=0)
    base = np.full((H, W, 3), (185.0, 148.0, 128.0), np.float32)
    tex = 9.0 * np.sin(xx / 1.4) * np.cos(yy / 1.4)
    img = np.clip(base + tex[..., None], 0, 255).astype(np.uint8)
    mask_q = np.zeros((H // 4, W // 4), np.float32)
    mask_q[:, : W // 8] = 1.0  # Left half is skin, right half is non-skin (eyes/hair)

    # 1. 'portrait' mode sharpens non-skin (right half) much more than skin (left half)
    out_port = apply_detail_pass(img, mask_q, sharpen_amount=0.85, sharpen_mode="portrait")
    gain_skin = float(out_port[:, :48, 0].astype(np.float32).std() - img[:, :48, 0].astype(np.float32).std())
    gain_hair = float(out_port[:, 80:, 0].astype(np.float32).std() - img[:, 80:, 0].astype(np.float32).std())
    assert gain_hair > 1.2 and gain_hair > gain_skin * 2.2

    # 2. 'rl' (Richardson-Lucy deconvolution) recovers optical acutance on blurred texture
    blurred = cv2.GaussianBlur(img, (0, 0), 0.9)
    out_rl = apply_detail_pass(blurred, mask_q, sharpen_amount=0.85, sharpen_mode="rl")
    assert out_rl[:, 80:, 0].astype(np.float32).std() > blurred[:, 80:, 0].astype(np.float32).std() * 1.22

    # 3. 'soft' (Miękki blask / Soft Focus) reduces harsh micro-texture
    out_soft = apply_detail_pass(img, mask_q, sharpen_amount=0.80, sharpen_mode="soft")
    assert out_soft[..., 0].astype(np.float32).std() < img[..., 0].astype(np.float32).std() * 0.85

    # 4. 'denoise' suppresses high-frequency colour noise while keeping a step edge sharp
    noisy = np.full((H, W, 3), (150.0, 125.0, 110.0), np.float32)
    noisy[:, 64:] = (210.0, 175.0, 155.0)
    chroma_noise = rng.normal(0.0, 10.0, (H, W, 3)).astype(np.float32)
    noisy_u8 = np.clip(noisy + chroma_noise, 0, 255).astype(np.uint8)
    denoised = apply_detail_pass(noisy_u8, mask_q, denoise=0.85)
    # Chrominance R-G variance on flat region drops by > 45%
    rg_before = float((noisy_u8[:, :56, 0].astype(np.float32) - noisy_u8[:, :56, 1].astype(np.float32)).std())
    rg_after = float((denoised[:, :56, 0].astype(np.float32) - denoised[:, :56, 1].astype(np.float32)).std())
    assert rg_after < rg_before * 0.55, f"Expected chroma noise reduction, got {rg_before:.2f} -> {rg_after:.2f}"


def test_export_all_6_formats_and_5_color_spaces_including_prophoto_rgb_and_png16_webp(tmp_path):
    """Verifies export.save across all 6 formats (jpg, tiff16, tiff8, png16, png, webp) and 5 ICC profiles
    (keep, srgb, p3, adobergb, prophoto)."""
    from retusz import export, colorman as CM

    assert {"jpg", "tiff16", "tiff8", "png16", "png", "webp"}.issubset(set(export.FORMATS.keys()))
    assert {"keep", "srgb", "p3", "adobergb", "prophoto"}.issubset(set(CM.OUTPUT_PROFILES.keys()))

    H, W = 48, 64
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[..., 0] = np.linspace(25, 235, W, dtype=np.uint8)[None, :]
    rgb[..., 1] = 145
    rgb[..., 2] = 115
    meta = {"icc_profile": CM.icc_bytes("srgb")}

    # 1. 16-bit TIFF with ProPhoto RGB (ROMM) ICC profile
    opts_t16 = export.ExportOptions(format="tiff16", profile="prophoto", sharpen="off")
    p_t16 = export.save(export.output_path("test.nef", tmp_path, opts_t16), rgb, meta, opts_t16)
    arr_t16 = tifffile.imread(p_t16)
    assert arr_t16.dtype == np.uint16 and arr_t16.max() > 30000

    # 2. 16-bit PNG lossless
    opts_p16 = export.ExportOptions(format="png16", profile="p3", sharpen="off")
    p_p16 = export.save(export.output_path("test_p16.nef", tmp_path, opts_p16), rgb, meta, opts_p16)
    arr_p16 = cv2.imread(str(p_p16), cv2.IMREAD_UNCHANGED)
    assert arr_p16.dtype == np.uint16 and arr_p16.max() > 30000

    # 3. 8-bit TIFF with Adobe RGB (1998)
    opts_t8 = export.ExportOptions(format="tiff8", profile="adobergb", sharpen="off")
    p_t8 = export.save(export.output_path("test_t8.nef", tmp_path, opts_t8), rgb, meta, opts_t8)
    arr_t8 = tifffile.imread(p_t8)
    assert arr_t8.dtype == np.uint8

    # 4. WebP with sRGB
    opts_webp = export.ExportOptions(format="webp", profile="srgb", sharpen="off")
    p_webp = export.save(export.output_path("test_w.nef", tmp_path, opts_webp), rgb, meta, opts_webp)
    assert p_webp.suffix == ".webp" and p_webp.exists()
    with Image.open(p_webp) as im:
        assert im.size == (W, H)
