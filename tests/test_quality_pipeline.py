"""TDD tests for v0.4.0 Studio Quality & 16-bit Precision Pipeline:
1. Chromaticity-preserving RAW highlight shoulder + multi-vendor RAW extensions.
2. Soft-knee highlight protection in spatial gain rendering (Plastyka 3D / brush / clarity never flat-clip pores).
3. Gamma-correct linear-light downsizing in export.resize_long_edge.
4. Halo-free clamped output sharpening at high-contrast edges (glasses/eyelashes).
5. 16-bit TIFF (LZW + ICC) and PNG export alongside 8-bit JPEG."""
from __future__ import annotations
import numpy as np
import tifffile
from PIL import Image
from retusz import rawio, export as EX, colorman as CM, batch
from retusz.engine import render as R, color as C
from retusz.engine.pipeline import Engine, Params


def test_raw_extensions_and_chromaticity_preserving_shoulder():
    for ext in (".nef", ".nrw", ".cr2", ".cr3", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef"):
        assert rawio.is_raw(f"DSC_0001{ext}")
        assert rawio.is_raw(f"DSC_0001{ext.upper()}")

    # Bright warm skin highlight in linear light: R=1.15 (above knee=0.80), G=0.72, B=0.54 (ratio R/G = 1.597, G/B = 1.333)
    lin = np.array([[[1.15, 0.72, 0.54]]], dtype=np.float32)
    out = rawio.shoulder_rgb(lin)
    assert out.shape == (1, 1, 3)
    assert float(out.max()) <= 1.0
    # True chromaticity-preserving shoulder keeps (R - B)/(G - B) hue ratio within 5%
    hue_in = (lin[0, 0, 0] - lin[0, 0, 2]) / (lin[0, 0, 1] - lin[0, 0, 2])
    hue_out = (out[0, 0, 0] - out[0, 0, 2]) / (out[0, 0, 1] - out[0, 0, 2])
    assert abs(float(hue_out - hue_in) / float(hue_in)) < 0.05


def test_render_spatial_gain_soft_knee_protects_highlight_pore_texture():
    # Create a bright forehead patch (sRGB ~ 232 +- 8 pore texture) and apply +0.30 spatial gain
    rng = np.random.default_rng(42)
    pores = rng.integers(-8, 9, size=(64, 64, 1), dtype=np.int32)
    base = np.array([[[236, 212, 198]]], dtype=np.int32)
    src = np.clip(base + pores, 0, 255).astype(np.uint8)
    lut = C.rgb_grid(33)
    mask_q = np.zeros((16, 16), dtype=np.float32)
    gain_q = np.full((16, 16, 3), 0.30, dtype=np.float32)

    out_numba = R.render(src, lut, lut, mask_q, gain_q)
    out_np = R.render_numpy(src, lut, lut, mask_q, gain_q)
    # Without soft-knee, 236 * 1.30 = 306.8 -> 100% of red channel pixels clip to 255 (std == 0)!
    # With soft-knee highlight protection, pore texture in the red channel remains unclipped (std > 1.2)
    assert float(out_numba[..., 0].astype(np.float32).std()) > 1.2
    assert float(out_np[..., 0].astype(np.float32).std()) > 1.2
    assert np.abs(out_numba.astype(np.float32) - out_np.astype(np.float32)).mean() < 1.0


def test_linear_light_downsizing_preserves_specular_photon_energy():
    # Checkerboard of bright catchlights (250) and dark iris (20):
    # Linear mean of (srgb_to_lin(250/255) + srgb_to_lin(20/255))/2 is ~0.480 -> sRGB ~185.
    # Naive sRGB uint8 averaging gives (250+20)/2 = 135 (much too dark!).
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[0::2, 0::2] = 250
    img[1::2, 1::2] = 250
    img[0::2, 1::2] = 20
    img[1::2, 0::2] = 20
    down = EX.resize_long_edge(img, 32)
    assert down.shape == (32, 32, 3)
    assert float(down.mean()) > 165.0, f"Expected gamma-correct linear downsizing > 165, got {down.mean():.1f}"


def test_halo_free_sharpen_limits_edge_overshoot():
    # Sharp step edge (dark glasses frame = 35 next to cheek skin = 190)
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[:, :40] = [35, 32, 30]
    img[:, 40:] = [190, 152, 134]
    sh = EX.sharpen(img, radius=1.5, amount=1.8, threshold=0.0)
    # Clamped halo-free USM limits bright-side overshoot above 190 to at most +6 levels (unclamped USM overshoots by > +35!)
    overshoot = int(sh[:, 40:, 0].max()) - 190
    assert overshoot <= 8, f"Bright halo overshoot too large: +{overshoot}"


def test_export_16bit_tiff_and_png_with_icc(tmp_path):
    from tests.synthetic import portrait_scene
    rgb, _ = portrait_scene(h=480, w=360, seed=3)
    src_p = tmp_path / "portret.jpg"
    Image.fromarray(rgb).save(src_p, "JPEG", quality=96)

    eng = Engine(None, "classic", depth_ai=False)
    p = Params(exposure=0.2, contrast=0.15, db=0.0, plastic=0.0)

    # 1. 16-bit TIFF export
    opts_tif = EX.ExportOptions(format="tiff16", profile="p3", sharpen="screen")
    dst_tif = batch.export_one(eng, None, src_p, p, tmp_path / "out_tif", opts_tif)
    assert dst_tif.suffix == ".tif" and dst_tif.exists()
    arr16 = tifffile.imread(dst_tif)
    assert arr16.dtype == np.uint16 and arr16.shape == (480, 360, 3)
    assert int(arr16.max()) > 10000

    # 2. PNG export
    opts_png = EX.ExportOptions(format="png", profile="srgb", sharpen="off")
    dst_png = batch.export_one(eng, None, src_p, p, tmp_path / "out_png", opts_png)
    assert dst_png.suffix == ".png" and dst_png.exists()
    im_png = Image.open(dst_png)
    assert im_png.size == (360, 480)
    assert im_png.info.get("icc_profile") is not None
