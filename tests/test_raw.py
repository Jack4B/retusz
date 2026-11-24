"""0.1.3: Nikon NEF input via rawpy/LibRaw. Real-file tests use public CC0 samples from raw.pixls.us
(not in the repo): set RETUSZ_TEST_NEF to a folder with them, default /workspace/nef."""
import os
from pathlib import Path
import numpy as np, cv2, pytest
from PIL import Image
from retusz import imageio, rawio, batch, colorman as CM
from retusz.export import ExportOptions
from retusz.reference import Reference
from retusz.engine import color as C
from retusz.engine.pipeline import Engine, Params
from conftest import synth

NEF_DIR = Path(os.environ.get("RETUSZ_TEST_NEF", "/workspace/nef"))
SAMPLES = sorted(NEF_DIR.glob("*.NEF")) if NEF_DIR.is_dir() else []
need_nef = pytest.mark.skipif(not SAMPLES, reason="no NEF sample (RETUSZ_TEST_NEF)")

def test_list_images_includes_nef_and_hides_its_jpg_twin(tmp_path):
    for n in ["DSC_1.NEF", "DSC_1.JPG", "DSC_2.jpg", "DSC_3.nef", "x.txt"]: (tmp_path/n).write_bytes(b"x")
    assert [p.name for p in imageio.list_images(tmp_path)] == ["DSC_1.NEF", "DSC_2.jpg", "DSC_3.nef"]
    assert rawio.is_raw("a.NEF") and rawio.is_raw("b.nrw") and not rawio.is_raw("c.jpg")

def test_orient_matches_libraw_flip_codes():
    a = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert (rawio.orient(a, 0) == a).all()
    assert (rawio.orient(a, 3) == a[::-1, ::-1]).all()
    assert rawio.orient(a, 5).shape == (3, 2) and rawio.orient(a, 5)[0, 0] == a[0, 2]     # 90° CCW
    assert rawio.orient(a, 6).shape == (3, 2) and rawio.orient(a, 6)[0, 0] == a[1, 0]     # 90° CW

def test_tone_is_monotone_and_shoulder_bounded():
    x = np.linspace(0, 8, 2000, dtype=np.float32)
    y = rawio.shoulder(x)
    assert np.all(np.diff(y) >= 0) and y.max() <= 1 and np.allclose(y[x <= rawio.KNEE], x[x <= rawio.KNEE])

def test_quantise_dither_keeps_mean():
    v = np.full((64, 64, 3), 100.3/255, np.float32)
    q = rawio.quantise(v)
    assert q.dtype == np.uint8 and abs(q.mean() - 100.3) < 0.05

def test_develop_table_matches_float_path():
    a16 = (np.random.default_rng(0).random((40, 50, 3))**2*65535).astype(np.uint16)
    want = rawio.encode(rawio.shoulder(a16.astype(np.float32)*np.float32(1.7/65535)))*255
    got = rawio.develop(a16, 1.7).astype(np.float32)
    assert np.abs(got - want).max() <= 1.0 and abs((got - want).mean()) < 0.05

@need_nef
@pytest.mark.parametrize("nef", SAMPLES, ids=lambda p: p.name)
def test_nef_full_load_srgb_exif_and_size(nef):
    rgb, meta = imageio.load(nef)
    assert rgb.dtype == np.uint8 and rgb.ndim == 3
    assert imageio.image_size(nef) == (rgb.shape[1], rgb.shape[0])
    assert CM.key(meta["icc_profile"]) == "srgb"
    ex = Image.Exif(); ex.load(meta["exif"])
    assert "NIKON" in ex.get(0x0110, "").upper() and ex.get(0x0112) == 1
    assert ex.get_ifd(0x8769).get(0x9003)                     # DateTimeOriginal
    assert 273 not in ex and 0x927C not in ex.get_ifd(0x8769) # no strip offsets / maker notes
    # exposure starts where the camera's own JPEG was (the D750 sample is a night shot)
    th = imageio.load_preview(nef, 300); small = cv2.resize(rgb, (th.shape[1], th.shape[0]), interpolation=cv2.INTER_AREA)
    assert abs(C.to_lab(small)[..., 0].mean() - C.to_lab(th)[..., 0].mean()) < 6

@need_nef
@pytest.mark.parametrize("nef", SAMPLES, ids=lambda p: p.name)
def test_nef_preview_matches_full_decode(nef):
    full, _ = imageio.load(nef)
    prev, icc = imageio.load_preview(nef, 1600, with_icc=True)
    assert max(prev.shape[:2]) == 1600 and CM.key(icc) == "srgb"
    ref = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    dE = np.linalg.norm(C.to_lab(prev) - C.to_lab(ref), axis=-1)
    assert dE.mean() < 1.0, dE.mean()

@need_nef
@pytest.mark.parametrize("nef", SAMPLES, ids=lambda p: p.name)
def test_nef_preview_is_downscale_of_ahd_export(nef):
    """Regression (0.1.3, his noisy D700): a PPG preview against an AHD export drifted (ΔE 1.35–1.53).
    The 1600 px preview must be a downscale of the same AHD develop as the export, not a faster demosaic."""
    full, _ = imageio.load(nef)
    prev = imageio.load_preview(nef, 1600)
    ahd = cv2.resize(full, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    dE_ahd = float(np.linalg.norm(C.to_lab(prev) - C.to_lab(ahd), axis=-1).mean())
    a16 = rawio._decode16(nef, fast=True)
    ppg = cv2.resize(rawio.develop(a16, rawio.exposure_gain(nef, a16)), (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA)
    dE_ppg = float(np.linalg.norm(C.to_lab(prev) - C.to_lab(ppg), axis=-1).mean())
    assert dE_ahd < 0.15, dE_ahd
    if dE_ppg > 0.2:
        assert dE_ahd < dE_ppg / 2, (dE_ahd, dE_ppg)

@need_nef
def test_nef_thumbnail_is_fast_embedded_and_upright():
    nef = SAMPLES[0]
    th = imageio.load_preview(nef, 240)
    w, h = imageio.image_size(nef)
    assert max(th.shape[:2]) == 240 and (th.shape[1] > th.shape[0]) == (w > h)

@need_nef
def test_nef_batch_export_writes_jpg_with_exif(tmp_path):
    nef = SAMPLES[0]; ref = tmp_path/"ref.png"; Image.fromarray(synth(200, 300, seed=3, tint=(1, .85, .7))).save(ref)
    e = Engine(None, "classic", depth_ai=False)
    out = batch.export_one(e, Reference(ref), nef, Params(), tmp_path/"Retusz", ExportOptions(long_edge=1200))
    im = Image.open(out); assert out.suffix == ".jpg" and max(im.size) == 1200
    assert "NIKON" in im.getexif().get(0x0110, "").upper()
    assert nef.stat().st_size > 1e6                            # original untouched
