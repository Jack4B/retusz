import io
import numpy as np, pytest
from PIL import Image, ImageCms
from retusz import imageio, models, batch
from retusz.export import ExportOptions
from retusz.reference import Reference
from retusz.engine.pipeline import Engine, Params
from retusz.sidecar import Series
from conftest import synth

def _jpeg_with_meta(path, rgb, orientation=1):
    ex = Image.Exif(); ex[0x010F] = "TestCam"; ex[0x0112] = orientation
    ex.get_ifd(0x8769)[0x9003] = "2026:11:24 12:00:00"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.fromarray(rgb).save(path, quality=92, exif=ex.tobytes(), icc_profile=icc); return icc

def test_save_keeps_exif_and_icc(tmp_path):
    src = tmp_path/"a.jpg"; icc = _jpeg_with_meta(src, synth())
    rgb, meta = imageio.load(src); out = imageio.save_jpg(tmp_path/"out"/"a.jpg", rgb, meta, 95)
    im = Image.open(out); ex = im.getexif()
    assert im.info.get("icc_profile") == icc and ex[0x010F] == "TestCam"
    assert ex.get_ifd(0x8769)[0x9003] == "2026:11:24 12:00:00"
    assert im.size == (rgb.shape[1], rgb.shape[0])

def test_orientation_applied_and_reset(tmp_path):
    src = tmp_path/"r.jpg"; _jpeg_with_meta(src, synth(100, 200), orientation=6)
    rgb, meta = imageio.load(src); assert rgb.shape[:2] == (200, 100)
    out = imageio.save_jpg(tmp_path/"o.jpg", rgb, meta); assert Image.open(out).getexif().get(0x0112) == 1

def test_list_images_filters(tmp_path):
    for n in ["b.JPG", "a.jpg", "c.txt", ".hidden.jpg", "d.png"]: (tmp_path/n).write_bytes(b"x")
    assert [p.name for p in imageio.list_images(tmp_path)] == ["a.jpg", "b.JPG", "d.png"]

def test_model_resolve_fallbacks():
    if models.have_ai(): models.ai_path().unlink()
    p, m = models.resolve("ai"); assert m == "ai_light" and p.name == models.LIGHT.file
    assert models.resolve("classic") == (None, "classic")

def test_install_rejects_wrong_checksum(tmp_path):
    f = tmp_path/"fake.onnx"; f.write_bytes(b"not a model")
    with pytest.raises(models.DownloadError): models.install(f)
    assert not models.have_ai()

def test_download_failure_is_clean(tmp_path):
    with pytest.raises(models.DownloadError): models.download(url="http://127.0.0.1:9/none.onnx")
    assert not models.have_ai() and not models.ai_path().with_suffix(".part").exists()

def test_batch_export_with_override(tmp_path):
    for i in range(3): _jpeg_with_meta(tmp_path/f"IMG_{i}.jpg", synth(seed=i))
    ref = tmp_path/"ref.jpg"; Image.fromarray(synth(seed=9, tint=(1, .7, .5))).save(ref)
    s = Series(tmp_path); s.reference = ref; s.save()
    photos = [tmp_path/f"IMG_{i}.jpg" for i in range(3)]
    s.set_override(photos[1], Params(strength=0.0, vignette=0, db=0, plastic=0))
    e = Engine(None, "classic", depth_ai=False)
    seen = []; res = batch.export_series(e, Reference(ref), s, photos, tmp_path/"Retusz", ExportOptions(sharpen="off"),
                                         progress=lambda *a: seen.append(a[0]))
    assert seen == [1, 2, 3] and all(r[1] and r[1].exists() and r[2] is None for r in res)
    o1 = np.asarray(Image.open(res[1][1])); s1 = imageio.load(photos[1])[0]
    o0 = np.asarray(Image.open(res[0][1])); s0 = imageio.load(photos[0])[0]
    assert np.abs(o1.astype(int)-s1).mean() < 3          # override: strength 0
    assert np.abs(o0.astype(int)-s0).mean() > 8          # series default: transferred
    assert Image.open(res[0][1]).getexif()[0x010F] == "TestCam"

def test_batch_never_overwrites_source(tmp_path):
    p = tmp_path/"x.jpg"; Image.fromarray(synth()).save(p)
    r = tmp_path/"ref.png"; Image.fromarray(synth(seed=5)).save(r)
    e = Engine(None, "classic", depth_ai=False)
    res = batch.export_series(e, Reference(r), Series(tmp_path), [p], tmp_path, ExportOptions())
    assert res[0][1] is None and "oryginał" in res[0][2]
