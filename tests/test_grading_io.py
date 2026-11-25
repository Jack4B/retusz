import hashlib, os
from pathlib import Path
import numpy as np, cv2, pytest
from PIL import Image
from synthetic import textured, grade, make_ai
from retusz import grading as GR, colorman as CM, imageio, rawio
from retusz.engine import transfer as TR

def _sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def _exif():
    ex = Image.Exif(); ex[0x010F] = "TestCam"; ex[0x0110] = "Model 1"; return ex.tobytes()

@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("pair")
    o, disc = textured(900, 1350, seed=21); gt = grade(o, disc)
    ai, T = make_ai(gt, 3, 1.0, (3, -2), 0.04)
    Image.fromarray(o).save(d/"IMG_7.jpg", quality=97, subsampling=0, exif=_exif(), icc_profile=CM.icc_bytes("srgb"))
    Image.fromarray(ai).save(d/"IMG_7_gemini.png")
    return d, o, gt

def test_run_writes_full_size_jpg_with_exif_icc_and_never_touches_inputs(pair, tmp_path):
    d, o, gt = pair
    h0, h1 = _sha(d/"IMG_7.jpg"), _sha(d/"IMG_7_gemini.png")
    r = GR.run(d/"IMG_7.jpg", d/"IMG_7_gemini.png", tmp_path/"out.jpg", TR.TransferParams())
    im = Image.open(tmp_path/"out.jpg")
    assert im.size == (1350, 900)
    assert im.getexif().get(0x010F) == "TestCam"
    assert im.info.get("icc_profile") == CM.icc_bytes("srgb")
    assert _sha(d/"IMG_7.jpg") == h0 and _sha(d/"IMG_7_gemini.png") == h1
    assert r["align_px"] < 1.0 and 0 <= r["changed"] < 0.05

def test_refuses_to_overwrite_an_input(pair):
    d, o, gt = pair
    with pytest.raises(ValueError):
        GR.run(d/"IMG_7.jpg", d/"IMG_7_gemini.png", d/"IMG_7.jpg", TR.TransferParams())

def test_tiff16_output(pair, tmp_path):
    import tifffile
    d, o, gt = pair
    GR.run(d/"IMG_7.jpg", d/"IMG_7_gemini.png", tmp_path/"out.tif", TR.TransferParams())
    with tifffile.TiffFile(tmp_path/"out.tif") as tf:
        a = tf.asarray(); icc = bytes(tf.pages[0].tags[34675].value)
    assert a.dtype == np.uint16 and a.shape == (900, 1350, 3)
    assert icc == CM.icc_bytes("srgb")

def test_adobe_rgb_original_with_srgb_ai(tmp_path):
    o, disc = textured(600, 900, seed=22); gt = grade(o, disc)
    ai, _ = make_ai(gt, 2, 0.5, (1, 1), 0.03)
    adobe = CM.icc_bytes("adobergb")
    o_ad = CM.convert(o, None, adobe)
    Image.fromarray(o_ad).save(tmp_path/"a.jpg", quality=97, subsampling=0, icc_profile=adobe)   # 4:2:0 would blur the chroma GT was made from
    Image.fromarray(ai).save(tmp_path/"a_ai.jpg", quality=95)            # no profile = sRGB
    GR.run(tmp_path/"a.jpg", tmp_path/"a_ai.jpg", tmp_path/"r.jpg", TR.TransferParams())
    im = Image.open(tmp_path/"r.jpg"); assert im.info.get("icc_profile") == adobe
    res = CM.convert(np.asarray(im.convert("RGB")), adobe, None)
    inside = np.zeros(o.shape[:2], bool); inside[80:-80, 110:-110] = True     # inside the AI frame
    de = TR.delta_e2000(TR.to_lab(res/255.)[inside], TR.to_lab(gt/255.)[inside])
    assert de.mean() < 2.5, de.mean()

def test_pair_files_by_name(tmp_path):
    a, b = tmp_path/"orig", tmp_path/"ai"; a.mkdir(); b.mkdir()
    for n in ["IMG_1.NEF", "IMG_10.jpg", "DSC_5.jpg", "lonely.jpg"]: (a/n).write_bytes(b"x")
    for n in ["IMG_1_ai.png", "IMG_10-gemini.jpg", "dsc_5.png", "other.png"]: (b/n).write_bytes(b"x")
    got = [(p.name, q.name) for p, q in GR.pair_files(a, b)]
    assert got == [("DSC_5.jpg", "dsc_5.png"), ("IMG_1.NEF", "IMG_1_ai.png"), ("IMG_10.jpg", "IMG_10-gemini.jpg")]

def test_pair_files_same_folder(tmp_path):
    for n in ["IMG_1.jpg", "IMG_1_ai.png", "IMG_2.jpg", "IMG_2 (gemini).jpg", "IMG_3.jpg"]: (tmp_path/n).write_bytes(b"x")
    got = [(p.name, q.name) for p, q in GR.pair_files(tmp_path, tmp_path)]
    assert got == [("IMG_1.jpg", "IMG_1_ai.png"), ("IMG_2.jpg", "IMG_2 (gemini).jpg")]

NEF_DIR = Path(os.environ.get("RETUSZ_TEST_NEF", "/workspace/nef"))
SAMPLES = sorted(NEF_DIR.glob("*.NEF")) if NEF_DIR.is_dir() else []

@pytest.mark.skipif(not SAMPLES, reason="no NEF sample (RETUSZ_TEST_NEF)")
def test_nef_load16_matches_8bit_develop():
    p = SAMPLES[0]
    a16, meta = rawio.load16(p); a8, _ = rawio.load(p)
    assert a16.dtype == np.uint16 and a16.shape == a8.shape
    assert abs(float((a16/257.0).mean()) - float(a8.mean())) < 1.0

def test_cli_single_and_batch(pair, tmp_path):
    from retusz import cli
    d, o, gt = pair
    assert cli.transfer_main([str(d/"IMG_7.jpg"), str(d/"IMG_7_gemini.png"), "--out", str(tmp_path/"x.jpg"), "--jasnosc", "70"]) == 0
    assert (tmp_path/"x.jpg").exists()
    assert cli.transfer_main(["--batch", str(d), str(d), "--out", str(tmp_path/"seria")]) == 0
    assert (tmp_path/"seria"/"IMG_7_grading-ai.jpg").exists()
