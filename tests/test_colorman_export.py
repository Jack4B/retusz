"""Colour management (preview == export), advanced export options, new adjustments, device choice."""
import io
import numpy as np, cv2, pytest
from PIL import Image, ImageCms
from retusz import colorman as CM, export as EX, imageio, batch
from retusz.reference import Reference
from retusz.engine.pipeline import Engine, Params, brush_map, skin_even_gain
from retusz.engine import color as C, device
from retusz.sidecar import Series
from conftest import synth

ADOBE = CM.icc_bytes("adobergb")

def _to_adobe(rgb):
    return CM.convert(rgb, None, ADOBE)

def test_profile_keys():
    assert CM.key(None) == "srgb" and CM.key(CM.icc_bytes("srgb")) == "srgb"
    assert CM.key(ADOBE) != "srgb" and CM.key(ADOBE) == CM.key(bytes(ADOBE))

def test_adobe_roundtrip_and_saturation():
    s = synth(); a = _to_adobe(s)
    # AdobeRGB numbers look desaturated if shown raw; converting back restores sRGB within rounding
    back = CM.convert(a, ADOBE, None)
    assert np.abs(back.astype(int)-s).mean() < 1.5
    sat = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].mean()
    assert sat(a) < sat(s) - 5

def test_preview_display_equals_export_viewed(tmp_path, monkeypatch):
    """The preview (display transform of the working image) must equal what a colour-managed viewer
    shows for the exported file, also when the export converts to another profile."""
    monkeypatch.setattr(CM, "monitor_icc", lambda: None)            # sRGB monitor
    src = tmp_path/"a.jpg"; s_adobe = _to_adobe(synth())
    Image.fromarray(s_adobe).save(src, quality=98, subsampling=0, icc_profile=ADOBE)
    rgb, icc = imageio.load_preview(src, 400, with_icc=True); assert icc == ADOBE
    preview = CM.for_display(rgb, icc)
    for prof in ("keep", "srgb", "p3"):
        opts = EX.ExportOptions(profile=prof, sharpen="off", quality=100)
        rgb_full, meta = imageio.load(src)
        out = EX.save(tmp_path/f"o_{prof}.jpg", rgb_full, meta, opts)
        im = Image.open(out); emb = im.info.get("icc_profile")
        assert emb is not None
        if prof == "keep": assert emb == ADOBE
        viewed = CM.for_display(np.asarray(im.convert("RGB")), emb)
        assert np.abs(viewed.astype(int)-preview.astype(int)).mean() < 2.5, prof

def test_reference_converted_to_source_space(tmp_path):
    """Same picture saved as sRGB and as AdobeRGB must be analysed alike once converted."""
    e = Engine(None, "classic", depth_ai=False)
    srgb = synth(seed=3); p1 = tmp_path/"r1.png"; Image.fromarray(srgb).save(p1)
    p2 = tmp_path/"r2.png"; Image.fromarray(_to_adobe(srgb)).save(p2, icc_profile=ADOBE)
    a1 = Reference(p1).analysis(e, None).fit_rgb; a2 = Reference(p2).analysis(e, None).fit_rgb
    assert np.abs(a1.astype(int)-a2.astype(int)).mean() < 2
    ref = Reference(p1); x = ref.analysis(e, ADOBE); assert ref.analysis(e, ADOBE) is x   # cached per profile

def test_export_options(tmp_path):
    src = tmp_path/"s.jpg"; ex = Image.Exif(); ex[0x010F] = "Cam"
    gps = ex.get_ifd(0x8825); gps[2] = (52.0, 13.0, 0.0)
    Image.fromarray(synth(300, 450)).save(src, exif=ex.tobytes())
    rgb, meta = imageio.load(src)
    o = EX.ExportOptions(long_edge=200, suffix="_web", dpi=72, metadata="no_gps", subsampling="4:2:0", profile="srgb")
    dst = EX.output_path(src, tmp_path/"out", o); assert dst.name == "s_web.jpg"
    EX.save(dst, rgb, meta, o); im = Image.open(dst)
    assert max(im.size) == 200 and round(im.info["dpi"][0]) == 72
    e2 = im.getexif(); assert e2[0x010F] == "Cam" and not e2.get_ifd(0x8825)
    EX.save(tmp_path/"n.jpg", rgb, meta, EX.ExportOptions(metadata="none"))
    assert 0x010F not in Image.open(tmp_path/"n.jpg").getexif()
    assert EX.ExportOptions.from_dict({"quality": 500, "suffix": "a/b:c", "profile": "x"}).to_dict()["quality"] == 100
    assert EX.ExportOptions.from_dict({"suffix": "a/b:c"}).suffix == "abc"

def test_sharpen_less_on_skin():
    img = np.clip(synth(200, 200).astype(int) + np.random.default_rng(0).integers(-20, 20, (200, 200, 1)), 0, 255).astype(np.uint8)
    m = np.zeros((50, 50), np.float32); m[:, 25:] = 1
    out = EX.sharpen(img, 1.0, 1.5, 0, skin_mask=m, skin_protect=1.0)
    d = np.abs(out.astype(int)-img)
    assert d[:, :90].mean() > 0.3 and d[:, 110:].mean() < d[:, :90].mean()*0.2

def test_vibrance_saturation():
    e = Engine(None, "classic", depth_ai=False); src = synth(seed=1); ref = e.analyse_reference(synth(seed=2, tint=(1, .8, .6)))
    fit = e.fit(src, ref); base = e.render(src, fit, Params(db=0, plastic=0, vignette=0))
    sat = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV)[..., 1].astype(float).mean()
    more = e.render(src, fit, Params(db=0, plastic=0, vignette=0, saturation=0.5))
    less = e.render(src, fit, Params(db=0, plastic=0, vignette=0, saturation=-1.0))
    vib = e.render(src, fit, Params(db=0, plastic=0, vignette=0, vibrance=0.8))
    assert sat(more) > sat(base) + 5 and sat(less) < 12 and sat(vib) > sat(base) + 2
    assert Params.from_dict({"saturation": 9}).saturation == 1.0

def test_skin_even_reduces_cast_variation():
    h, w = 120, 120; lab = np.zeros((h, w, 3), np.float32); lab[..., 0] = 60
    lab[..., 1] = 15; lab[..., 2] = 18; lab[:, :60, 1] += 10; lab[:, :60, 2] -= 10     # green/magenta mixed light
    rgb = C.from_lab(lab) if hasattr(C, "from_lab") else cv2.cvtColor(lab.astype(np.float32)/[1, 1, 1], cv2.COLOR_LAB2RGB)
    mask = np.ones((h, w), np.float32)
    g = skin_even_gain(rgb, lab, mask, [(30, 30, 60, 60)], 1.0)
    out = np.clip(rgb.astype(np.float32)*(1+g), 0, 255).astype(np.uint8)
    lab2 = C.to_lab(out)
    before = abs(lab[:, :40, 1].mean()-lab[:, 80:, 1].mean()); after = abs(lab2[:, :40, 1].mean()-lab2[:, 80:, 1].mean())
    assert after < before*0.6

def test_brush_softness():
    st = {"mode": "dodge", "size": 0.1, "strength": 1.0, "points": [[0.2, 0.5], [0.8, 0.5]]}
    hard = brush_map([{**st, "softness": 0.0}], 100, 100); soft = brush_map([{**st, "softness": 1.0}], 100, 100)
    edge = lambda m: m[30:45, 50].mean()
    assert soft.max() > 0.5 and edge(soft) > edge(hard)

def test_device_cpu_mode(tmp_path, monkeypatch):
    from retusz.paths import bundled_model
    monkeypatch.setattr(device, "MODE", "cpu")
    s = device.make_session(bundled_model("face_detection_yunet_2023mar.onnx"), device.zeros_feed)
    assert s.get_providers() == ["CPUExecutionProvider"] and device.summary() == "procesor (CPU)"

def test_wide_look_matches_unmanaged_widegamut_display():
    """'Pełne barwy' 100% = sRGB numbers viewed unmanaged on an AdobeRGB display (within clipping)."""
    from retusz.engine.pipeline import wide_gamut_look
    x = synth(seed=4)
    ref = CM.convert(x, ADOBE, None).astype(int)                      # what the eye gets from the old preview
    got = (wide_gamut_look(x.astype(np.float32)/255)*255+0.5).astype(int)
    inside = (ref > 2).all(-1) & (ref < 253).all(-1)                   # compare where nothing clipped
    assert np.abs(got[inside]-ref[inside]).mean() < 2.5
    gray = np.full((4, 4, 3), 0.5, np.float32); assert np.abs(wide_gamut_look(gray)-0.5).max() < 0.02

def test_wide_param_increases_chroma():
    muted = lambda x: (x.astype(np.float32)*0.5 + 64).astype(np.uint8)      # typical photo colours, inside sRGB
    e = Engine(None, "classic", depth_ai=False); src = muted(synth(seed=1)); fit = e.fit(src, e.analyse_reference(muted(synth(seed=2))))
    b = Params(db=0, plastic=0, vignette=0); w = Params(db=0, plastic=0, vignette=0, wide=1.0)
    ch = lambda x: np.hypot(*C.to_lab(x)[..., 1:].transpose(2, 0, 1)).mean()
    assert ch(e.render(src, fit, w)) > ch(e.render(src, fit, b))*1.1
