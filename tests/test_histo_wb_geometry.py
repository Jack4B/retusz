"""0.1.3: histogram/clipping, white balance (temp/tint + eyedropper), crop & straighten."""
import json
import numpy as np, cv2, pytest
from conftest import synth
from retusz.engine import histo as H, wb as WB, geometry as G, color as C
from retusz.engine.pipeline import Engine, Params

# ---------- histogram ----------
def test_histogram_normalised_and_clipping_fractions():
    img = np.full((100, 100, 3), 128, np.uint8)
    img[:10] = 255; img[10:15] = 0
    h = H.histogram(img)
    for k in "rgbl":
        assert h[k].shape == (256,) and abs(h[k].sum() - 1) < 1e-6
    lo, hi = H.clipping(img)[:2]
    assert abs(hi - 0.10) < 1e-6 and abs(lo - 0.05) < 1e-6

def test_highlight_is_any_channel_shadow_is_all_channels():
    img = np.full((10, 10, 3), 100, np.uint8)
    img[0, :, 0] = 255                    # one channel clipped = highlight clip
    img[1, :] = (0, 0, 40)                # only two channels at 0 = not a shadow clip
    img[2, :] = 0
    lo, hi, mlo, mhi = H.clipping(img)
    assert mhi[0].all() and not mhi[1:].any()
    assert mlo[2].all() and not mlo[1].any()

def test_overlay_paints_only_clipped():
    img = np.full((10, 10, 3), 100, np.uint8); img[0] = 255; img[9] = 0
    o = H.overlay(img)
    assert tuple(o[0, 0]) == H.HI_COLOR and tuple(o[9, 0]) == H.LO_COLOR
    assert (o[1:9] == img[1:9]).all()

# ---------- white balance ----------
def test_gains_identity_and_direction():
    assert np.allclose(WB.gains(0, 0), 1)
    g = WB.gains(0.5, 0)
    assert g[0] > 1 > g[2]                                   # warmer
    assert abs(float(WB.Y709 @ g) - 1) < 1e-6                 # neutrals keep their luminance
    assert WB.gains(0, 0.5)[1] < 1                            # +tint = magenta (less green)

def test_eyedropper_neutralises_a_cast_patch():
    grey = np.full((7, 7, 3), 150, np.uint8)
    lin = WB.to_lin(grey)*WB.gains(-0.3, 0.2)                 # bluish-magenta cast
    cast = WB.to_srgb8(lin)
    d = WB.solve(cast)
    assert d is not None
    fixed = WB.to_srgb8(WB.to_lin(cast)*WB.gains(*d))
    lab = C.to_lab(fixed).reshape(-1, 3).mean(0)
    assert abs(lab[1]) < 1.0 and abs(lab[2]) < 1.0

def test_eyedropper_refuses_clipped_or_black():
    assert WB.solve(np.full((7, 7, 3), (255, 200, 180), np.uint8)) is None
    assert WB.solve(np.full((7, 7, 3), 5, np.uint8)) is None

@pytest.fixture(scope="module")
def classic():
    e = Engine(None, "classic", depth_ai=False); src, ref = synth(seed=1), synth(seed=2, tint=(1, .75, .55))
    return e, src, e.fit(src, e.analyse_reference(ref))

def test_render_temp_warms_and_neutral_default(classic):
    e, src, fit = classic
    base = C.to_lab(e.render(src, fit, Params())).reshape(-1, 3).mean(0)
    warm = C.to_lab(e.render(src, fit, Params(temp=0.4))).reshape(-1, 3).mean(0)
    assert warm[2] > base[2] + 3                              # more yellow
    assert abs(warm[0] - base[0]) < 2.5                       # brightness kept

def test_render_eyedropper_round_trip(classic):
    e, src, fit = classic
    p = Params(temp=0.25, tint=-0.2)
    out = e.render(src, fit, p)
    y, x = 120, 180; patch = out[y-3:y+4, x-3:x+4]
    d = WB.solve(patch); assert d is not None
    p2 = Params(temp=p.temp + d[0], tint=p.tint + d[1])
    out2 = e.render(src, fit, p2)
    lab = C.to_lab(out2[y-3:y+4, x-3:x+4]).reshape(-1, 3).mean(0)
    assert np.hypot(lab[1], lab[2]) < 2.0

def test_params_wb_geometry_clamped_and_in_colour_key():
    p = Params.from_dict({"temp": 3, "tint": -5, "angle": 90, "crop": [0.9, 0.1, 0.2, 0.8], "aspect": "zzz"})
    assert p.temp == 1 and p.tint == -1 and p.angle == 45 and p.aspect == "orig"
    x0, y0, x1, y1 = p.crop; assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1
    assert Params(temp=0.1).colour_key() != Params().colour_key()
    assert Params(tint=0.1).colour_key() != Params().colour_key()
    json.dumps(p.to_dict())

# ---------- geometry ----------
def test_identity_and_exact_crop_slice():
    img = synth(120, 180)
    assert (G.apply(img, 0, None) == img).all()
    out = G.apply(img, 0, [0.25, 0.5, 0.75, 1.0])
    assert (out == img[60:120, 45:135]).all()

def test_straighten_has_no_empty_corners():
    img = np.full((200, 300, 3), 200, np.uint8)
    out = G.apply(img, 7, None)
    assert out.shape == img.shape and out.min() >= 199

def test_point_maps_round_trip():
    for ang in (0, 3.5, -12, 40):
        for p in ((0.1, 0.2), (0.5, 0.5), (0.93, 0.7)):
            q = G.frame_to_src(G.src_to_frame(p, ang, 300, 200), ang, 300, 200)
            assert np.allclose(q, p, atol=1e-9)

def test_frame_to_src_matches_pixels():
    img = np.zeros((200, 300, 3), np.uint8); img[48:53, 198:203] = 255     # dot at (200, 50)
    ang = 9; out = G.apply(img, ang, None)
    ys, xs = np.nonzero(out[..., 0] > 100); fx, fy = xs.mean()/300, ys.mean()/200
    sx, sy = G.frame_to_src((fx, fy), ang, 300, 200)
    assert abs(sx*300 - 200) < 1.5 and abs(sy*200 - 50) < 1.5

def test_horizon_tool_levels_the_line():
    a, b = (40, 60), (260, 95)                                 # pixel coords, sloping down to the right
    d = G.horizon_angle(a, b)
    M = cv2.getRotationMatrix2D((150, 100), d, 1.0)
    pa, pb = M @ np.array([*a, 1.0]), M @ np.array([*b, 1.0])
    assert abs(pa[1] - pb[1]) < 1e-6
    assert abs(G.horizon_angle((50, 10), (60, 200))) < 45    # near-vertical line = vertical reference

def test_aspect_rect_ratio_and_inside():
    for key, ratio in (("1:1", 1.0), ("2:3", 2/3), ("4:5", 4/5), ("16:9", 16/9)):
        for portrait in (True, False):
            r = G.aspect_rect(key, 6000, 4000, portrait)
            w, h = (r[2]-r[0])*6000, (r[3]-r[1])*4000
            want = ratio if ratio == 1 else (min(ratio, 1/ratio) if portrait else max(ratio, 1/ratio))
            assert abs(w/h - want) < 1/h*2 + 1e-3
            assert 0 <= r[0] < r[2] <= 1 and 0 <= r[1] < r[3] <= 1
            assert abs((r[0]+r[2])/2 - 0.5) < 1e-9
    assert G.aspect_rect("free", 6000, 4000, True) is None

def test_fit_aspect_keeps_anchor_and_ratio():
    r = G.fit_aspect([0.2, 0.2, 0.9, 0.5], 2/3, 3000, 2000, anchor=(0.2, 0.2))
    w, h = (r[2]-r[0])*3000, (r[3]-r[1])*2000
    assert abs(w/h - 2/3) < 0.01 and r[0] == 0.2 and r[1] == 0.2 and r[3] <= 1

def test_clamp_crop_order_and_min_size():
    c = G.clamp_crop([0.8, 0.9, 0.1, 0.9])
    assert c[0] < c[2] and c[1] < c[3] and c[3]-c[1] >= G.MIN_CROP - 1e-9
    assert G.clamp_crop(None) is None and G.clamp_crop([0, 0, 1, 1]) is None
