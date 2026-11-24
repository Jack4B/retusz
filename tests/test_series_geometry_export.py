"""0.1.3: WB sync to series, crop/angle per photo, export applies the geometry."""
import numpy as np, cv2
from PIL import Image
from retusz import batch, imageio
from retusz.export import ExportOptions
from retusz.reference import Reference
from retusz.engine import geometry as G
from retusz.engine.pipeline import Engine, Params
from retusz.sidecar import Series
from conftest import synth

def _photos(tmp_path, n=3):
    ps = []
    for i in range(n):
        p = tmp_path/f"p{i}.jpg"; Image.fromarray(synth(120, 180, seed=i)).save(p, quality=95); ps.append(p)
    return ps

def test_sync_wb_updates_series_and_existing_overrides_only(tmp_path):
    ps = _photos(tmp_path); s = Series(tmp_path)
    s.set_override(ps[0], Params(db=0.9, crop=[0.1, 0.1, 0.6, 0.6], angle=3,
                                 strokes=[{"mode": "dodge", "size": .1, "strength": .5, "points": [[.5, .5]]}]))
    s.sync_fields(ps, {"temp": 0.2, "tint": -0.1})
    s2 = Series(tmp_path)
    assert s2.params.temp == 0.2 and s2.params.tint == -0.1
    p0 = s2.params_for(ps[0])
    assert p0.temp == 0.2 and p0.tint == -0.1 and p0.db == 0.9 and p0.angle == 3 and p0.crop and p0.strokes
    assert not s2.has_override(ps[1]) and s2.params_for(ps[1]).temp == 0.2

def test_series_defaults_never_store_geometry(tmp_path):
    s = Series(tmp_path); s.params = Params(angle=5, crop=[0.1, 0.1, 0.5, 0.5], temp=0.1); s.save()
    s2 = Series(tmp_path)
    assert s2.params.angle == 0 and s2.params.crop is None and s2.params.temp == 0.1

def test_export_applies_angle_and_crop(tmp_path):
    ps = _photos(tmp_path, 1); ref = tmp_path/"ref.png"; Image.fromarray(synth(120, 180, seed=7, tint=(1, .8, .6))).save(ref)
    e = Engine(None, "classic", depth_ai=False); R = Reference(ref)
    crop = G.aspect_rect("2:3", 180, 120, True)
    p = Params(angle=5, crop=crop, aspect="2:3")
    opts = ExportOptions(sharpen="off", quality=100, subsampling="4:4:4")
    out = batch.export_one(e, R, ps[0], p, tmp_path/"out", opts)
    got = np.asarray(Image.open(out).convert("RGB")).astype(np.float32)
    rgb, _ = imageio.load(ps[0]); fit = e.fit(rgb, R.analysis(e, None))
    want = G.apply(e.render(rgb, fit, p), p.angle, p.crop).astype(np.float32)
    assert got.shape == want.shape
    h, w = got.shape[:2]; assert abs(w/h - 2/3) < 0.03
    assert np.abs(got - want).mean() < 1.5
