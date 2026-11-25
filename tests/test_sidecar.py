import json
from pathlib import Path
from retusz.sidecar import Series, sidecar_path, SERIES_FILE
from retusz.engine.pipeline import Params

def _photos(tmp_path, n=3):
    ps = []
    for i in range(n):
        p = tmp_path/f"IMG_{i}.jpg"; p.write_bytes(b"x"); ps.append(p)
    return ps

def test_override_affects_only_that_photo(tmp_path):
    a, b, c = _photos(tmp_path); s = Series(tmp_path)
    s.params = Params(strength=0.8); s.save()
    s.set_override(b, Params(strength=0.3, db=0.9, strokes=[{"mode": "burn", "size": 0.02, "strength": 0.5, "points": [[0.1, 0.1]]}]))
    s2 = Series(tmp_path)
    assert s2.params_for(a).strength == 0.8 and s2.params_for(c).strength == 0.8
    pb = s2.params_for(b); assert pb.strength == 0.3 and pb.db == 0.9 and len(pb.strokes) == 1
    assert s2.has_override(b) and not s2.has_override(a)
    assert sidecar_path(b).name == "IMG_1.jpg.retusz.json"

def test_series_change_does_not_touch_overrides(tmp_path):
    a, b = _photos(tmp_path, 2); s = Series(tmp_path)
    s.set_override(b, Params(vignette=0.1)); s.params = Params(vignette=0.9); s.save()
    s2 = Series(tmp_path); assert s2.params_for(a).vignette == 0.9 and s2.params_for(b).vignette == 0.1

def test_clear_override_returns_to_series(tmp_path):
    a, = _photos(tmp_path, 1); s = Series(tmp_path)
    s.set_override(a, Params(strength=0.1)); s.clear_override(a); s.clear_override(a)
    assert not s.has_override(a) and s.params_for(a).strength == Params().strength

def test_reference_stored_relative_and_strokes_never_in_series(tmp_path):
    ref = tmp_path/"wzorzec.jpg"; ref.write_bytes(b"x"); s = Series(tmp_path); s.reference = ref
    s.params = Params(strokes=[{"mode": "dodge", "points": [[0, 0]]}]); s.save()
    d = json.loads((tmp_path/SERIES_FILE).read_text(encoding="utf-8"))
    assert d["reference"] == "wzorzec.jpg" and d["params"]["strokes"] == []
    assert Series(tmp_path).reference == tmp_path/"wzorzec.jpg"

def test_corrupt_sidecar_falls_back_to_series(tmp_path):
    a, = _photos(tmp_path, 1); sidecar_path(a).write_text("{not json", encoding="utf-8")
    assert Series(tmp_path).params_for(a) == Params()

def test_light_angle_roundtrip(tmp_path):
    a, = _photos(tmp_path, 1); s = Series(tmp_path)
    s.set_override(a, Params(light_angle=135.0, plastic=0.9))
    p = Series(tmp_path).params_for(a); assert p.light_angle == 135.0 and p.plastic == 0.9

def test_tone_params_roundtrip(tmp_path):
    a, = _photos(tmp_path, 1); s = Series(tmp_path)
    s.set_override(a, Params(exposure=0.85, contrast=0.35, whites=0.2, blacks=-0.25, clarity=-0.4))
    p = Series(tmp_path).params_for(a)
    assert p.exposure == 0.85 and p.contrast == 0.35 and p.whites == 0.2 and p.blacks == -0.25 and p.clarity == -0.4

