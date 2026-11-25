from retusz import presets as P
from retusz.engine.pipeline import Params


def test_builtin_presets_apply_and_preserve_crop_and_reference():
    builtins = P.builtin_presets()
    assert len(builtins) >= 6
    names = [b.name for b in builtins]
    assert "Portret naturalny" in names
    assert "Ciepły złoty" in names
    assert "B&W Klasyczny" in names

    cur = Params(strength=0.72, skin=0.85, angle=4.2, crop=(0.1, 0.15, 0.9, 0.85), aspect="2:3")
    gold = next(b for b in builtins if b.name == "Ciepły złoty")
    out = gold.apply_to(cur)
    assert abs(out.exposure - 0.15) < 1e-6
    assert abs(out.temp - 0.18) < 1e-6
    assert abs(out.strength - 0.72) < 1e-6
    assert abs(out.skin - 0.85) < 1e-6
    assert abs(out.angle - 4.2) < 1e-6
    assert out.crop == [0.1, 0.15, 0.9, 0.85]
    assert out.aspect == "2:3"


def test_custom_preset_save_list_import_export_and_delete(tmp_path):
    p = Params(exposure=0.35, contrast=0.22, clarity=0.18, temp=-0.12, db=0.60, plastic=0.45)
    saved = P.save_preset("Mój styl studyjny", p, folder=tmp_path)
    assert not saved.builtin

    all_p = P.list_presets(folder=tmp_path)
    found = [x for x in all_p if x.name == "Mój styl studyjny"]
    assert len(found) == 1
    applied = found[0].apply_to(Params())
    assert abs(applied.exposure - 0.35) < 1e-6
    assert abs(applied.contrast - 0.22) < 1e-6
    assert abs(applied.clarity - 0.18) < 1e-6
    assert abs(applied.db - 0.60) < 1e-6

    exp_file = tmp_path / "exported.retusz-preset.json"
    P.export_preset_file(found[0], exp_file)
    loaded = P.load_preset_file(exp_file)
    assert loaded.name == "Mój styl studyjny"

    assert P.delete_preset("Mój styl studyjny", folder=tmp_path)
    assert not any(x.name == "Mój styl studyjny" for x in P.list_presets(folder=tmp_path))
