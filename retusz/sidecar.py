"""Series settings and per-photo overrides.

  <folder>/retusz-seria.json     reference path + series defaults
  <photo file name>.retusz.json  this photo's own controls and brush strokes (override)

A photo without a sidecar follows the series. Changing one photo never touches the others."""
from __future__ import annotations
import json, os
from pathlib import Path
from .engine.pipeline import Params

SERIES_FILE = "retusz-seria.json"
SIDECAR_SUFFIX = ".retusz.json"
VERSION = 1

def sidecar_path(photo) -> Path:
    photo = Path(photo); return photo.with_name(photo.name + SIDECAR_SUFFIX)

def _write_json(path: Path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)

def _read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

def _per_photo_only(p: Params) -> Params:
    """Brush strokes, straighten and crop always belong to one photo, never to the series."""
    p.strokes = []; p.angle = 0.0; p.crop = None
    return p

class Series:
    def __init__(self, folder):
        self.folder = Path(folder); self.reference: Path | None = None; self.params = Params()
        d = _read_json(self.folder / SERIES_FILE)
        if d:
            ref = d.get("reference")
            if ref:
                rp = Path(ref); rp = rp if rp.is_absolute() else self.folder / rp
                self.reference = rp
            self.params = Params.from_dict(d.get("params"))
            _per_photo_only(self.params)

    def save(self):
        ref = None
        if self.reference:
            try: ref = str(Path(self.reference).resolve().relative_to(self.folder.resolve()))
            except ValueError: ref = str(Path(self.reference).resolve())
        p = _per_photo_only(Params.from_dict(self.params.to_dict())).to_dict()
        _write_json(self.folder / SERIES_FILE, {"version": VERSION, "app": "Retusz", "reference": ref, "params": p})

    def has_override(self, photo) -> bool:
        return sidecar_path(photo).exists()

    def params_for(self, photo) -> Params:
        d = _read_json(sidecar_path(photo))
        if d and "params" in d:
            return Params.from_dict(d["params"])
        return Params.from_dict(self.params.to_dict())

    def set_override(self, photo, params: Params):
        _write_json(sidecar_path(photo), {"version": VERSION, "app": "Retusz", "photo": Path(photo).name, "params": params.to_dict()})

    def clear_override(self, photo):
        try: sidecar_path(photo).unlink()
        except FileNotFoundError: pass

    def sync_fields(self, photos, values: dict):
        """Copy some controls (e.g. white balance) to the series defaults and to every photo that has
        its own settings; photos without a sidecar follow the series anyway."""
        for k, v in values.items(): setattr(self.params, k, v)
        self.params = Params.from_dict(self.params.to_dict()); self.save()
        for ph in photos:
            if self.has_override(ph):
                p = self.params_for(ph)
                for k, v in values.items(): setattr(p, k, v)
                self.set_override(ph, Params.from_dict(p.to_dict()))
