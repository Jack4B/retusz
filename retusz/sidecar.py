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
    """Brush strokes, straighten, crop, star rating, and per-photo reference override always belong to one photo."""
    p.strokes = []; p.angle = 0.0; p.crop = None; p.rating = 0; p.reference_path = None; p.ref_colorchecker = False
    return p

class Series:
    def __init__(self, folder):
        self.folder = Path(folder); self.reference: Path | None = None; self.params = Params()
        self.ref_colorchecker: bool = False
        self.reference_bank: list[Path] = []
        d = _read_json(self.folder / SERIES_FILE)
        if d:
            ref = d.get("reference")
            if ref:
                rp = Path(ref); rp = rp if rp.is_absolute() else self.folder / rp
                self.reference = rp
            self.ref_colorchecker = bool(d.get("ref_colorchecker", False))
            for item in d.get("reference_bank") or []:
                bp = Path(item); bp = bp if bp.is_absolute() else self.folder / bp
                if bp not in self.reference_bank:
                    self.reference_bank.append(bp)
            if self.reference and self.reference not in self.reference_bank:
                self.reference_bank.insert(0, self.reference)
            self.params = Params.from_dict(d.get("params"))
            _per_photo_only(self.params)

    def _rel_or_abs(self, path: Path) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.folder.resolve()))
        except ValueError:
            return str(Path(path).resolve())

    def add_to_reference_bank(self, path: Path) -> Path:
        p = Path(path)
        p = p if p.is_absolute() else (self.folder / p)
        if all(b.resolve() != p.resolve() for b in self.reference_bank):
            self.reference_bank.append(p)
        return p

    def effective_reference_for(self, photo, params: Params | None = None) -> tuple[Path | None, bool]:
        """Resolve the effective reference (path, colorchecker_flag) for a specific photo:
        - params.reference_path == '__none__' -> (None, False) (standalone edit for this photo)
        - params.reference_path set -> (resolved_path, params.ref_colorchecker)
        - otherwise -> (self.reference, self.ref_colorchecker)"""
        p = params if params is not None else self.params_for(photo)
        rp = getattr(p, "reference_path", None)
        if rp == "__none__":
            return None, False
        if rp:
            path = Path(rp)
            if not path.is_absolute():
                path = self.folder / path
            if path.exists():
                return path, bool(getattr(p, "ref_colorchecker", False))
        return self.reference, self.ref_colorchecker

    def save(self):
        ref = self._rel_or_abs(self.reference) if self.reference else None
        if self.reference:
            self.add_to_reference_bank(self.reference)
        bank = [self._rel_or_abs(b) for b in self.reference_bank]
        p = _per_photo_only(Params.from_dict(self.params.to_dict())).to_dict()
        _write_json(self.folder / SERIES_FILE, {
            "version": VERSION, "app": "Retusz", "reference": ref,
            "ref_colorchecker": self.ref_colorchecker, "reference_bank": bank, "params": p,
        })

    def has_override(self, photo) -> bool:
        return sidecar_path(photo).exists()

    def params_for(self, photo) -> Params:
        d = _read_json(sidecar_path(photo))
        if d and "params" in d:
            return Params.from_dict(d["params"])
        return Params.from_dict(self.params.to_dict())

    def set_override(self, photo, params: Params):
        if params.reference_path and params.reference_path != "__none__":
            rp = Path(params.reference_path)
            rp = rp if rp.is_absolute() else (self.folder / rp)
            if rp.exists():
                self.add_to_reference_bank(rp)
                params.reference_path = self._rel_or_abs(rp)
        _write_json(sidecar_path(photo), {"version": VERSION, "app": "Retusz", "photo": Path(photo).name, "params": params.to_dict()})

    def clear_override(self, photo):
        try: sidecar_path(photo).unlink()
        except FileNotFoundError: pass

    def sync_fields(self, photos, values: dict):
        """Copy some controls (e.g. white balance) to the series defaults and to every photo that has
        its own settings; photos without a sidecar follow the series anyway. Per-photo rating, geometry,
        and per-photo reference_path are preserved unless explicitly included in values."""
        for k, v in values.items():
            if k not in ("strokes", "angle", "crop", "rating", "reference_path", "ref_colorchecker"):
                setattr(self.params, k, v)
        self.params = Params.from_dict(self.params.to_dict()); self.save()
        for ph in photos:
            if self.has_override(ph):
                p = self.params_for(ph)
                for k, v in values.items(): setattr(p, k, v)
                self.set_override(ph, Params.from_dict(p.to_dict()))
