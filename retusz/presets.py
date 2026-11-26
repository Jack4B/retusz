"""Studio presets (`*.retusz-preset.json`) compatible 1:1 with `retusz-core` (Rust).
Author: Jack4B.

Includes 6 built-in portrait/studio presets and user preset load/save/delete in
`data_dir()/presets/`. Presets apply tonal, colour, white balance, and portrait 3D/D&B
settings while preserving the current photo's crop/straighten geometry and reference weights."""
from __future__ import annotations
from dataclasses import dataclass, field
import json, re
from pathlib import Path
from .paths import data_dir
from .engine.pipeline import Params

PRESET_EXT = ".retusz-preset.json"
PRESET_KEYS = (
    "exposure", "contrast", "brightness", "dehaze", "highlights", "shadows", "whites", "blacks",
    "clarity", "texture", "sharpen_amount", "denoise", "vibrance", "saturation", "vignette", "db", "plastic",
    "wide", "temp", "tint", "skin_even", "skin_cast", "skin_hue",
    "deshine", "blemish", "eyes_teeth",
)


@dataclass
class Preset:
    name: str
    category: str = "Własne"
    builtin: bool = False
    values: dict = field(default_factory=dict)

    def apply_to(self, current: Params) -> Params:
        d = current.to_dict()
        # Reset preset-controlled keys to defaults first so selecting a preset is deterministic
        base = Params().to_dict()
        for k in PRESET_KEYS:
            d[k] = self.values.get(k, base[k])
        return Params.from_dict(d)

    def to_dict(self) -> dict:
        p = Params().to_dict()
        for k in PRESET_KEYS:
            if k in self.values:
                p[k] = float(self.values[k])
        return {
            "name": self.name,
            "category": self.category,
            "builtin": self.builtin,
            "params": p,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Preset:
        raw_p = dict(d.get("params") or d.get("values") or {})
        if "dnb" in raw_p and "db" not in raw_p:
            raw_p["db"] = raw_p["dnb"]
        if "vivid" in raw_p and "wide" not in raw_p:
            raw_p["wide"] = raw_p["vivid"]
        vals = {k: float(raw_p[k]) for k in PRESET_KEYS if k in raw_p}
        return cls(
            name=str(d.get("name") or "Własny preset"),
            category=str(d.get("category") or "Własne"),
            builtin=bool(d.get("builtin", False)),
            values=vals,
        )


def builtin_presets() -> list[Preset]:
    return [
        Preset("Portret naturalny", "Portret", True,
               dict(db=0.40, plastic=0.30)),
        Preset("Ciepły złoty", "Portret", True,
               dict(exposure=0.15, contrast=0.12, highlights=-0.20, shadows=0.22, whites=0.08,
                    temp=0.18, tint=0.04, vibrance=0.18, db=0.45, plastic=0.35, vignette=-0.15)),
        Preset("Filmowy miękki", "Filmowe", True,
               dict(exposure=0.05, contrast=-0.12, highlights=-0.30, shadows=0.28, whites=-0.10,
                    blacks=0.15, clarity=-0.22, vibrance=0.10, saturation=-0.08, db=0.50,
                    plastic=0.25, vignette=-0.20)),
        Preset("High-Key (jasny)", "Studyjne", True,
               dict(exposure=0.45, contrast=-0.08, highlights=-0.25, shadows=0.35, whites=0.15,
                    blacks=0.10, clarity=-0.12, vibrance=0.10, db=0.50, plastic=0.20)),
        Preset("Studyjny kontrast 3D", "Studyjne", True,
               dict(contrast=0.24, highlights=-0.18, shadows=0.15, whites=0.12, blacks=-0.10,
                    clarity=0.25, vibrance=0.15, db=0.55, plastic=0.55, vignette=-0.22)),
        Preset("B&W Klasyczny", "Monochrom", True,
               dict(exposure=0.08, contrast=0.28, highlights=-0.22, shadows=0.18, whites=0.14,
                    blacks=-0.12, clarity=0.20, saturation=-1.0, vibrance=-1.0, db=0.50,
                    plastic=0.45, vignette=-0.25)),
    ]


def presets_dir(custom_dir: Path | None = None) -> Path:
    d = custom_dir if custom_dir is not None else (data_dir() / "presets")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _slug(name: str) -> str:
    s = re.sub(r"[^\w\-ąćęłńóśźżĄĆĘŁŃÓŚŹŻ ]+", "", name.strip()).replace(" ", "_")
    return s or "preset"


def preset_path(name: str, folder: Path | None = None) -> Path:
    return presets_dir(folder) / f"{_slug(name)}{PRESET_EXT}"


def save_preset(name: str, params: Params, category: str = "Własne", folder: Path | None = None) -> Preset:
    d = params.to_dict()
    vals = {k: float(d[k]) for k in PRESET_KEYS if k in d}
    pr = Preset(name=name.strip(), category=category, builtin=False, values=vals)
    p = preset_path(pr.name, folder)
    p.write_text(json.dumps(pr.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return pr


def load_preset_file(path: Path) -> Preset:
    data = json.loads(path.read_text(encoding="utf-8"))
    pr = Preset.from_dict(data)
    pr.builtin = False
    return pr


def export_preset_file(preset: Preset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(preset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def delete_preset(name: str, folder: Path | None = None) -> bool:
    removed = False
    p = preset_path(name, folder)
    if p.exists():
        p.unlink()
        removed = True
    for f in presets_dir(folder).glob(f"*{PRESET_EXT}"):
        try:
            if load_preset_file(f).name == name:
                f.unlink()
                removed = True
        except Exception:
            continue
    return removed


def list_presets(folder: Path | None = None) -> list[Preset]:
    out = builtin_presets()
    seen = {p.name for p in out}
    for f in sorted(presets_dir(folder).glob("*.json")):
        try:
            pr = load_preset_file(f)
            if pr.name not in seen:
                out.append(pr)
                seen.add(pr.name)
        except Exception:
            continue
    return out
