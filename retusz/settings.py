"""App settings (%LOCALAPPDATA%\\Retusz\\settings.json)."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, fields, field
from .paths import data_dir
from .export import ExportOptions

DEVICES = {"auto": "Automatycznie (test szybkości)", "cpu": "Procesor (CPU)", "gpu": "Karta graficzna (DirectML)"}

@dataclass
class Settings:
    jpg_quality: int = 95            # kept for older settings files; export.quality is used
    output_subfolder: str = "Retusz"
    preview_side: int = 1600
    method: str = "ai"
    show_mask: bool = False
    asked_download: bool = False
    last_folder: str = ""
    device: str = "auto"
    export: dict = field(default_factory=dict)

    @classmethod
    def load(cls):
        try:
            d = json.loads((data_dir() / "settings.json").read_text(encoding="utf-8"))
            names = {f.name for f in fields(cls)}; s = cls(**{k: v for k, v in d.items() if k in names})
        except (OSError, ValueError, TypeError):
            s = cls()
        s.jpg_quality = max(60, min(100, int(s.jpg_quality)))
        if s.device not in DEVICES: s.device = "auto"
        if not isinstance(s.export, dict) or not s.export: s.export = {"quality": s.jpg_quality}
        s.export = ExportOptions.from_dict(s.export).to_dict()
        return s

    def export_options(self) -> ExportOptions:
        return ExportOptions.from_dict(self.export)

    def save(self):
        (data_dir() / "settings.json").write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")
