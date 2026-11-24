"""Resource and per-user data locations (works from source and from a PyInstaller bundle)."""
import os, sys
from pathlib import Path

def resource_dir() -> Path:
    base = getattr(sys, "_MEIPASS", None)
    return Path(base) / "retusz" / "assets" if base else Path(__file__).resolve().parent / "assets"

def bundled_model(name: str) -> Path:
    return resource_dir() / "models" / name

def data_dir() -> Path:
    env = os.environ.get("RETUSZ_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "Retusz"
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "retusz"
    p.mkdir(parents=True, exist_ok=True)
    return p

def models_dir() -> Path:
    p = data_dir() / "models"; p.mkdir(parents=True, exist_ok=True); return p
