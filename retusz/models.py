"""AI model management: the big model (ModFlows B6 fp16, 119 MB) is downloaded on first run;
the light model (B0, 18 MB) ships inside the app; without either the engine uses IDT (no AI)."""
from __future__ import annotations
import hashlib, os, shutil, urllib.request
from dataclasses import dataclass
from pathlib import Path
from .paths import models_dir, bundled_model

@dataclass(frozen=True)
class ModelInfo:
    key: str; label: str; file: str; size_mb: int; sha256: str; url: str | None

# The download URL is a GitHub Release asset that Jack4B publishes once (see BUILD.md).
# Override with the RETUSZ_MODEL_URL environment variable or "Wskaż plik" in Ustawienia.
DEFAULT_URL = "https://github.com/Jack4B/retusz/releases/download/models-v1/modflows_b6_fp16.onnx"

AI = ModelInfo("ai", "AI (ModFlows B6)", "modflows_b6_fp16.onnx", 119,
               "92b73d4416be5fba85ac68623033fcc2d11e97535afe7c3ea25a13e2c34bdb09", DEFAULT_URL)
LIGHT = ModelInfo("ai_light", "AI lekki (B0)", "modflows_b0_fp32.onnx", 18,
                  "19cadf28524220bf773278c9bb8d4cb4bd7ef99e41fdbcb0b4d2f6eb6fbbedf7", None)
METHODS = {"ai": "AI (najlepsza jakość)", "ai_light": "AI lekki (szybki)", "classic": "Bez AI (IDT)"}

def ai_path() -> Path:
    return models_dir() / AI.file

def have_ai() -> bool:
    return ai_path().exists()

def resolve(method: str):
    """Returns (model_path or None, effective method). Falls back ai -> ai_light -> classic."""
    if method == "ai" and have_ai():
        return ai_path(), "ai"
    if method in ("ai", "ai_light") and bundled_model(LIGHT.file).exists():
        return bundled_model(LIGHT.file), "ai_light"
    return None, "classic"

def sha256(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

class DownloadError(Exception):
    pass

def download(progress=None, cancel=None, url=None) -> Path:
    """Download the AI model with progress(done_bytes, total_bytes); verifies SHA-256."""
    url = url or os.environ.get("RETUSZ_MODEL_URL") or AI.url
    dst = ai_path(); part = dst.with_suffix(".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Retusz"})
        with urllib.request.urlopen(req, timeout=30) as r, open(part, "wb") as f:
            total = int(r.headers.get("Content-Length") or AI.size_mb*1_000_000); done = 0
            while True:
                if cancel and cancel(): raise DownloadError("Przerwano pobieranie.")
                b = r.read(1 << 20)
                if not b: break
                f.write(b); done += len(b)
                if progress: progress(done, total)
    except DownloadError:
        part.unlink(missing_ok=True); raise
    except Exception as e:
        part.unlink(missing_ok=True); raise DownloadError(f"Nie udało się pobrać modelu: {e}") from e
    return install(part, move=True)

def install(path, move=False) -> Path:
    """Install a model file chosen by the user (or a finished download) after checking SHA-256."""
    path = Path(path)
    if sha256(path) != AI.sha256:
        if move: path.unlink(missing_ok=True)
        raise DownloadError("Plik modelu ma niepoprawną sumę kontrolną (SHA-256).")
    dst = ai_path()
    if path.resolve() == dst.resolve(): return dst
    (shutil.move if move else shutil.copyfile)(str(path), str(dst))
    return dst
