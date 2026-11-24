"""The series reference, analysed once per working colour space (a reference in sRGB used with
AdobeRGB photos is first converted to AdobeRGB, so colours are compared in the same space)."""
from __future__ import annotations
import threading
from pathlib import Path
from . import imageio, colorman as CM

class Reference:
    def __init__(self, path):
        self.path = Path(path)
        self.rgb, meta = imageio.load(self.path); self.icc = meta.get("icc_profile")
        self._cache = {}; self._lock = threading.Lock()

    def analysis(self, engine, src_icc=None):
        k = (CM.key(src_icc), id(engine), engine.model, engine.depth_ai)
        with self._lock:
            ra = self._cache.get(k)
        if ra is None:
            ra = engine.analyse_reference(CM.convert(self.rgb, self.icc, src_icc))
            with self._lock: self._cache[k] = ra
        return ra
