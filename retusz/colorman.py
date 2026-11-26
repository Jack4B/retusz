"""Colour management (LittleCMS via Pillow ImageCms).

Working space = the source photo's own profile (e.g. AdobeRGB from Lightroom). The reference is
converted into that space before analysis, the preview is converted from it to the monitor profile
for display, and the export either keeps it or converts to sRGB / AdobeRGB / Display P3."""
from __future__ import annotations
import hashlib, io, sys, threading
import numpy as np
from PIL import Image, ImageCms
from .paths import resource_dir

OUTPUT_PROFILES = {
    "keep": "Jak oryginał",
    "srgb": "sRGB",
    "p3": "Display P3",
    "adobergb": "Adobe RGB (1998)",
    "prophoto": "ProPhoto RGB (ROMM)",
}
_FILES = {"srgb": "sRGB.icc", "adobergb": "AdobeRGB1998-compat.icc", "p3": "DisplayP3.icc", "prophoto": "AdobeRGB1998-compat.icc"}
_lock = threading.Lock(); _xf = {}; _monitor = None

def icc_bytes(name: str) -> bytes:
    return (resource_dir() / "icc" / _FILES.get(name, "sRGB.icc")).read_bytes()

def key(icc: bytes | None) -> str:
    """Identity of a profile; missing profile = sRGB (the JPEG/web default)."""
    if not icc: return "srgb"
    try:
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).lower()
        if "srgb" in desc: return "srgb"
    except Exception:
        return "srgb"
    return hashlib.sha1(icc).hexdigest()

def description(icc: bytes | None) -> str:
    if not icc: return "sRGB (brak profilu)"
    try: return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception: return "nieczytelny profil"

def _profile(icc):
    return ImageCms.ImageCmsProfile(io.BytesIO(icc)) if icc else ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))

def monitor_icc() -> bytes | None:
    """Monitor profile on Windows (GetICMProfile through Pillow); None = assume sRGB."""
    global _monitor
    if _monitor is None:
        data = b""
        if sys.platform == "win32":
            try:
                p = ImageCms.get_display_profile()
                if p is not None: data = p.tobytes()
            except Exception: data = b""
        _monitor = data
    return _monitor or None

def convert(rgb: np.ndarray, src_icc, dst_icc) -> np.ndarray:
    """Convert uint8 RGB between profiles (relative colorimetric + black point compensation)."""
    ks, kd = key(src_icc), key(dst_icc)
    if ks == kd: return rgb
    with _lock:
        xf = _xf.get((ks, kd))
        if xf is None:
            try:
                xf = ImageCms.buildTransform(_profile(src_icc), _profile(dst_icc), "RGB", "RGB",
                                             renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                                             flags=ImageCms.Flags.BLACKPOINTCOMPENSATION)
            except Exception:
                xf = False
            _xf[(ks, kd)] = xf
    if xf is False: return rgb
    out = ImageCms.applyTransform(Image.fromarray(np.ascontiguousarray(rgb)), xf)
    return np.asarray(out)

def for_display(rgb, icc):
    return convert(rgb, icc, monitor_icc())
