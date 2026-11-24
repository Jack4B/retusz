"""JPEG/PNG/TIFF input; JPG output (quality 95 by default) keeping EXIF and the ICC profile."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
ORIENTATION = 0x0112

def list_images(folder) -> list[Path]:
    folder = Path(folder)
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS and not p.name.startswith(".")),
                  key=lambda p: p.name.lower())

def load(path):
    """Returns (rgb uint8 HxWx3, meta) with EXIF orientation applied; meta keeps icc + exif bytes."""
    im = Image.open(path); im.load()
    icc = im.info.get("icc_profile"); exif_raw = im.info.get("exif")
    exif = im.getexif(); orient = exif.get(ORIENTATION, 1)
    if orient not in (0, 1):
        im = ImageOps.exif_transpose(im)
        exif[ORIENTATION] = 1; exif_raw = exif.tobytes()
    rgb = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(rgb), {"icc_profile": icc, "exif": exif_raw}

def load_preview(path, side, with_icc=False):
    """Fast downscaled load (JPEG draft mode) for thumbnails and previews."""
    im = Image.open(path); icc = im.info.get("icc_profile")
    try: im.draft("RGB", (side*2, side*2))   # DCT scaling only down to >= 2x the target, then a real filter
    except Exception: pass
    im = ImageOps.exif_transpose(im).convert("RGB"); im.thumbnail((side, side), Image.LANCZOS)
    a = np.asarray(im)
    return (a, icc) if with_icc else a

def image_size(path):
    with Image.open(path) as im:
        w, h = im.size
        return (h, w) if im.getexif().get(ORIENTATION, 1) in (5, 6, 7, 8) else (w, h)

def save_jpg(path, rgb, meta=None, quality=95):
    meta = meta or {}; path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(quality=int(quality), subsampling=0, optimize=False)
    if meta.get("exif"): kw["exif"] = meta["exif"]
    if meta.get("icc_profile"): kw["icc_profile"] = meta["icc_profile"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(rgb).save(tmp, "JPEG", **kw)
    tmp.replace(path)
    return path
