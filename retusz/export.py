"""Advanced export: long-edge resize, output sharpening (skin-aware), colour profile conversion,
JPG quality/chroma subsampling, DPI, file-name suffix, metadata keep/strip-GPS/strip-all."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import io
import numpy as np, cv2
from PIL import Image
from . import colorman as CM

SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
METADATA = {"all": "Wszystkie", "no_gps": "Bez lokalizacji GPS", "none": "Usuń wszystkie"}
SHARPEN = {"off": "Wył.", "screen": "Ekran", "print": "Druk", "custom": "Własne"}
SHARPEN_PRESETS = {"screen": (0.6, 0.6, 2.0), "print": (1.0, 1.2, 3.0)}   # radius px, amount, threshold (0..255)

@dataclass
class ExportOptions:
    quality: int = 95
    subsampling: str = "4:4:4"
    profile: str = "keep"
    dpi: int = 300
    long_edge: int = 0              # 0 = original size
    suffix: str = ""
    metadata: str = "all"
    sharpen: str = "screen"
    radius: float = 0.8
    amount: float = 0.8
    threshold: float = 2.0
    sharpen_skin: float = 0.6       # how much less sharpening on skin (0 = same, 1 = none)

    def to_dict(self): return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}; o = cls(**{k: v for k, v in (d or {}).items() if k in names})
        o.quality = int(np.clip(o.quality, 50, 100)); o.dpi = int(np.clip(o.dpi, 0, 2400)); o.long_edge = max(0, int(o.long_edge))
        if o.subsampling not in SUBSAMPLING: o.subsampling = "4:4:4"
        if o.profile not in CM.OUTPUT_PROFILES: o.profile = "keep"
        if o.metadata not in METADATA: o.metadata = "all"
        if o.sharpen not in SHARPEN: o.sharpen = "off"
        o.suffix = "".join(ch for ch in str(o.suffix) if ch not in '<>:"/\\|?*')[:40]
        return o

    def sharpen_params(self):
        if self.sharpen == "off": return None
        return SHARPEN_PRESETS.get(self.sharpen, (self.radius, self.amount, self.threshold))

def resize_long_edge(rgb, long_edge):
    h, w = rgb.shape[:2]
    if not long_edge or max(h, w) <= long_edge: return rgb
    s = long_edge/max(h, w)
    return cv2.resize(rgb, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)

def sharpen(rgb, radius, amount, threshold, skin_mask=None, skin_protect=0.6):
    """Unsharp mask on luminance (no colour fringes), soft threshold, reduced on skin."""
    if amount <= 0: return rgb
    Y = cv2.transform(rgb.astype(np.float32), np.array([[0.299, 0.587, 0.114]], np.float32))
    d = Y - cv2.GaussianBlur(Y, (0, 0), max(0.3, radius))
    if threshold > 0:
        t = threshold*0.5
        d *= np.clip((np.abs(d) - t)*(1/max(t, 1e-3)), 0, 1)
    d *= amount
    if skin_mask is not None and skin_protect > 0:
        m = cv2.resize(skin_mask.astype(np.float32), (Y.shape[1], Y.shape[0]), interpolation=cv2.INTER_LINEAR)
        d *= 1 - skin_protect*np.clip(m, 0, 1)
    d16 = np.round(d).astype(np.int16)                    # integer add: much less memory traffic than float RGB
    out = rgb.astype(np.int16); out += d16[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)

def _exif(meta, mode):
    raw = meta.get("exif")
    if not raw or mode == "none": return None
    if mode == "no_gps":
        ex = Image.Exif(); ex.load(raw if raw.startswith(b"Exif") else b"Exif\x00\x00" + raw)
        ex.get_ifd(0x8825); ex.pop(0x8825, None)
        return ex.tobytes()
    return raw

def output_path(photo, out_dir, opts: ExportOptions) -> Path:
    return Path(out_dir) / (Path(photo).stem + opts.suffix + ".jpg")

def finalize(rgb, meta, opts: ExportOptions, skin_mask=None):
    """Returns (rgb, save kwargs). rgb is in the source colour space on input."""
    out = resize_long_edge(rgb, opts.long_edge)
    sp = opts.sharpen_params()
    if sp: out = sharpen(out, *sp, skin_mask=skin_mask, skin_protect=opts.sharpen_skin)
    src_icc = meta.get("icc_profile")
    if opts.profile == "keep":
        icc = src_icc
    else:
        icc = CM.icc_bytes(opts.profile); out = CM.convert(out, src_icc, icc)
    kw = dict(quality=opts.quality, subsampling=SUBSAMPLING[opts.subsampling], optimize=True)
    if icc: kw["icc_profile"] = icc
    ex = _exif(meta, opts.metadata)
    if ex: kw["exif"] = ex
    if opts.dpi: kw["dpi"] = (opts.dpi, opts.dpi)
    return out, kw

def save(path, rgb, meta, opts: ExportOptions, skin_mask=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    out, kw = finalize(rgb, meta, opts, skin_mask)
    tmp = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(out).save(tmp, "JPEG", **kw); tmp.replace(path)
    return path
