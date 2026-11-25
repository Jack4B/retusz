"""Nikon RAW (NEF/NRW) input through rawpy (MIT) / LibRaw (used unmodified, CDDL-1.0).

Decode: camera white balance, LibRaw's camera matrix to linear sRGB, 16 bits per channel.
Then, through a 65536-entry float table: exposure (matched to the camera's embedded JPEG, so the photo starts as bright as on
the camera screen), a soft highlight shoulder, sRGB encoding, and one ordered-dither quantisation
to 8 bits for the engine. Both export and the 1600 px preview demosaic with AHD and then develop at
full resolution (the preview is a downscale of that). LibRaw's half-size mode shifts R against B by
half a pixel (preview-vs-export ΔE 2.7); a faster PPG preview drifted to ΔE 1.35–1.53 on noisy D700
files. The series strip uses the embedded camera JPEG (instant)."""
from __future__ import annotations
import io, threading
from functools import lru_cache
from pathlib import Path
import numpy as np, cv2
from PIL import Image

RAW_EXTS = {".nef", ".nrw"}
KNEE = 0.8                   # linear value where the highlight shoulder starts
TARGET_DEFAULT = 50.0        # mean CIE L* when the file has no embedded JPEG
_Y = np.array([0.2126, 0.7152, 0.0722], np.float32)
_BAYER = (np.array([[0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26], [12, 44, 4, 36, 14, 46, 6, 38],
                    [60, 28, 52, 20, 62, 30, 54, 22], [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
                    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21]], np.float32) + 0.5)/64 - 0.5
_lock = threading.Lock()

def is_raw(path) -> bool:
    return Path(path).suffix.lower() in RAW_EXTS

def _rawpy():
    import rawpy
    return rawpy

def orient(a, flip):
    """Apply a LibRaw flip code (0 none, 3 = 180°, 5 = 90° CCW, 6 = 90° CW)."""
    if flip == 3: return a[::-1, ::-1]
    if flip == 5: return np.rot90(a, 1)
    if flip == 6: return np.rot90(a, -1)
    return a

def shoulder(x):
    x = np.asarray(x, np.float32); k = KNEE
    return np.where(x <= k, x, k + (1-k)*(1-np.exp(-(x-k)/(1-k)))).astype(np.float32)

def encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin*12.92, 1.055*np.power(lin, 1/2.4, dtype=np.float32) - 0.055).astype(np.float32)

def quantise(v):
    """0..1 float -> uint8 with an 8x8 ordered dither (no banding after strong colour grading)."""
    h, w = v.shape[:2]
    d = np.tile(_BAYER, (h//8+1, w//8+1))[:h, :w]
    if v.ndim == 3: d = d[..., None]
    return np.clip(np.floor(v*255 + 0.5 + d), 0, 255).astype(np.uint8)

def _decode16(path, half=False, fast=False):
    rawpy = _rawpy()
    kw = {"half_size": True} if half else {"demosaic_algorithm": rawpy.DemosaicAlgorithm.PPG if fast else rawpy.DemosaicAlgorithm.AHD}
    with rawpy.imread(str(path)) as r:
        return r.postprocess(use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB, output_bps=16,
                             gamma=(1, 1), no_auto_bright=True, **kw)

def _thumb_jpeg(path):
    rawpy = _rawpy()
    try:
        with rawpy.imread(str(path)) as r:
            t = r.extract_thumb(); flip = r.sizes.flip
    except Exception:
        return None, 0
    if t.format == rawpy.ThumbFormat.JPEG: return t.data, flip
    return None, flip

def _small(a16, side=512):
    h, w = a16.shape[:2]; s = side/max(h, w)
    return cv2.resize(a16, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA).astype(np.float32)/65535

def _mean_lstar(lin):
    """Mean CIE L* of linear RGB (Rec.709/sRGB primaries)."""
    y = np.clip(lin.reshape(-1, 3) @ _Y, 0, 1)
    f = np.where(y > 216/24389, np.cbrt(y), (24389/27*y + 16)/116)
    return float((116*f - 16).mean())

def _decode_srgb(a):
    return np.where(a <= 0.04045, a/12.92, ((a+0.055)/1.055)**2.4).astype(np.float32)

@lru_cache(maxsize=64)
def _target(path, mtime):
    """Mean L* of the camera's embedded JPEG (what the photographer saw on the camera screen)."""
    data, _ = _thumb_jpeg(path)
    if data is None: return TARGET_DEFAULT
    im = Image.open(io.BytesIO(data)); im.draft("RGB", (512, 512)); a = np.asarray(im.convert("RGB"), np.float32)/255
    return _mean_lstar(_decode_srgb(a))

_gain_cache = {}

def exposure_gain(path, a16=None):
    """Linear gain so the developed photo's mean L* matches the embedded camera JPEG (cached per file).
    Mean L*, not median luminance: with a high-key frame the camera's S-curve and our gentler shoulder
    give the same median but a mean L* up to 7 apart (measured on D700 files)."""
    p = Path(path); k = (str(p.resolve()), p.stat().st_mtime_ns)
    with _lock:
        if k in _gain_cache: return _gain_cache[k]
    if a16 is None: a16 = _decode16(p, True)
    small = _small(a16); tgt = _target(*k)
    lo, hi = np.log(0.25), np.log(8.0)
    for _ in range(28):
        m = (lo+hi)/2
        cur = _mean_lstar(shoulder(small*np.exp(m)))
        lo, hi = (m, hi) if cur < tgt else (lo, m)
    g = float(np.exp((lo+hi)/2))
    with _lock: _gain_cache[k] = g
    return g

@lru_cache(maxsize=8)
def _tone_lut(gain):
    return (encode(shoulder(np.arange(65536, dtype=np.float32)*np.float32(gain/65535)))*255 + 0.5).astype(np.float32)

def develop(a16, gain, rows=256):
    """16-bit linear sRGB -> 8-bit sRGB through the tone table, in strips (bounded memory for 24 Mpx)."""
    lut = _tone_lut(round(float(gain), 6)); out = np.empty(a16.shape, np.uint8); h, w = a16.shape[:2]
    d = np.tile(_BAYER, (rows//8+1, w//8+1))[:rows, :w, None]
    for y in range(0, h, rows):
        v = lut[a16[y:y+rows]]; v += d[:v.shape[0]]
        np.clip(v, 0, 255, out=v); out[y:y+rows] = v.astype(np.uint8)
    return out

def _exif(path):
    """EXIF from the NEF's TIFF header (camera, lens, exposure, date, GPS), orientation reset to 1."""
    try:
        src = Image.open(str(path)).getexif()
    except Exception:
        return None
    ex = Image.Exif()
    for t in (0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298):
        if t in src: ex[t] = src[t]
    ex[0x0112] = 1
    try:
        e = src.get_ifd(0x8769); d = ex.get_ifd(0x8769)
        for t, v in e.items():
            if t not in (0x927C, 0xA005) and not isinstance(v, (dict, tuple)) or isinstance(v, tuple) and len(v) <= 8: d[t] = v
        d.pop(0x927C, None); d.pop(0xA005, None)
        g = src.get_ifd(0x8825)
        if g: ex.get_ifd(0x8825).update(g)
    except Exception:
        pass
    try:
        return ex.tobytes()
    except Exception:
        ex2 = Image.Exif(); [ex2.__setitem__(t, ex[t]) for t in list(ex) if t not in (0x8769, 0x8825)]
        return ex2.tobytes()

def _meta(path):
    from . import colorman as CM
    return {"icc_profile": CM.icc_bytes("srgb"), "exif": _exif(path)}

def load(path):
    a16 = _decode16(path)
    return np.ascontiguousarray(develop(a16, exposure_gain(path, a16))), _meta(path)

def load_preview(path, side, with_icc=False):
    from . import colorman as CM
    a = None
    if side <= 400:                                   # series strip: embedded camera JPEG, instant
        data, flip = _thumb_jpeg(path)
        if data is not None:
            im = Image.open(io.BytesIO(data)); im.draft("RGB", (side*2, side*2)); im = im.convert("RGB")
            im.thumbnail((side, side), Image.LANCZOS); a = np.ascontiguousarray(orient(np.asarray(im), flip))
    if a is None:
        a16 = _decode16(path)
        rgb = develop(a16, exposure_gain(path, a16))
        h, w = rgb.shape[:2]; s = min(1.0, side/max(h, w))
        a = rgb if s >= 1 else cv2.resize(rgb, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)
    return (a, CM.icc_bytes("srgb")) if with_icc else a

def image_size(path):
    with _rawpy().imread(str(path)) as r:
        s = r.sizes; w, h = s.width, s.height
        return (h, w) if s.flip in (5, 6) else (w, h)

@lru_cache(maxsize=8)
def _tone_lut16(gain):
    return (encode(shoulder(np.arange(65536, dtype=np.float32)*np.float32(gain/65535)))*65535 + 0.5).astype(np.uint16)

def load16(path):
    """Same develop as load(), kept at 16 bit (for the AI grading transfer / 16-bit TIFF output)."""
    a16 = _decode16(path)
    return np.ascontiguousarray(_tone_lut16(round(float(exposure_gain(path, a16)), 6))[a16]), _meta(path)
