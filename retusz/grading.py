"""AI grading transfer, file level: load an original (JPG/PNG/TIFF, 8/16 bit, or NEF) and its AI version,
bring the AI version into the original's colour space, run engine.transfer, write a JPG (EXIF + ICC kept)
or a 16-bit TIFF (ICC kept) of the original's size. Originals are only ever read."""
from __future__ import annotations
from pathlib import Path
import re, time
import numpy as np, cv2
from PIL import Image
from . import imageio, rawio, colorman as CM
from .engine import transfer as TR

SUFFIXES = ("jpg", "tif")

def load_original(path, bits16=False):
    """(rgb uint8 or uint16, meta). 16 bit for NEF when bits16, and for 16-bit PNG/TIFF files."""
    p = Path(path)
    if rawio.is_raw(p):
        return rawio.load16(p) if bits16 else rawio.load(p)
    if p.suffix.lower() in (".tif", ".tiff", ".png"):
        a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if a is not None and a.dtype == np.uint16 and a.ndim == 3:
            with Image.open(p) as im: icc = im.info.get("icc_profile"); exif = im.info.get("exif")
            return np.ascontiguousarray(a[..., 2::-1]), {"icc_profile": icc, "exif": exif}
    return imageio.load(p)

def load_ai(path, dst_icc):
    """AI version converted into the original's colour space (no profile = sRGB, as AI tools write)."""
    ai, meta = imageio.load(path)
    return CM.convert(ai, meta.get("icc_profile"), dst_icc)

def default_out(orig, fmt="jpg", out_dir=None):
    p = Path(orig); d = Path(out_dir) if out_dir else p.parent / "Retusz"
    return d / f"{p.stem}_grading-ai.{'tif' if fmt == 'tif' else 'jpg'}"

def _guard(out, *inputs):
    o = Path(out).resolve()
    for p in inputs:
        if o == Path(p).resolve(): raise ValueError("Plik wynikowy nie może nadpisać oryginału ani wersji AI.")

def save_tiff16(path, rgb16, icc):
    import tifffile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tifffile.imwrite(tmp, rgb16, photometric="rgb", compression="zlib", iccprofile=icc or CM.icc_bytes("srgb"))
    tmp.replace(path); return path

def save(path, prep, orig, meta, params, mask=None, quality=95):
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        return save_tiff16(path, TR.render(prep, orig, params, mask, out_dtype=np.uint16), meta.get("icc_profile"))
    return imageio.save_jpg(path, TR.render(prep, orig, params, mask, out_dtype=np.uint8), meta, quality)

def run(orig_path, ai_path, out_path, params: TR.TransferParams, mask=None, quality=95, flow="auto", loaded=None):
    """One pair. Returns a summary dict; raises align.AlignmentError when the AI version does not match."""
    _guard(out_path, orig_path, ai_path)
    t0 = time.perf_counter()
    tif = Path(out_path).suffix.lower() in (".tif", ".tiff")
    orig, meta = loaded or load_original(orig_path, bits16=tif)
    ai = load_ai(ai_path, meta.get("icc_profile"))
    prep = TR.prepare(orig, ai, flow=flow, protect=params.protect)
    save(out_path, prep, orig, meta, params, mask, quality)
    return {"out": Path(out_path), "align_px": prep.al.residual_px, "changed": prep.changed_frac,
            "coverage": prep.coverage_frac, "scale": prep.al.scale, "flow": prep.al.flow is not None,
            "size": orig.shape[1::-1], "seconds": time.perf_counter() - t0}

_SEP = re.compile(r"^[\s_\-.(\[]")

def _ai_for(stem, candidates):
    s = stem.lower(); best = None
    for c in candidates:
        cs = c.stem.lower()
        if cs == s or (cs.startswith(s) and _SEP.match(cs[len(s):])):
            if best is None or len(cs) < len(best.stem): best = c
    return best

def pair_files(orig_dir, ai_dir):
    """(original, AI version) pairs matched by file name: IMG_1.NEF <-> IMG_1_ai.png / IMG_1-gemini.jpg /
    'IMG_1 (gemini).jpg' / img_1.png. In one shared folder a file named like another file plus a suffix is
    the AI version of it."""
    orig_dir, ai_dir = Path(orig_dir), Path(ai_dir)
    origs = imageio.list_images(orig_dir); same = orig_dir.resolve() == ai_dir.resolve()
    ais = [p for p in imageio.list_images(ai_dir) if not rawio.is_raw(p)]
    if same:
        stems = [p.stem.lower() for p in origs]
        is_ai = lambda p: any(p.stem.lower() != s and p.stem.lower().startswith(s) and _SEP.match(p.stem.lower()[len(s):]) for s in stems)
        ai_set = [p for p in origs if is_ai(p)]
        origs = [p for p in origs if p not in ai_set]; ais = ai_set
    out = []
    for o in origs:
        a = _ai_for(o.stem, [p for p in ais if p != o])
        if a is not None: out.append((o, a))
    return sorted(out, key=lambda t: t[0].name.lower())

def batch(pairs, out_dir, params, fmt="jpg", quality=95, progress=None):
    res = []
    for i, (o, a) in enumerate(pairs, 1):
        try:
            r = run(o, a, default_out(o, fmt, out_dir), params, quality=quality); err = None
        except Exception as e:  # noqa: BLE001 - reported per photo
            r, err = None, str(e)
        res.append((o, r, err))
        if progress: progress(i, len(pairs), o, r, err)
    return res
