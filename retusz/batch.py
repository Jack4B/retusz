"""Batch export of a series: one reference, per-photo fit + overrides, advanced JPG export."""
from __future__ import annotations
import time
import numpy as np, cv2
from pathlib import Path
from . import imageio, export
from .engine.pipeline import Engine
from .engine import geometry as G
from .reference import Reference
from .sidecar import Series

def export_one(engine: Engine, reference: Reference, photo, params, out_dir, opts, fit=None):
    dst = export.output_path(photo, out_dir, opts)
    if dst.resolve() == Path(photo).resolve():
        raise ValueError("Plik wyjściowy nadpisałby oryginał.")
    rgb, meta = imageio.load(photo)
    if fit is None:
        fit = engine.fit(rgb, reference.analysis(engine, meta.get("icc_profile")))
    out, mask_q = engine.render(rgb, fit, params, return_mask=True)
    if params.angle or params.crop is not None:
        out = G.apply(out, params.angle, params.crop)
        mask_q = G.apply(mask_q.astype(np.float32), params.angle, params.crop, interp=cv2.INTER_LINEAR)
    return export.save(dst, out, meta, opts, skin_mask=mask_q)

def export_series(engine: Engine, reference: Reference, series: Series, photos, out_dir, opts=None,
                  progress=None, cancel=None):
    """progress(i, n, photo, seconds). Returns list of (photo, output or None, error or None)."""
    opts = opts or export.ExportOptions(); out_dir = Path(out_dir); results = []; n = len(photos)
    for i, ph in enumerate(photos):
        if cancel and cancel(): break
        t = time.perf_counter()
        try:
            results.append((ph, export_one(engine, reference, ph, series.params_for(ph), out_dir, opts), None))
        except Exception as e:  # keep going with the rest of the series
            results.append((ph, None, str(e)))
        if progress: progress(i+1, n, ph, time.perf_counter()-t)
    return results
