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

def export_one(engine: Engine, reference: Reference | None, photo, params, out_dir, opts, fit=None):
    dst = export.output_path(photo, out_dir, opts)
    if dst.resolve() == Path(photo).resolve():
        raise ValueError("Plik wyjściowy nadpisałby oryginał.")
    rgb, meta = imageio.load(photo)
    if fit is None:
        cc_flag = bool(reference is not None and getattr(reference, "colorchecker", False))
        ra = reference.analysis(engine, meta.get("icc_profile")) if reference is not None else None
        fit = engine.fit(rgb, ra, colorchecker=cc_flag)
    out, mask_q = engine.render(rgb, fit, params, return_mask=True)
    if params.angle or params.crop is not None:
        out = G.apply(out, params.angle, params.crop)
        mask_q = G.apply(mask_q.astype(np.float32), params.angle, params.crop, interp=cv2.INTER_LINEAR)
    return export.save(dst, out, meta, opts, skin_mask=mask_q)

def export_series(engine: Engine, reference: Reference | None, series: Series, photos, out_dir, opts=None,
                  progress=None, cancel=None):
    """progress(i, n, photo, seconds). Supports per-photo colour reference overrides from series."""
    opts = opts or export.ExportOptions(); out_dir = Path(out_dir); results = []; n = len(photos)
    ref_cache: dict[tuple[str, bool], Reference] = {}
    if reference is not None:
        ref_cache[(str(reference.path.resolve()), bool(getattr(reference, "colorchecker", False)))] = reference

    for i, ph in enumerate(photos):
        if cancel and cancel(): break
        t = time.perf_counter()
        try:
            p = series.params_for(ph)
            if p.reference_path == "__none__":
                eff_ref = None
            elif p.reference_path:
                rpath, rcc = series.effective_reference_for(ph, p)
                if rpath is not None and rpath.exists():
                    key = (str(rpath.resolve()), bool(rcc))
                    if key not in ref_cache:
                        ref_cache[key] = Reference(rpath, colorchecker=rcc)
                    eff_ref = ref_cache[key]
                else:
                    eff_ref = reference
            else:
                eff_ref = reference
            results.append((ph, export_one(engine, eff_ref, ph, p, out_dir, opts), None))
        except Exception as e:  # keep going with the rest of the series
            results.append((ph, None, str(e)))
        if progress: progress(i+1, n, ph, time.perf_counter()-t)
    return results
