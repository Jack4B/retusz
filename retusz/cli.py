"""Headless batch: python -m retusz cli <folder|photo> --reference ref.jpg [--out DIR] [--quality 95]"""
import argparse, time
from pathlib import Path
from . import imageio, models, batch, export
from .reference import Reference
from .engine.pipeline import Engine
from .sidecar import Series

def main(argv=None):
    ap = argparse.ArgumentParser(prog="retusz cli")
    ap.add_argument("path"); ap.add_argument("--reference", "-r", required=True)
    ap.add_argument("--out"); ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--method", choices=list(models.METHODS), default="ai")
    ap.add_argument("--long-edge", type=int, default=0); ap.add_argument("--suffix", default="")
    ap.add_argument("--profile", choices=list(export.CM.OUTPUT_PROFILES), default="keep")
    ap.add_argument("--sharpen", choices=list(export.SHARPEN), default="screen")
    a = ap.parse_args(argv)
    path, ref_p = Path(a.path), Path(a.reference)
    folder = path if path.is_dir() else path.parent
    photos = [p for p in imageio.list_images(folder) if p.resolve() != ref_p.resolve()] if path.is_dir() else [path]
    series = Series(folder)
    mp, eff = models.resolve(a.method); eng = Engine(mp, eff)
    print(f"Silnik: {eff} · {eng.device()} · {len(photos)} zdjęć")
    ref = Reference(ref_p)
    out = Path(a.out) if a.out else folder / "Retusz"; t = time.perf_counter()
    opts = export.ExportOptions(quality=a.quality, long_edge=a.long_edge, profile=a.profile, sharpen=a.sharpen, suffix=a.suffix)
    res = batch.export_series(eng, ref, series, photos, out, opts,
                              progress=lambda i, n, ph, s: print(f"{i}/{n} {Path(ph).name} {s:.2f} s"))
    errs = [r for r in res if r[2]]
    for ph, _, e in errs: print("BŁĄD", ph, e)
    print(f"Zapisano {len(res)-len(errs)} w {out} ({time.perf_counter()-t:.1f} s)")
    return 1 if errs else 0
