"""Headless batch: python -m retusz cli <folder|photo> [--reference ref.jpg] [--colorchecker chart.jpg] [--preset NAME] [--out DIR] [--quality 95]"""
import argparse, sys, time
from pathlib import Path
from . import imageio, models, batch, export, presets as PR
from .reference import Reference
from .engine.pipeline import Engine
from .sidecar import Series

def _utf8_console():
    """Polish text must not crash on cp1252/cp852 Windows consoles."""
    for st in (sys.stdout, sys.stderr):
        try: st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError): pass

def main(argv=None):
    _utf8_console()
    ap = argparse.ArgumentParser(prog="retusz cli")
    ap.add_argument("path")
    ap.add_argument("--reference", "-r", default=None, help="zdjęcie wzorcowe (opcjonalne)")
    ap.add_argument("--colorchecker", "-c", default=None, help="zdjęcie z tablicą ColorChecker 24 do kalibracji serii")
    ap.add_argument("--preset", "-p", default=None, help="nazwa presetu do nałożenia na serię")
    ap.add_argument("--out"); ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--method", choices=list(models.METHODS), default="ai")
    ap.add_argument("--long-edge", type=int, default=0); ap.add_argument("--suffix", default="")
    ap.add_argument("--profile", choices=list(export.CM.OUTPUT_PROFILES), default="keep")
    ap.add_argument("--sharpen", choices=list(export.SHARPEN), default="screen")
    a = ap.parse_args(argv)
    path = Path(a.path)
    ref_p = Path(a.colorchecker) if a.colorchecker else (Path(a.reference) if a.reference else None)
    is_cc = bool(a.colorchecker)
    folder = path if path.is_dir() else path.parent
    photos = [p for p in imageio.list_images(folder) if (ref_p is None or p.resolve() != ref_p.resolve() or is_cc)] if path.is_dir() else [path]
    series = Series(folder)
    if a.preset:
        pr = next((x for x in PR.list_presets() if x.name.lower() == a.preset.lower()), None)
        if pr is None:
            print(f"BŁĄD: Nieznany preset '{a.preset}'. Dostępne: {', '.join(x.name for x in PR.list_presets())}")
            return 1
        series.params = pr.apply_to(series.params)
    mp, eff = models.resolve(a.method); eng = Engine(mp, eff)
    print(f"Silnik: {eff} · {eng.device()} · {len(photos)} zdjęć")
    ref = Reference(ref_p, colorchecker=is_cc) if ref_p is not None else None
    out = Path(a.out) if a.out else folder / "Retusz"; t = time.perf_counter()
    opts = export.ExportOptions(quality=a.quality, long_edge=a.long_edge, profile=a.profile, sharpen=a.sharpen, suffix=a.suffix)
    res = batch.export_series(eng, ref, series, photos, out, opts,
                              progress=lambda i, n, ph, s: print(f"{i}/{n} {Path(ph).name} {s:.2f} s"))
    errs = [r for r in res if r[2]]
    for ph, _, e in errs: print("BŁĄD", ph, e)
    print(f"Zapisano {len(res)-len(errs)} w {out} ({time.perf_counter()-t:.1f} s)")
    return 1 if errs else 0

def transfer_main(argv=None):
    """python -m retusz transfer ORYGINAL WERSJA_AI [--out PLIK] | --batch FOLDER_ORYG FOLDER_AI --out FOLDER"""
    _utf8_console()
    from . import grading as GR
    from .engine.transfer import TransferParams
    from .engine.align import AlignmentError
    ap = argparse.ArgumentParser(prog="retusz transfer", description="Przenieś grading z wersji AI na pełny oryginał")
    ap.add_argument("orig"); ap.add_argument("ai"); ap.add_argument("--out")
    ap.add_argument("--batch", action="store_true", help="orig i ai to foldery; pary po nazwach plików")
    ap.add_argument("--kolor", type=float, default=100); ap.add_argument("--jasnosc", type=float, default=100)
    ap.add_argument("--no-protect", action="store_true", help="nie chroń miejsc zmienionych przez AI")
    ap.add_argument("--tiff", action="store_true", help="TIFF 16-bit zamiast JPG"); ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--flow", choices=["auto", "on", "off"], default="auto")
    a = ap.parse_args(argv)
    prm = TransferParams(kolor=a.kolor/100, jasnosc=a.jasnosc/100, protect=not a.no_protect)
    fmt = "tif" if a.tiff else "jpg"
    def show(r): return (f"dopasowanie {r['align_px']:.2f} px, skala {r['scale']:.3f}, zmienione przez AI {r['changed']*100:.1f}%, "
                         f"pokrycie {r['coverage']*100:.0f}%, {r['size'][0]}×{r['size'][1]}, {r['seconds']:.1f} s")
    if a.batch:
        pairs = GR.pair_files(a.orig, a.ai)
        if not pairs: print("Nie znaleziono par (oryginał + wersja AI o tej samej nazwie)."); return 1
        out = Path(a.out) if a.out else Path(a.orig) / "Retusz"
        res = GR.batch(pairs, out, prm, fmt, a.quality, progress=lambda i, n, o, r, e: print(f"{i}/{n} {Path(o).name}: " + (show(r) if r else "BŁĄD " + e)))
        return 1 if any(e for _, _, e in res) else 0
    out = Path(a.out) if a.out else GR.default_out(a.orig, fmt)
    try:
        r = GR.run(a.orig, a.ai, out, prm, quality=a.quality, flow={"on": True, "off": False}.get(a.flow, "auto"))
    except (AlignmentError, ValueError) as e:
        print("BŁĄD:", e); return 1
    print("Zapisano", out, "·", show(r)); return 0
