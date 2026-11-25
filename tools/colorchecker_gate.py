"""ColorChecker 24-patch quality gate: automatic detection under rotation, Root-Polynomial + Bradford
calibration to D65, per-patch ΔE00 report (mean < 1.2, skin #1/#2 < 1.0, neutral #19..#24 < 0.8),
and visual comparison sheets.
Usage: python tools/quality_gate.py --colorchecker OUTDIR [photo ...]"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retusz import imageio
from retusz.engine import colorchecker as CC, color as C
from retusz.engine.pipeline import Engine, Params
from tests.synthetic import portrait_with_colorchecker


def _label(im: Image.Image, text: str) -> Image.Image:
    im = im.copy()
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        f = ImageFont.load_default()
    if text:
        d.rectangle([0, 0, im.width, 32], fill=(18, 18, 22))
        d.text((8, 5), text, fill=(245, 245, 245), font=f)
    return im


def _grid(tiles: list[Image.Image]) -> Image.Image:
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    g = Image.new("RGB", (len(tiles) * (w + 6), h + 6), (24, 24, 28))
    for i, t in enumerate(tiles):
        g.paste(t, (i * (w + 6), 3))
    return g


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print("Usage: quality_gate.py --colorchecker OUTDIR [photo ...]")
        return 2
    out = Path(args.pop(0))
    out.mkdir(parents=True, exist_ok=True)

    eng = Engine(None, "classic", depth_ai=False)
    cases = []
    if args:
        for p in args:
            ph = Path(p)
            rgb, _ = imageio.load(ph)
            cases.append((ph.stem, rgb, None, None))
    else:
        for name, gains, ev, ang, seed in (
            ("wolfram_2850K", (1.32, 0.92, 0.64), -0.50, 6.0, 1),
            ("cien_7500K", (0.76, 1.02, 1.28), -0.25, -8.5, 2),
            ("led_zielony", (0.88, 1.18, 0.84), -0.35, 4.0, 3),
        ):
            cast_rgb, gt_rgb, _pts, info = portrait_with_colorchecker(
                h=1400, w=1050, patch_px=48, angle=ang, gains=gains, ev=ev, seed=seed
            )
            cases.append((name, cast_rgb, gt_rgb, info))

    results = []
    all_ok = True

    for name, src_rgb, gt_rgb, info in cases:
        cal = CC.calibrate_chart(src_rgb)
        if cal is None:
            print(f"== {name}: FAIL (nie wykryto tablicy ColorChecker 24)")
            all_ok = False
            results.append({"name": name, "detected": False, "ok": False})
            continue

        rep = cal.report
        fit = eng.prepare(src_rgb, colorchecker=True)
        p = Params(db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0)
        cal_img = eng.render(src_rgb, fit, p)

        fails = []
        if rep.de00_after >= 1.2:
            fails.append(f"średnie ΔE00 {rep.de00_after:.2f} >= 1.2")
        if rep.skin_de00 >= 1.0:
            fails.append(f"skóra (#1,#2) ΔE00 {rep.skin_de00:.2f} >= 1.0")
        if rep.gray_de00 >= 0.8:
            fails.append(f"szarości (#19..#24) ΔE00 {rep.gray_de00:.2f} >= 0.8")

        portrait_skin_de = None
        if gt_rgb is not None and info is not None:
            sk = info["skin"] > 0.5
            portrait_skin_de = CC.mean_de00_rgb(cal_img[sk], gt_rgb[sk])
            if portrait_skin_de >= 2.0:
                fails.append(f"skóra portretu ΔE00 {portrait_skin_de:.2f} >= 2.0")

        ok = len(fails) == 0
        all_ok = all_ok and ok
        ov = CC.draw_chart_overlay(src_rgb, cal)
        tiles = [
            _label(Image.fromarray(C.small(ov, 800)), f"{name}: źródło + detekcja 24 (ΔE₀₀={rep.de00_before:.1f})"),
            _label(
                Image.fromarray(C.small(cal_img, 800)),
                f"po kalibracji (ΔE₀₀={rep.de00_after:.2f} · skóra={rep.skin_de00:.2f}) {'✓' if ok else '✗'}",
            ),
        ]
        if gt_rgb is not None:
            tiles.append(_label(Image.fromarray(C.small(gt_rgb, 800)), "cel wzorcowy D65"))
        sheet_path = out / f"cc_{name}_arkusz.jpg"
        _grid(tiles).save(sheet_path, quality=92)

        print(
            f"== {name}: {rep.summary_pl()}"
            + (f" · skóra portretu ΔE₀₀={portrait_skin_de:.2f}" if portrait_skin_de is not None else "")
            + f" -> {'OK' if ok else 'FAIL: ' + '; '.join(fails)}"
        )
        results.append(
            {
                "name": name,
                "detected": True,
                "report": rep.to_dict(),
                "portrait_skin_de00": portrait_skin_de,
                "fails": fails,
                "ok": ok,
                "sheet": str(sheet_path),
            }
        )

    (out / "gate_colorchecker.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nGATE:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
