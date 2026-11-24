"""Quality gate: Retusz result vs the reference, per stage, plus preview == export.
Usage: quality_gate.py OUTDIR SRC1 REF1 [SRC2 REF2 ...]   (or --pair-export EXPORT REF to score an existing export)
Gate (defaults): skin chroma <= ref +5 %, skin hue within 4 deg of ref, global L/a/b mean within 3 of ref,
L std within 15 % of ref, clipped pixels <= max(src, ref) + 0.3 %, preview vs export dE < 1."""
import sys, json, time
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retusz import imageio, models, colorman as CM, export as EX
from retusz.reference import Reference
from retusz.engine.pipeline import Engine, Params
from retusz.engine import geometry as G
from retusz import batch
from retusz.engine import color as C, faces as F

def colourfulness(rgb):
    """Hasler & Süsstrunk 2003 colourfulness M (as in the color-grade skill)."""
    f = rgb.astype(np.float64); rg = f[..., 0]-f[..., 1]; yb = 0.5*(f[..., 0]+f[..., 1])-f[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3*np.hypot(rg.mean(), yb.mean()))

def stats(rgb):
    s = C.small(rgb, 1200); fs = F.pick_faces(s); m = F.skin_mask(s, fs); lab = C.to_lab(s)
    sk = m > 0.5; ch = np.hypot(lab[..., 1], lab[..., 2])
    hp = lambda L: cv2.GaussianBlur(L, (0, 0), 1.5) - cv2.GaussianBlur(L, (0, 0), 12)
    out = {"L": lab[..., 0].mean(), "a": lab[..., 1].mean(), "b": lab[..., 2].mean(), "Lsd": lab[..., 0].std(),
           "C": ch.mean(), "M": colourfulness(s), "clip": float(((s <= 1) | (s >= 254)).all(-1).mean()*100), "faces": len(fs)}
    if sk.sum() > 200:
        a, b = lab[..., 1][sk], lab[..., 2][sk]
        out |= {"skinC": float(np.hypot(a, b).mean()), "skinH": float(np.degrees(np.arctan2(np.median(b), np.median(a)))),
                "skinL": float(lab[..., 0][sk].mean()), "skinLC": float(np.abs(hp(lab[..., 0]))[sk].mean())}
    return out

def gate(o, r, src):
    fails = []
    if "skinC" in o and "skinC" in r:
        if o["skinC"] > r["skinC"]*1.05: fails.append(f"skóra chroma {o['skinC']:.1f} > wzorzec {r['skinC']:.1f}+5%")
        if abs(o["skinH"]-r["skinH"]) > 4: fails.append(f"odcień skóry {o['skinH']:.1f}° vs {r['skinH']:.1f}°")
        if o["skinLC"] > max(r["skinLC"], src.get("skinLC", 0))*1.15: fails.append(f"kontrast lokalny skóry {o['skinLC']:.2f} vs {r['skinLC']:.2f}")
    for k in "Lab":
        if abs(o[k]-r[k]) > 3: fails.append(f"{k} śr. {o[k]:.1f} vs {r[k]:.1f}")
    if abs(o["Lsd"]/r["Lsd"]-1) > 0.15: fails.append(f"L std {o['Lsd']:.1f} vs {r['Lsd']:.1f}")
    if o["C"] > r["C"]*1.05: fails.append(f"chroma {o['C']:.1f} > {r['C']:.1f}+5%")
    if o["M"] > r["M"]*1.05: fails.append(f"kolorowość M {o['M']:.1f} > {r['M']:.1f}+5%")
    if o["clip"] > max(src["clip"], r["clip"]) + 0.3: fails.append(f"przycięte {o['clip']:.2f}%")
    return fails

def fmt(d): return " ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())

def grid(tiles, cols):
    w = max(t.width for t in tiles); h = max(t.height for t in tiles); rows = (len(tiles)+cols-1)//cols
    g = Image.new("RGB", (cols*(w+6), rows*(h+6)), (25, 25, 25))
    for i, t in enumerate(tiles): g.paste(t, ((i % cols)*(w+6), (i//cols)*(h+6)))
    return g

def label(im, text):
    im = im.copy(); d = ImageDraw.Draw(im)
    try: f = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError: f = ImageFont.load_default()
    if text: d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0)); d.text((8, 5), text, fill=(255, 255, 255), font=f)
    return im

STAGES = [("kolory", dict(db=0, plastic=0, vignette=0, skin_even=0)), ("+winieta+równanie", dict(db=0, plastic=0)),
          ("+D&B", dict(plastic=0)), ("domyślne", {})]

def run_pair(eng, src_p, ref_p, out, idx):
    rgb, meta = imageio.load(src_p); icc = meta.get("icc_profile"); ref = Reference(ref_p)
    ref_srgb = C.crop_border(CM.convert(ref.rgb, ref.icc, None)); r = stats(ref_srgb); s = stats(CM.convert(rgb, icc, None))
    fit = eng.fit(rgb, ref.analysis(eng, icc)); rows = []; tiles = [label(Image.fromarray(C.small(CM.convert(rgb, icc, None), 700)), "źródło"),
                                                                       label(Image.fromarray(C.small(ref_srgb, 700)), "wzorzec")]
    res = {"src": src_p.name, "ref": ref_p.name, "ref_stats": r, "src_stats": s, "stages": {}}
    print(f"\n== {src_p.name} -> {ref_p.name}\n  wzorzec: {fmt(r)}\n  źródło : {fmt(s)}")
    nodb = None
    for name, kw in STAGES:
        o_rgb = eng.render(rgb, fit, Params(**kw)); o = stats(CM.convert(o_rgb, icc, None)); f = gate(o, r, s)
        if name == "+winieta+równanie": nodb = o_rgb
        if name == "domyślne" and nodb is not None:          # seams: line-like steps in what D&B/3D changed
            d = cv2.resize(o_rgb.astype(np.float32).mean(2) - nodb.astype(np.float32).mean(2), (1200, int(1200*rgb.shape[0]/rgb.shape[1])), interpolation=cv2.INTER_AREA)
            col = np.abs(np.diff(d, axis=1)).mean(0); row = np.abs(np.diff(d, axis=0)).mean(1)
            seam = float(max(col.max(), row.max())); o["seam"] = seam          # mean |step| along the worst line, 8-bit units
            if seam > 1.0: f.append(f"szwy/halo {seam:.2f}")
        if src_p.resolve() == ref_p.resolve():               # identity: same photo as reference must stay almost unchanged
            de = float(np.linalg.norm(C.to_lab(C.small(o_rgb, 1200)) - C.to_lab(C.small(rgb, 1200)), axis=2).mean()); o["dE_id"] = de
            if name == "kolory" and de > 3: f.append(f"tożsamość dE {de:.1f} > 3")
        res["stages"][name] = {"stats": o, "fails": f}
        print(f"  {name:18s}: {fmt(o)}\n  {'':18s}  {'OK' if not f else 'FAIL: ' + '; '.join(f)}")
        tiles.append(label(Image.fromarray(C.small(CM.convert(o_rgb, icc, None), 700)), name + (" ✓" if not f else " ✗")))
    # preview == export (export downscaled, colour-managed comparison in sRGB), for the defaults and for
    # white balance and straighten + crop (0.1.3); the export goes through the real batch.export_one path
    prev, picc = imageio.load_preview(src_p, 1600, with_icc=True); fp = eng.fit(prev, ref.analysis(eng, picc))
    cases = {"domyślne": Params(), "balans bieli": Params(temp=0.3, tint=-0.15),
             "prostowanie+kadr": Params(angle=4.0, aspect="4:5", crop=G.aspect_rect("4:5", rgb.shape[1], rgb.shape[0], True))}
    res["pve"] = {}
    for cname, cp in cases.items():
        P = eng.render(prev, fp, cp)
        geo = bool(cp.angle or cp.crop is not None)
        if geo: P = G.apply(P, cp.angle, cp.crop, interp=cv2.INTER_CUBIC)   # what the GUI shows
        P = CM.convert(P, picc, None)
        dst = batch.export_one(eng, ref, src_p, cp, out/f"export_{idx}", EX.ExportOptions(sharpen="off", quality=98, suffix="_"+cname.replace(" ", "_").replace("+", "_")), fit=fit)
        im = Image.open(dst); E = CM.convert(np.asarray(im.convert("RGB")), im.info.get("icc_profile"), None)
        E = cv2.resize(E, (P.shape[1], P.shape[0]), interpolation=cv2.INTER_AREA)
        de = float(np.linalg.norm(C.to_lab(P)-C.to_lab(E), axis=2).mean()); note = ""
        if geo:   # a rotated 1600 px preview is resampled once more than the export: compare colour, not sharpness
            b = lambda x: cv2.GaussianBlur(x, (0, 0), 1.0)
            raw = de; de = float(np.linalg.norm(C.to_lab(b(P))-C.to_lab(b(E)), axis=2).mean()); note = f" (kolor po rozmyciu 1 px; ostrość: {raw:.2f})"
        res["pve"][cname] = de
        print(f"  podgląd vs eksport [{cname}]: dE {de:.2f} {'OK' if de < 1.0 else 'FAIL'}{note}  ({im.size[0]}×{im.size[1]})")
    res["preview_vs_export_dE"] = max(res["pve"].values())
    sheet = grid(tiles, 3)
    faces = F.pick_faces(C.small(CM.convert(rgb, icc, None), 700))
    if faces:   # face close-ups: union of all faces, padded, from every tile (reference by its own faces)
        fx = [f for f in faces]; x0 = min(f[0] for f in fx); y0 = min(f[1] for f in fx); x1 = max(f[0]+f[2] for f in fx); y1 = max(f[1]+f[3] for f in fx)
        pad = 0.35*(x1-x0); crops = []
        for t in tiles:
            k = t.width/700 if t is not tiles[1] else None
            if k is None:
                rf = F.pick_faces(np.asarray(t))
                if not rf: continue
                X0 = min(f[0] for f in rf); Y0 = min(f[1] for f in rf); X1 = max(f[0]+f[2] for f in rf); Y1 = max(f[1]+f[3] for f in rf); pd = 0.35*(X1-X0)
                box = (X0-pd, Y0-pd, X1+pd, Y1+pd*1.5)
            else: box = (x0-pad, y0-pad, x1+pad, y1+pad*1.5)
            c = t.crop(tuple(int(v) for v in box)).resize((520, int(520*(box[3]-box[1])/(box[2]-box[0]))))
            crops.append(label(c, ""))
        grid(crops, 3).save(out/f"faces_{idx}_{src_p.stem}.jpg", quality=90)
    sheet.save(out/f"sheet_{idx}_{src_p.stem}.jpg", quality=88); return res

def main():
    args = sys.argv[1:]; out = Path(args.pop(0)); out.mkdir(parents=True, exist_ok=True)
    if args and args[0] == "--pair-export":
        e, r = [np.asarray(Image.open(p).convert("RGB")) for p in args[1:3]]
        so, sr = stats(e), stats(r); print("eksport:", fmt(so)); print("wzorzec:", fmt(sr)); print("FAIL:" if gate(so, sr, sr) else "OK", gate(so, sr, sr)); return
    p, eff = models.resolve("ai"); eng = Engine(p, eff); print("silnik", eff)
    results = [run_pair(eng, Path(args[i]), Path(args[i+1]), out, i//2) for i in range(0, len(args), 2)]
    (out/"gate.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    ok = all(not r["stages"]["domyślne"]["fails"] and r["preview_vs_export_dE"] < 1 for r in results)
    print("\nGATE:", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
