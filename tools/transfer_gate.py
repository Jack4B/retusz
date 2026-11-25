"""Quality gate for the AI grading transfer (run via: quality_gate.py --transfer OUTDIR [photo.jpg|photo.NEF ...]).
Ground truth GT = grade(original) (tone curve, colour matrix, smooth relight, sharp-edged local grade); the
"AI version" is GT downscaled 2-4x, cropped, shifted, rotated, JPEG q90 (+ one non-rigid drift, one removed
object, one identity case) + a generative case (grainy original; the AI copy is re-rendered: local relight and
strong regrade, other framing, non-rigid drift, all fine detail redrawn and smoothed, one region redrawn - the
change map must flag only that region and the result there must not show the AI's content (no ghost)).
Thresholds: see THRESH. Sheets: original 100 % | naive AI upscale | result | GT."""
import sys, json, time, hashlib, tempfile
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tests"))
from synthetic import portrait_scene, textured, grade, make_ai, warp_error_px, content_mask, make_generative, strong_grade, relight
from retusz import imageio, grading as GR
from retusz.engine import transfer as TR
from quality_gate import grid, label

# sharp = detail_slope (Laplacian amplitude vs GT)
THRESH = dict(align_px=0.5, dE=2.0, sharp=0.95, halo_dE=3.0, identity_dE=0.6, removed_dE=4.0, changed_gen_pct=12.0, sharp_rel=0.97, ghost_ncc=0.2, own_ncc=0.9)

def hf(img, m=None):
    g = img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return float(lap[m].var() if m is not None else lap.var())

def detail_slope(out, gt, m):
    """Fraction of the ground truth's fine detail (Laplacian) amplitude reproduced: <L(out),L(gt)>/<L(gt),L(gt)>.
    1.0 = as sharp as GT; a blurred result scales it down. Unlike a variance ratio it does not count, as lost
    sharpness, GT texture that no smooth colour transform can explain (e.g. a luminance-keyed mask)."""
    Y = lambda a: a.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    lo, lg = cv2.Laplacian(Y(out), cv2.CV_32F)[m], cv2.Laplacian(Y(gt), cv2.CV_32F)[m]
    return float(np.dot(lo, lg)/max(np.dot(lg, lg), 1e-9))

def lab(x): return TR.to_lab(x.astype(np.float32)/255)

def de_map(a, b): return TR.delta_e2000(lab(a), lab(b)).astype(np.float32)

def drift(ai, ax=1.2, ay=1.0):
    h, w = ai.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(ai, xx + ax*np.sin(xx/90), yy + ay*np.cos(yy/70), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

def drift_true(T, P, ax=1.2, ay=1.0):
    q = (T @ np.c_[P, np.ones(len(P))].T).T; q2 = q.copy()
    for _ in range(12): q2 = q - np.c_[ax*np.sin(q2[:, 0]/90), ay*np.cos(q2[:, 1]/70)]
    return q2

def _hp(img, s=1.5):
    g = img.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return g - cv2.GaussianBlur(g, (0, 0), 4*s) if s <= 0 else cv2.GaussianBlur(g, (0, 0), s/2) - cv2.GaussianBlur(g, (0, 0), 4*s)

def _corr(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum() + 1e-9))

def run_case(name, o, gt, ai, true_map, outdir, kind="normal", patch=None, changed_ai=None, ref_out=None):
    t0 = time.perf_counter()
    prep = TR.prepare(o, ai); t1 = time.perf_counter()
    out = TR.render(prep, o, TR.TransferParams()); t2 = time.perf_counter()
    H, W = o.shape[:2]
    cov = prep.coverage_full(o.shape) > 0.999
    inside = cv2.erode(cov.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    res = {"case": name, "size": f"{W}x{H}", "ai": f"{ai.shape[1]}x{ai.shape[0]}", "prepare_s": round(t1-t0, 2), "render_s": round(t2-t1, 2),
           "residual_px": round(prep.al.residual_px, 3), "flow": prep.al.flow is not None, "changed_%": round(100*prep.changed_frac, 2)}
    fails = []
    if true_map is not None:
        res["align_px"] = round(warp_error_px(true_map, prep.al.full_to_ai, o.shape, ai.shape) if not callable(true_map) else
                                float(np.mean([np.linalg.norm(a-b) for a, b in zip(*_pts(true_map, prep, o))])/prep.al.scale), 3)
        if res["align_px"] >= THRESH["align_px"]: fails.append("align")
    dm = de_map(out, gt)
    res["dE_mean"] = round(float(dm[inside].mean()), 3); res["dE_p95"] = round(float(np.percentile(dm[inside], 95)), 2)
    al = prep.al; mx, my = _full_maps(al, H, W)
    naive = cv2.remap(ai, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    res["sharp"] = round(detail_slope(out, gt, inside), 3); res["sharp_naive"] = round(detail_slope(naive, gt, inside), 3)
    res["hf_var_ratio"] = round(hf(out, inside)/hf(gt, inside), 3)
    edges = cv2.dilate(cv2.Canny(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY), 80, 180), np.ones((25, 25), np.uint8)).astype(bool) & inside
    res["halo_dE"] = round(float(dm[edges].mean()), 3) if edges.any() else 0.0
    if kind == "info":
        pass                                              # stress case: reported, not gated
    elif kind == "identity":
        res["identity_dE"] = res["dE_mean"]
        if res["identity_dE"] >= THRESH["identity_dE"]: fails.append("identity")
    else:
        if res["dE_mean"] >= THRESH["dE"]: fails.append("dE")
        if res["halo_dE"] >= THRESH["halo_dE"]: fails.append("halo")
    if ref_out is not None:                               # generative: sharpness vs the same grade on a plain AI copy
        res["sharp_ref"] = round(detail_slope(ref_out, gt, inside), 3); res["sharp_rel"] = round(res["sharp"]/res["sharp_ref"], 3)
        if res["sharp_rel"] < THRESH["sharp_rel"]: fails.append("sharp")
    elif res["sharp"] < THRESH["sharp"] and kind != "info": fails.append("sharp")
    if changed_ai is not None:                            # generative: only the redrawn region may be flagged
        fx, fy = _full_maps(prep.al, H, W)
        tm = cv2.remap(changed_ai, fx, fy, cv2.INTER_LINEAR, borderValue=0) > 0.5
        core = cv2.erode(tm.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool) & inside
        res["redrawn_dE"] = round(float(dm[core].mean()), 3) if core.any() else 0.0   # info: no AI data there
        ho, hg, hn = _hp(out)[core], _hp(o)[core], _hp(naive)[core]   # ghost: the AI's redrawn detail in the result; own: the original's
        hg0, hn0 = hg - hg.mean(), hn - hn.mean()
        hn_new = hn0 - hg0*float((hn0*hg0).sum()/max((hg0*hg0).sum(), 1e-9))   # AI detail the original does not explain
        X = np.stack([(cv2.GaussianBlur(c, (0, 0), 0.75) - cv2.GaussianBlur(c, (0, 0), 6))[core] for c in
                      np.moveaxis(o.astype(np.float32), -1, 0)] + [np.ones(int(core.sum()), np.float32)], 1)
        coef = np.linalg.lstsq(X, ho, rcond=None)[0]                  # detail of an affine colour transform of the
        own = 1 - float(((ho - X @ coef)**2).sum()/max(((ho - ho.mean())**2).sum(), 1e-9))   # original: R^2 = 1
        res["ghost_ncc"] = round(_corr(ho, hn_new), 3); res["own_ncc"] = round(float(np.sqrt(max(own, 0))), 3)
        if res["changed_%"] >= THRESH["changed_gen_pct"]: fails.append("changed")
        if res["ghost_ncc"] >= THRESH["ghost_ncc"] or res["own_ncc"] < THRESH["own_ncc"]: fails.append("ghost")
        rest = inside & ~cv2.dilate(tm.astype(np.uint8), np.ones((61, 61), np.uint8)).astype(bool)
        res["dE_mean"] = round(float(dm[rest].mean()), 3); res["dE_p95"] = round(float(np.percentile(dm[rest], 95)), 2)
        if "dE" in fails: fails.remove("dE")
        if res["dE_mean"] >= THRESH["dE"]: fails.append("dE")
        if patch is None: ys, xs = np.nonzero(core); patch = (int(ys.mean()), int(ys.mean())+1, int(xs.mean()), int(xs.mean())+1) if core.any() else None
        patch_metric = False
    else:
        patch_metric = patch is not None
    if patch_metric:
        Y0, Y1, X0, X1 = patch; res["removed_dE"] = round(float(dm[Y0:Y1, X0:X1].mean()), 3)
        if res["removed_dE"] >= THRESH["removed_dE"]: fails.append("removed")
    if out.shape != o.shape: fails.append("size")
    if kind == "info": res["info"] = "nie bramkowane"
    res["fails"] = fails
    # sheet: the most edge-rich 400x300 window inside the AI frame
    g = cv2.Canny(cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY), 60, 150).astype(np.float32)*inside
    e = cv2.boxFilter(g, -1, (301, 301))
    if patch is not None: cy, cx = (patch[0]+patch[1])//2, (patch[2]+patch[3])//2
    else: cy, cx = np.unravel_index(np.argmax(e), e.shape)
    y0 = int(np.clip(cy-150, 0, H-300)); x0 = int(np.clip(cx-200, 0, W-400))
    crop = lambda a: Image.fromarray(np.ascontiguousarray(a[y0:y0+300, x0:x0+400]))
    tiles = [label(crop(a), t) for a, t in ((o, "oryginał 100%"), (naive, "AI powiększone"), (out, "wynik"), (gt, "wzorzec (GT)"))]
    hm = cv2.applyColorMap(np.clip(dm*40, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    tiles.append(label(crop(hm), "ΔE2000 ×40")); tiles.append(label(crop(cv2.cvtColor((cv2.resize(prep.changed, (W, H))*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)), "zmiany AI"))
    grid(tiles, 3).save(outdir/f"transfer_{name}.jpg", quality=90)
    print(f"  {name:<28} " + " ".join(f"{k}={v}" for k, v in res.items() if k not in ("case",)), flush=True)
    return res

def _full_maps(al, H, W):
    """full-res pixel -> AI pixel maps (affine + flow), for the naive upscale."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    wx, wy = al.sx*(xx+0.5)-0.5, al.sy*(yy+0.5)-0.5
    if al.flow is not None:
        wx = wx + cv2.remap(al.flow[..., 0], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        wy = wy + cv2.remap(al.flow[..., 1], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    from retusz.engine.align import apply_map
    return apply_map(al.M.astype(np.float32), wx, wy)

def _pts(true_fn, prep, o):
    H, W = o.shape[:2]; ys, xs = np.meshgrid(np.linspace(80, H-80, 24), np.linspace(80, W-80, 24), indexing="ij")
    P = np.c_[xs.ravel(), ys.ravel()]
    a, b = true_fn(P), prep.al.full_to_ai(P)
    ha, wa = prep.al.ai_wh[1], prep.al.ai_wh[0]
    ok = (a[:, 0] > 3) & (a[:, 0] < wa-4) & (a[:, 1] > 3) & (a[:, 1] < ha-4)
    return a[ok], b[ok]

def generative_case(tag, o, outdir):
    rng = np.random.default_rng(5)                        # generative re-render of a grainy original
    og = np.clip(o.astype(np.float32) + rng.normal(0, 8, o.shape), 0, 255).astype(np.uint8); lit = relight(og)
    h, w = o.shape[:2]; f = 1.7*max(1.0, max(h, w)/2600)
    ai, ch = make_generative(lit, (round(w/f), round(h/f)), shift=(6, -5), zoom=1.04, drift_amp=0.004,
                             island=(0.58, 0.40, 0.14, 0.005, -0.003), regen=((0.08, 0.10, 0.33, 0.38),), fine=True)
    gt = strong_grade(lit)
    # reference: the same strong (non-affine: hue rotation, saturation) grade delivered as a plain downscaled copy;
    # the generative re-render must not cost sharpness relative to it
    plain, _ = make_ai(gt, f, 0.0, (0, 0), 0.0)
    pr = TR.prepare(og, plain); ref = TR.render(pr, og, TR.TransferParams())
    return run_case(f"{tag}_generatywny", og, gt, ai, None, outdir, kind="generative", changed_ai=ch, ref_out=ref)

def cases_for(tag, o, mask, outdir, hard_mask=None):
    gt = grade(o, mask); res = []
    for k, ang, sh, cr, size in ((2, 1.5, (7.3, -4.6), 0.04, None), (3, -1.0, (-3.2, 2.5), 0.06, None), (4, 0.5, (2.1, 1.3), 0.0, "square")):
        sz = None
        if size == "square": s = round(min(o.shape[:2])/k*0.92); sz = (s, s)
        ai, T = make_ai(gt, k, ang, sh, cr, size=sz)
        res.append(run_case(f"{tag}_k{k}{'_kwadrat' if sz else ''}", o, gt, ai, T, outdir))
    ai, T = make_ai(gt, 3, 1.0, (0, 0), 0.05); ai = drift(ai)
    res.append(run_case(f"{tag}_k3_dryf", o, gt, ai, lambda P, T=T: drift_true(T, P), outdir))
    gt0 = grade(o, None); ai, T = make_ai(gt0, 3, 0.0, (0, 0), 0.0, jpeg=95)
    h, w = ai.shape[:2]; y0, x0 = int(h*0.3), int(w*0.25); y1, x1 = y0 + h//8, x0 + w//8
    ai[y0:y1, x0:x1] = ai[max(0, y0-20):y0-10, x0:x1].reshape(-1, 3).mean(0).astype(np.uint8)
    res.append(run_case(f"{tag}_k3_usuniety_obiekt", o, gt0, ai, T, outdir, patch=(y0*3+15, y1*3-15, x0*3+15, x1*3-15)))
    ai = cv2.resize(o, (o.shape[1]//3, o.shape[0]//3), interpolation=cv2.INTER_AREA)
    res.append(run_case(f"{tag}_tozsamosc", o, o, ai, None, outdir, kind="identity"))
    res.append(generative_case(tag, o, outdir))
    if hard_mask is not None:                             # stress: mask edge cutting through smooth areas
        gth = grade(o, hard_mask); ai, T = make_ai(gth, 3, -1.0, (-3.2, 2.5), 0.06)
        res.append(run_case(f"{tag}_k3_twarda_maska", o, gth, ai, T, outdir, kind="info"))
    return res

def file_safety(outdir):
    """End to end through files: output size == original, inputs byte-identical afterwards."""
    d = Path(tempfile.mkdtemp()); o, disc = textured(1200, 1800, seed=31); ai, _ = make_ai(grade(o, disc), 3)
    Image.fromarray(o).save(d/"o.jpg", quality=96); Image.fromarray(ai).save(d/"o_ai.png")
    h = [hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ("o.jpg", "o_ai.png")]
    GR.run(d/"o.jpg", d/"o_ai.png", d/"r.jpg", TR.TransferParams())
    ok = Image.open(d/"r.jpg").size == (1800, 1200) and h == [hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ("o.jpg", "o_ai.png")]
    print("  pliki: rozmiar wyniku = oryginał, wejścia nietknięte:", "OK" if ok else "FAIL"); return ok

def main(argv):
    outdir = Path(argv[0]); outdir.mkdir(parents=True, exist_ok=True); results = []
    print("syntetyczne 4500x3000:")
    o, disc = textured(3000, 4500, seed=101); results += cases_for("synt", o, disc, outdir)
    print("syntetyczny portret 2000x3000:")
    o, _ = portrait_scene(3000, 2000, seed=3); results.append(generative_case("portret", o, outdir))
    for p in map(Path, argv[1:]):
        o, _ = imageio.load(p); print(f"{p.name} {o.shape[1]}x{o.shape[0]}:")
        results += cases_for(p.stem, o, content_mask(o), outdir, content_mask(o, hard=True))
    ok = file_safety(outdir) and all(not r["fails"] for r in results)
    (outdir/"transfer_gate.json").write_text(json.dumps({"thresholds": THRESH, "results": results}, indent=1, default=float), encoding="utf-8")
    print("\nTRANSFER GATE:", "PASS" if ok else "FAIL", "· progi", THRESH)
    return 0 if ok else 1

if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
