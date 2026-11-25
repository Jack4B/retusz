"""Dodge & Burn / Plastyka 3D quality gate (run via: quality_gate.py --dnb OUTDIR [photo.jpg ...]).

Cases: a synthetic portrait (repo-safe, known blotch pattern, nose shadow, glasses) at 1800x1200 and 4500x3000,
plus any photos given (faces/landmarks from YuNet, subject mask from the depth model; photos stay outside the repo).
For D&B alone at 25/50/75/100 %, new (v2, guided filter) vs old (0.1.5 v1) and the comparison candidates:
  skin  - blotch left (synthetic: slope against the known pattern; photos: blotch-band variance ratio),
          fine texture kept, large-scale modelling kept, nose-shadow step kept (synthetic), monotonic,
          |dL*| in a ring outside the skin and on glasses/eyes/lips, saturation C*/(L*+16) where darkened,
          a*b* change, new clipping;
For the look (D&B + Plastyka 3D at the slider value), new vs old:
  halo  - outside: |mean dL* in 0-3 % of the width outside the person - 6-12 % outside|; inside: a narrow line along
          the outline (local deviation of dL* in the 0-3 % band over the 3-6 % band); background change far away;
  flat  - RMS L* contrast per band (fine sigma 1-2 px, mid W/250-W/60, coarse W/30-W/15) on the whole image and the
          person (out/in), and person/background separation |L person - L background ring| (out/in);
  preview (1/4) vs export dE < 1.
Sheets (100 % crops): old vs new at 0/50/100 %, Plastyka edge, and gauss / guided / WLS at 100 %."""
import sys, json, time
from pathlib import Path
import numpy as np, cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"tests")); sys.path.insert(0, str(ROOT/"tools"))
from synthetic import portrait_scene
from retusz import imageio
from retusz.engine import dnb, color as C, faces as F, relight as RL, render as R
from dnb_v1 import dodge_burn_v1, relight_gain_v1
from quality_gate import grid, label

THRESH = dict(blotch_left=0.6, hf=0.93, model_lp=0.92, nose=0.62, ring_dL=1.0, feat_dL=1.0, sat=1.01, ab_dE=2.0,
              clip_add_pct=0.05, halo_out=1.0, halo_in=0.5, bg_far=0.5, contrast=0.9, separation=0.95, prev_export_dE=1.0)
STRENGTHS = (0.25, 0.5, 0.75, 1.0)
METHODS = ("gauss", "guided", "wls")

def Lab(x): return C.to_lab(x)
def Lstar(x): return Lab(x)[..., 0]

def nblur(x, w, s): return cv2.GaussianBlur(x*w, (0, 0), s)/(cv2.GaussianBlur(w, (0, 0), s) + 1e-6)

def ring(mask, lo, hi, outside=True):
    p = (mask > 0.5).astype(np.uint8); d = cv2.distanceTransform(1 - p if outside else p, cv2.DIST_L2, 5)
    return (d > lo) & (d <= hi)

# ---------------------------------------------------------------- cases
def synthetic_case(h, w, seed=4):
    rgb, info = portrait_scene(h, w, seed=seed, grain=2)
    return dict(tag=f"syntetyczny_{w}x{h}", rgb=rgb, faces=[info["face"]], lms=[info["lms"]], fw=info["fw"],
                skin_true=info["skin"], features=info["features"], blotch=info["blotch"], nose=info["nose_shadow"],
                person=info["person"])

def photo_case(path):
    rgb, _ = imageio.load(path)
    if rgb.dtype != np.uint8: rgb = (rgb.astype(np.float32)/257).astype(np.uint8)
    s = C.small(rgb, 1280); k = rgb.shape[1]/s.shape[1]
    fs, lms = F.pick_faces(s, landmarks=True)
    if not fs: print(f"  {path.name}: brak twarzy, pomijam"); return None
    return dict(tag=Path(path).stem, rgb=rgb, faces=[tuple(f*k) for f in fs], lms=[l*k for l in lms],
                fw=float(np.median([f[2]*k for f in fs])), skin_true=None, features=None, blotch=None, nose=None, person=None)

def prepare(case):
    rgb = case["rgb"]; H, W = rgb.shape[:2]
    s = C.small(rgb, 400); k = s.shape[1]/W
    case["skin_q"] = cv2.resize(F.skin_mask(s, [tuple(np.asarray(f)*k) for f in case["faces"]]), (max(2, W//4), max(2, H//4)))
    a = C.small(rgb, 1200); ka = a.shape[1]/W
    skin_a = cv2.resize(case["skin_q"], (a.shape[1], a.shape[0]))
    case["rl"] = RL.Relight(a, skin_a, [np.asarray(f, np.float32)/W for f in case["faces"]])
    case["skin_a"] = skin_a
    if case["person"] is None:                               # photos: the depth model's subject, snapped to the image
        fg = cv2.resize(case["rl"].fg, (W, H)); case["person"] = (fg > 0.5).astype(np.float32)
    sk = cv2.resize(case["skin_q"], (W, H))
    case["skin"] = case["skin_true"] if case["skin_true"] is not None else (sk > 0.5).astype(np.float32)
    return case

# ---------------------------------------------------------------- operators
def apply_dnb(case, s, algo):
    if algo == "v1": return dodge_burn_v1(case["rgb"], case["skin_q"], case["faces"], s)
    return dnb.dodge_burn_v2(case["rgb"], case["skin_q"], case["faces"], case["lms"], s, method=algo)

def apply_plastic(rgb, case, s, old, angle=120.0):
    H, W = rgb.shape[:2]; qh, qw = max(2, H//4), max(2, W//4)
    rgb_q = cv2.resize(rgb, (qw, qh), interpolation=cv2.INTER_AREA)
    Lq = Lstar(rgb_q)
    g = relight_gain_v1(case["rl"], case["skin_a"], qh, qw, Lq, s, angle) if old else case["rl"].gain(qh, qw, Lq, s, angle, rgb_q=rgb_q)
    up = R._bilinear_up(g.astype(np.float32), H, W)[..., None]
    return np.clip(rgb.astype(np.float32)*(1 + up) + 0.5, 0, 255).astype(np.uint8)

def look(case, s, old):
    out = apply_dnb(case, s, "v1" if old else "guided")
    return apply_plastic(out, case, s, old)

# ---------------------------------------------------------------- metrics
def skin_metrics(case, out):
    rgb = case["rgb"]; fw = case["fw"]; L0, L1 = Lstar(rgb), Lstar(out)
    w = (case["skin"] > 0.5).astype(np.float32)
    core = cv2.erode((case["skin"] > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if case["skin_true"] is None:                              # photos: clean skin = skin without strong features
        lev = nblur(L0, w, fw/14); core &= np.abs(L0 - lev) < 10
    band = lambda L: nblur(L, w, fw/80) - nblur(L, w, fw/12)
    r = {}
    if case["blotch"] is not None:
        bt = band(case["blotch"].astype(np.float32))[core]
        sl = lambda L: float(np.dot(band(L)[core], bt)/np.dot(bt, bt)); r["blotch_left"] = sl(L1)/sl(L0)
    else:
        r["blotch_left"] = float(np.sqrt(band(L1)[core].var()/max(band(L0)[core].var(), 1e-9)))
    hp = lambda L: L - cv2.GaussianBlur(L, (0, 0), 1.2); lp = lambda L: cv2.GaussianBlur(L, (0, 0), fw/6)
    r["hf"] = float(hp(L1)[core].std()/hp(L0)[core].std()); r["model_lp"] = float(lp(L1)[core].std()/lp(L0)[core].std())
    if case["nose"] is not None:
        ns = case["nose"]; ins = core & (ns > 0.9); o_ = core & (cv2.dilate((ns > 0.05).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (ns < 0.05)
        st = lambda L: L[o_].mean() - L[ins].mean(); r["nose"] = float(st(L1)/st(L0))
    dL = np.abs(L1 - L0); sk = (case["skin"] > 0.5).astype(np.uint8)
    feat = case["features"] > 0.5 if case["features"] is not None else np.zeros_like(core)
    rg = ring(sk, 0, fw/8) & ~feat
    r["ring_dL"] = float(dL[rg].mean()); r["feat_dL"] = float(dL[feat].mean()) if feat.any() else 0.0
    a, b = Lab(rgb), Lab(out); dark = core & (b[..., 0] < a[..., 0] - 1)
    sat = lambda lab: np.hypot(lab[..., 1], lab[..., 2])/(lab[..., 0] + 16)
    r["sat"] = float(sat(b)[dark].mean()/sat(a)[dark].mean()) if dark.sum() > 50 else 1.0
    r["ab_dE"] = float(np.linalg.norm(b[..., 1:] - a[..., 1:], axis=2)[w > 0].mean())
    clip = lambda x: float((x.max(2) >= 254).mean()*100); r["clip_add_pct"] = clip(out) - clip(rgb)
    r["max_dL"] = float(dL.max())
    return r

def look_metrics(case, out):
    rgb = case["rgb"]; H, W = rgb.shape[:2]; L0, L1 = Lstar(rgb), Lstar(out); dL = L1 - L0
    p = case["person"]
    no, fo = ring(p, 0, 0.03*W), ring(p, 0.06*W, 0.12*W)
    far = ring(p, 0.15*W, 10*W)
    # inside, Plastyka 3D shades the body on purpose (a broad ramp towards the edges); a halo is a narrow line along
    # the outline (bright on one side, dark on another: a ring average cancels it). Local measure: |dL - dL smoothed
    # inside the person (sigma 1.5 % of the width)| in the 0-3 % band minus the same in the 3-6 % band.
    wp = (p > 0.5).astype(np.float32); dev = np.abs(dL - nblur(dL, wp, 0.015*W))
    rim = float(dev[ring(p, 0, 0.03*W, False)].mean() - dev[ring(p, 0.03*W, 0.06*W, False)].mean())
    r = dict(halo_out=float(abs(dL[no].mean() - dL[fo].mean())), halo_in=rim,
             bg_far=float(abs(dL[far].mean())) if far.any() else 0.0)
    bands = dict(fine=(1, 2), mid=(W/250, W/60), coarse=(W/30, W/15))
    pm = p > 0.5
    for k, (a, b) in bands.items():
        bd = lambda L: cv2.GaussianBlur(L, (0, 0), a) - cv2.GaussianBlur(L, (0, 0), b)
        b0, b1 = bd(L0), bd(L1)
        r[f"contrast_{k}"] = float(b1.std()/b0.std()); r[f"contrast_{k}_person"] = float(b1[pm].std()/b0[pm].std())
    bgr = ring(p, 0, 0.12*W)
    sep = lambda L: abs(L[pm].mean() - L[bgr].mean()); r["separation"] = float(sep(L1)/max(sep(L0), 1e-6))
    return r

def prev_export(case, s):
    full = look(case, s, False); H, W = case["rgb"].shape[:2]; w, h = W//4, H//4
    small = dict(case); small["rgb"] = cv2.resize(case["rgb"], (w, h), interpolation=cv2.INTER_AREA)
    small["faces"] = [tuple(np.asarray(f)/4) for f in case["faces"]]; small["lms"] = [l/4 for l in case["lms"]]
    small["skin_q"] = cv2.resize(case["skin_q"], (max(2, w//4), max(2, h//4)), interpolation=cv2.INTER_AREA)
    prev = look(small, s, False)
    ref = cv2.resize(full, (w, h), interpolation=cv2.INTER_AREA)
    return float(np.linalg.norm(Lab(prev) - Lab(ref), axis=2).mean())

def check(r, keys):
    fails = []
    for k in keys:
        if k not in r: continue
        v, t = r[k], THRESH[k if not k.startswith("contrast") else "contrast"]
        bad = v < t if k in ("hf", "model_lp", "nose", "separation") or k.startswith("contrast") else (v > t)
        if bad: fails.append(k)
    return fails

# ---------------------------------------------------------------- sheets
def crop_face(case, img, side=None):
    x, y, w, h = [int(v) for v in case["faces"][0]]; H, W = img.shape[:2]
    cw, ch = int(1.3*w), int(1.05*h); cx, cy = x + w//2, y + int(0.55*h)
    x0 = int(np.clip(cx - cw//2, 0, W - cw)); y0 = int(np.clip(cy - ch//2, 0, H - ch))
    c = img[y0:y0+ch, x0:x0+cw]
    return Image.fromarray(np.ascontiguousarray(c)).resize((420, int(420*ch/cw))) if side is None else Image.fromarray(c)

def crop_edge(case, img, size=(420, 320)):
    """100 % crop on the person/background edge next to the face (hair / shoulder)."""
    x, y, w, h = [int(v) for v in case["faces"][0]]; H, W = img.shape[:2]
    cx = min(W - size[0]//2, x + w + int(0.25*w)); cy = y + int(0.3*h)
    x0 = int(np.clip(cx - size[0]//2, 0, W - size[0])); y0 = int(np.clip(cy - size[1]//2, 0, H - size[1]))
    return Image.fromarray(np.ascontiguousarray(img[y0:y0+size[1], x0:x0+size[0]]))

def dmap(a, b, scale=10):
    d = Lstar(b) - Lstar(a)
    v = np.clip(128 + d*scale, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(v, cv2.COLORMAP_COOLWARM if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET)[..., ::-1]

def sheets(case, outdir):
    t = case["tag"]; rgb = case["rgb"]
    tiles = []
    for old in (True, False):
        for s in (0.0, 0.5, 1.0):
            o = rgb if s == 0 else apply_dnb(case, s, "v1" if old else "guided")
            tiles.append(label(crop_face(case, o), f"D&B {'0.1.5' if old else 'v2'} {int(s*100)}%"))
    grid(tiles, 3).save(outdir/f"dnb_{t}_stary_nowy.jpg", quality=92)
    tiles = []
    for m in METHODS:
        o = apply_dnb(case, 1.0, m)
        tiles.append(label(crop_face(case, o, 1), f"{m} 100%")); tiles.append(label(crop_face(case, dmap(rgb, o), 1), f"{m}: ΔL* (±12)"))
    tiles = [im.resize((420, int(420*im.height/im.width))) for im in tiles]
    grid([label(crop_face(case, rgb), "oryginał")] + tiles, 3).save(outdir/f"dnb_{t}_algorytmy_100.jpg", quality=92)
    tiles = []
    for old in (True, False):
        for s in (0.0, 0.5, 1.0):
            o = rgb if s == 0 else look(case, s, old)
            tiles.append(label(crop_edge(case, o), f"D&B+Plastyka {'0.1.5' if old else 'nowa'} {int(s*100)}%"))
    grid(tiles, 3).save(outdir/f"plastyka_{t}_krawedz_100.jpg", quality=92)
    small = lambda a: Image.fromarray(C.small(a, 520))
    grid([label(small(rgb), "oryginał"), label(small(look(case, 1.0, True)), "0.1.5: D&B+Plastyka 100%"),
          label(small(look(case, 1.0, False)), "nowa: D&B+Plastyka 100%")], 3).save(outdir/f"plastyka_{t}_calosc.jpg", quality=90)

# ---------------------------------------------------------------- main
def run_case(case, outdir):
    t0 = time.perf_counter(); prepare(case); res = {"case": case["tag"], "dnb": {}, "look": {}}
    fails = []
    print(f"{case['tag']}:")
    for algo in ("v1",) + METHODS:
        rows = []
        for s in STRENGTHS:
            tt = time.perf_counter(); o = apply_dnb(case, s, algo); dt = time.perf_counter() - tt
            r = skin_metrics(case, o); r["s"] = s; r["time_s"] = dt; rows.append(r)
        mono = all(rows[i+1]["blotch_left"] <= rows[i]["blotch_left"] + 1e-3 for i in range(len(rows)-1))
        f100 = check(rows[-1], ("blotch_left", "hf", "model_lp", "nose", "ring_dL", "feat_dL", "sat", "ab_dE", "clip_add_pct"))
        if not mono: f100.append("monotonic")
        res["dnb"][algo] = {"rows": rows, "fails": f100}
        r = rows[-1]
        print(f"  D&B {algo:<6} 100%: " + " ".join(f"{k}={r[k]:.3f}" for k in r if k not in ("s",)) + f" fails={f100}")
        if algo == "guided": fails += [f"dnb:{x}" for x in f100]
    for old in (True, False):
        nm = "0.1.5" if old else "nowa"; rows = []
        for s in (0.5, 1.0):
            r = look_metrics(case, look(case, s, old)); r["s"] = s; rows.append(r)
        f = check(rows[-1], ("halo_out", "halo_in", "bg_far", "contrast_fine", "contrast_mid", "contrast_coarse", "separation"))
        res["look"][nm] = {"rows": rows, "fails": f}
        r = rows[-1]; print(f"  wygląd {nm:<5} 100%: " + " ".join(f"{k}={r[k]:.3f}" for k in r if k != "s") + f" fails={f}")
        if not old: fails += [f"wygląd:{x}" for x in f]
    pe = prev_export(case, 1.0); res["prev_export_dE"] = pe; print(f"  podgląd vs eksport ΔE={pe:.3f}")
    if pe >= THRESH["prev_export_dE"]: fails.append("podgląd≠eksport")
    sheets(case, outdir); res["fails"] = fails; res["time_s"] = time.perf_counter() - t0
    return res

def main(argv):
    outdir = Path(argv[0]); outdir.mkdir(parents=True, exist_ok=True)
    cases = [synthetic_case(1800, 1200), synthetic_case(4500, 3000, seed=9)]
    cases += [c for c in (photo_case(Path(p)) for p in argv[1:]) if c is not None]
    results = [run_case(c, outdir) for c in cases]
    ok = all(not r["fails"] for r in results)
    (outdir/"dnb_gate.json").write_text(json.dumps({"thresholds": THRESH, "results": results}, indent=1, default=float), encoding="utf-8")
    print("\nD&B GATE:", "PASS" if ok else "FAIL", [(r["case"], r["fails"]) for r in results if r["fails"]])
    return 0 if ok else 1

if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
