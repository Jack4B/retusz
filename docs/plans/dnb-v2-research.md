# D&B v2 / Plastyka 3D – algorithm study (status: desk study, measurements NOT yet run)

User feedback: 0.1.4 D&B at high slider "wygląda tragicznie"; "poświata dookoła ludzi i płaski obraz".

## Reproduced failure modes (v1, box test portraits, sheets kept privately, not in repo)
| # | Symptom | Root cause (code) |
|---|---------|-------------------|
| 1 | Orange / over-saturated skin when darkening | `color.dodge_burn` lowers L* but keeps a*/b*; LAB→sRGB clips per channel |
| 2 | Dark/bright rims at glasses, eyes, lips, moustache | mask-normalised blurs use `soft mask > 0.3` → dark frame/brow pixels enter the local means |
| 3 | Flat, muddy face at 100 % | band fw/60…fw/8 contains the face's own modelling; 0.8× removed linearly, no per-pixel cap |
| 4 | Glow (halo) around people | `relight.Relight.gain`: `-0.08*(1-fg)` background darkening with a soft, depth-derived `fg` (blurred w/150, DA-V2 edges are blurry/misaligned) + `shade` blurred across the silhouette; quarter-res gain bilinear-upsampled across sharp edges |
| 5 | Global flatness | D&B (3) + clarity on guided base at strength 0.15 only on subject, background darkened → subject/background contrast compressed near edges |

## Candidates (edge-preserving base / decomposition for a ΔL* field)
| Candidate | Halo-free | Texture / flatness | Speed @24 Mpx (field at face=256 px, applied full-res) | Licence |
|---|---|---|---|---|
| Gaussian frequency separation (v1) | poor at strong edges | removes modelling band → flat | fast | n/a |
| Guided filter (He 2010), self-guided | good; gradient-reversal-free | keeps structure edges | O(N), very fast | paper; MIT impls |
| Fast guided filter + guided upsampling (He 2015) | good, edges follow full-res guide | – | apply step O(N) at full res | paper |
| Bilateral / bilateral grid (Paris & Durand 2006, Chen 2007) | ok, some gradient reversal | ok | fast (cv2.bilateralFilter at low res) | paper; OpenCV Apache |
| Domain transform RF (Gastal & Oliveira 2011) | good | good | O(N), needs numba loop (no ximgproc in our cv2) | paper |
| WLS (Farbman 2008) | best (halo-free multiscale) | best for detail/base split | sparse solve; ok only at working res; needs scipy (not a dependency) | paper |
| Local Laplacian (Paris 2011; fast Aubry 2014) | halo-free by construction | best for local contrast w/o flattening | heavy; feasible at working res with ~8 levels | paper |
| Landmark-based contour D&B | n/a (smooth shapes, skin-masked) | adds modelling back | trivial | own |

## Recommendation (to be confirmed by measurement)
- **D&B:** field computed on a crop resampled to face = 256 px (preview == export); blotch map =
  mask-normalised small blur (σ≈fw/80) minus **guided-filter** base (r≈fw/12) using only *clean* skin
  weights (sharp per-pixel CrCb Mahalanobis skin test ∩ face mask, eyes/brows/lips excluded via YuNet
  landmarks); redescending weight x/(1+(x/T)²), T≈10 L*; tanh cap ≈10 L*; + soft landmark contour
  (dodge forehead/nose bridge/cheekbones/chin ≈+5 L*, burn under cheekbones/jaw ≈−6 L* at 100 %).
  Apply as **luminance-only linear-light gain** (chromaticity kept), highlight-protected, in strips;
  final mask **guided-upsampled** with full-res L as guide (He 2015) so nothing leaks past skin edges.
- **Plastyka 3D halo:** remove/limit the background darkening term, refine `fg` with a guided filter
  using the image as guide, compute `shade` with mask-normalised blur inside the subject only, and
  guided-upsample the gain map instead of bilinear. Local Laplacian (Aubry 2014) is the candidate for
  "clarity" if guided clarity still flattens.

## Gate metrics to add (tools/quality_gate.py --dnb)
- Halo: luminance overshoot profile across person/background edge – mean ΔL* in ring 0–3 % of width
  outside the subject minus ring 6–12 % outside (target < 1 L*), same inside.
- Flatness: RMS L* contrast per Laplacian-pyramid band (fine σ1–2, mid σ4–16, coarse σ32–64) on subject
  and whole image, out/in ratio (coarse ≥ 0.9, fine ≥ 0.9).
- Skin: blotch-band variance reduction (50 %/100 %), HF texture ≥ 90 %, ΔE_ab (a/b only) mean < 2,
  no new clipping, monotonic with slider, preview vs export ΔE < 1.

## Measured (0.1.6, cloud box, synthetic portraits only – no client photos)
`python tools/quality_gate.py --dnb OUT [photo …]`; sheets `dnb_*_stary_nowy.jpg`, `dnb_*_algorytmy_100.jpg`,
`plastyka_*_krawedz_100.jpg`, `plastyka_*_calosc.jpg`. Synthetic portrait 1200×1800 (face 360 px, glasses, nose shadow,
blotches ~2.5 L*, pores, grain), D&B at 100 %:

| algorithm | blotch left | nose shadow kept | fine texture | modelling (fw/6) | ring outside skin \|ΔL*\| | glasses/eyes/lips \|ΔL*\| | C*/(L*+16) where darkened | max \|ΔL*\| | time (crop) |
|---|---|---|---|---|---|---|---|---|---|
| 0.1.5 v1 (Gaussian band, a*b* kept) | 0.35 | 0.41 | 1.00 | 1.02 | **4.5** | **13.0** | **+9.6 %** | 33.7 | 0.12 s |
| v2 base = Gaussian | 0.27 | **0.55** | 1.00 | 0.98 | 0.05 | 0.08 | +0.1 % | 6.3 | 0.23 s |
| **v2 base = guided filter (shipped)** | 0.42 | 0.66 | 1.00 | 0.98 | 0.03 | 0.03 | +0.1 % | 6.0 | 0.17 s |
| v2 base = WLS (CG) | 0.60 | 0.77 | 1.00 | 0.98 | 0.02 | 0.04 | +0.1 % | 5.6 | 1.45 s |

Guided vs WLS: WLS keeps real shading edges a bit better but evens blotches less at the same settings and is ~8×
slower (24 Mpx: guided 1.1 s, WLS 3.1 s for the whole D&B step); the Gaussian base flattens the nose shadow. Local
Laplacian was not implemented: at the face=256 px analysis scale it would be affordable, but the guided base already
meets every target and the user chose it. Full-strength look (D&B + Plastyka 3D at 100 %):

| | halo outside (0–3 % vs 6–12 %) | line inside the outline | background far away | contrast fine / mid / coarse | person mid | person/background separation |
|---|---|---|---|---|---|---|
| 0.1.5 | 0.78 L* | 0.67 | **−3.4 L*** | 0.98 / **0.80** / **0.85** | **0.71** | **0.57** |
| 0.1.6 | 0.03 L* | 0.14 | 0.0 | 1.02 / 1.03 / 1.00 | 1.05 | 0.95 |

Preview (1/4) vs export ΔE 0.31. Root causes fixed in Plastyka 3D besides the background term: the depth model's
soft, too-wide outline (now binarised, snapped to image edges, face-seeded so background objects and skin-coloured
background patches drop out), normals/curvature at the depth step (subject-only depth, smooth seam, curvature faded
and shading blended to the broad subject shading in the outer ~3 %).
Open: real photos (glasses, beards, several faces, low key) must be run through the gate on the PC.
