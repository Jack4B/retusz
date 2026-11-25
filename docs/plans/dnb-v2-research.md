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
