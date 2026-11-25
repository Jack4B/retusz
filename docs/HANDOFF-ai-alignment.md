# Handoff: "Przenieś grading z AI" fails on real generative portraits (branch `fix/ai-alignment-robust`)

Author: Jack4B. No client photos are in the repo; the real pair lives only in `/workspace/private-tests/` (box,
gitignored) and on the PC in `D:\AplikacjaRetusz\private_pairs\` (copies; originals under
`D:\Zdjęcia 2026\21.11.2026 - fogel\print\{mateusz_N.jpg, book\book_N.jpg}` are never touched).

## Report from the user
v0.1.4: original `mateusz_1.jpg` (2938×4401) + AI `book_1.jpg` (1686×2528, Gemini/Nano Banana) shows
"Wersja AI nie pasuje do oryginału (dopasowanie niespójne)".

## Root cause 1 (FIXED in this branch): registration matched on noise
- `align._sift_affine` ran SIFT on LCN(σ=3) luminance. On a grainy camera original vs a smooth regenerated AI image
  that gives few real matches: 276 good matches / 5000-kp cap → 76 → RANSAC (2.5 px) found 20 inliers, the check
  was `n < 20 or n < 0.15*good`, so it passed or failed by RANSAC luck (differs per process → GUI failed).
- Measured on the real pair (inliers at 6 px): LCN3 80 · LCN6 437 · plain luma 774 · CLAHE luma 934.
- Fix: pool matches from CLAHE luminance + LCN(σ=6), ratio 0.8, RANSAC threshold `clip(0.0025*max(ai side),3,8)` px,
  `cv2.setRNGSeed` for determinism, acceptance `n>=12 and n>=0.08*matches` (final judge stays ECC cc>=0.2).
  Real pair now: 1313 inliers, ECC cc 0.856, deterministic. mateusz_13/book_13 also OK.
- Regression test: `tests/test_align.py::test_generative_style_ai_grainy_original_registers`
  (synthetic: grain + `synthetic.make_generative` = strong regrade, other framing/zoom, non-rigid drift, moved island,
  redrawn patch). Helpers added to `tests/synthetic.py`: `smooth_field`, `strong_grade`, `make_generative`.
- `.gitignore`: `private-tests/`, `.private/`.

## Root cause 2 (FIXED in 0.1.5): change map flagged ~80% of a real regenerated image
End-to-end on the real pair after fix 1: align OK (residual 1.47 px full-res, flow kept, coverage 99%), but
`prepare().changed_frac` = **79%**. The result sheet `/workspace/private-tests/sheet_mateusz1_result.jpg`
(orig | AI | result | change map) shows the result ≈ original (almost nothing transferred), because the protected fit
gives changed pixels weight 0.
- Why: `transfer.changed_map` uses local NCC of high-pass luminance at scales (hp σ 1.5, win 4) and (5, 12). A
  generative AI image is smoother and re-draws fine detail/grain, so fine-scale NCC is low everywhere, plus real
  non-rigid drift of 1–3 px. The synthetic gate never had that.
- Ideas, in order: (1) drop the fine scale for generative inputs / measure the AI's detail loss first and blur the
  original to match before NCC; (2) use only (5,12) and coarser (σ 8–16) scales; (3) compare *structure* via
  gradient-direction agreement or edge-aware MS-SSIM on LCN(σ≈6); (4) make `protect` soft: never let changed
  weight exceed ~0.7 and always keep a coarse (cell 48+) global/relight fit from ALL pixels; (5) flag as "changed"
  only compact regions (area opening) — true regenerated content (middle poster, shirt colour, wristwatch) is blobby.
  Target: changed_frac on this pair ≈ 3–10 %, localised on poster/hands/edges, not everywhere.
- Then visually check at 100% for ghosting/double edges and halos around people (user complaint in t61u:
  "poświata dookoła ludzi i płaski obraz" was about dodge&burn, a separate task).

### Fix (0.1.5)
`transfer.changed_map` now compares structure in DoG bands (2, 5, 12 work px; the finest starts at a 1 px blur),
counts a band only where either side is above NOISE_K=3 x its own noise level (Immerkaer 1996 noise estimate per
image: the grainy original vs the smooth AI copy) and keeps only compact regions (area opening >= (0.016*long side)^2,
closing, holes filled). Synthetic reproduction (`synthetic.portrait_scene` grain 8 + `make_generative(fine=True)` +
`relight`, redrawn poster = 7 %): old map 77.6 %, new 7.2 %, recall on the poster 97 %; tests in
`tests/test_transfer.py` (generative fixture). Registration also tries ECC MOTION_HOMOGRAPHY from the affine fit and
keeps it only when it clearly fits better (`Alignment.projective`). Gate: `--transfer` has `*_generatywny` cases
(changed < 12 %, sharpness >= 0.97 x the same grade on a plain AI copy, ghost NCC < 0.2, own-detail R >= 0.9).
Not yet checked on the real pair (no access to private photos from the cloud box): run the commands below on the PC.

## Residual numbers (real pair, full-res px)
rigid 2.2 → with DIS flow 1.47 (tile phase-correlation metric). Edge overlays show the AI redraws glasses/edges with
~1–2 px offsets, so <0.5 px is not achievable for regenerated detail; transfer uses a σ=3 work-px prefiltered fit, so
~1 px is tolerable. DIS variants tried (pre-blur, patch 12/20/24/32, 1–3 iterations, smoothing 6–20, clamp 4–12): all
within 1.4–2.1; none clearly better than current.

## Commands
```
cd /workspace/retusz-app && source .venv/bin/activate
python -m pytest -q                                   # was 111 passed before; +1 new test
python tools/quality_gate.py --transfer OUT /workspace/retouch-proto/input/photo_a.jpg \
   /workspace/retouch-proto/input/photo_b.jpg /workspace/nef/_GDN0443.NEF   # synthetic-AI gate, must stay green
python -m retusz transfer /workspace/private-tests/mateusz_1.jpg /workspace/private-tests/book_1.jpg --kolor 100 --jasnosc 100
```
Real-pair sheet script: see `/tmp/e2e.py` pattern: `TR.prepare` → `TR.render` → hstack(orig, AI, result, change map).

## Still to do for v0.1.5
1. ~~Fix root cause 2 (above), rerun the gate~~ done: synthetic gate PASS (see docs/plans/grading-transfer.md).
2. Visual check of the real pair at 100% (`crop100_result.png` pattern); also pairs 10–15 from `private_pairs`.
3. ~~Bump version to 0.1.5~~ done; (`retusz/__init__.py`, `pyproject.toml`), push branch, CI green, tag from the pushed commit
   (push directly; no GitHub web merge, it leaks the account email), release asset via CI.
4. PC: `git pull`, `PyInstaller retusz.spec`, run transfer on `D:\AplikacjaRetusz\private_pairs\mateusz_1.jpg` +
   `book_1.jpg` writing only to a `Retusz\` subfolder.
