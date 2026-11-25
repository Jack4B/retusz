# Przenoszenie gradingu z wersji AI na pełny oryginał – plan

> Implementation should proceed task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Issue: "Przenoszenie gradingu z wersji AI na pełny oryginał" · branch `feature/grading-transfer`.

**Goal:** An AI tool (Gemini / Nano Banana, etc.) returns a regraded and relit copy of a photo at a lower
resolution, possibly rescaled, cropped, shifted or slightly rotated. Retusz aligns that copy over the
full-resolution original (JPG/TIFF/PNG or NEF) and transfers its colour and light, keeping 100 % of the
original's resolution and optical sharpness. This is the Photoshop trick "AI layer in Color mode +
partial Luminosity on a mask", done automatically and precisely.

**Architecture:**
1. **Registration** (`engine/align.py`). SIFT features (OpenCV, Apache-2.0) on local-contrast-normalised
   luminance (robust to the AI's new grading and relighting) + RANSAC affine (handles scale, crop,
   shift, small rotation, slight aspect change) → ECC refinement (`cv2.findTransformECC`, MOTION_AFFINE,
   Evangelidis & Psarakis 2008) on the same normalised images → optional dense residual correction with
   DIS optical flow (OpenCV), smoothed and clamped to a few pixels, for AI generators that redraw the
   picture with small local drifts. Residual error is measured by tile phase correlation and reported in
   full-resolution pixels.
2. **Transfer** (`engine/bgu.py`, `engine/transfer.py`). Bilateral Guided Upsampling, fast local variant
   (Chen, Adams, Wadhwa, Hasinoff, SIGGRAPH Asia 2016; reference code google/bgu is Apache-2.0 – we
   implement from the paper, nothing copied). At the AI's resolution we fit, in a bilateral grid
   (x, y, luminance bins), 3×4 affine colour matrices that map *original → AI*. Their sufficient
   statistics are blurred in the grid (smoothness, no false edges), regularised towards one global
   affine fit, and solved per cell. The full-resolution original is then "sliced" through the grid
   (trilinear in x, y, luminance; numba kernel, strips → bounded memory). Because the output is
   `A(x,y,L)·original + b`, every pixel of fine detail comes from the original; only the smooth, edge-aware
   colour/tone transform comes from the AI. Luminance bins separate sky from subject, so the transform
   does not bleed across edges (no halos) – that is why this is chosen over a plain spatial guided filter.
3. **Kolor / Jasność** (`engine/transfer.py`). The full transfer `R` is mixed with the original in CIELAB:
   `a*,b* = O + kolor·(R − O)`, `L* = O + jasność·mask·(R − O)`. Defaults: Kolor 100 %, Jasność 100 %;
   an optional brush mask lowers/raises Jasność locally (the "Luminosity on a mask" part).
4. **Confidence** (`engine/transfer.py`). Two-pass fit: residual `|slice(O_small) − AI|` after pass 1,
   smoothed → map of places where the AI changed content (removed/added objects, redrawn faces). Pass 2
   gives those pixels zero weight, so there the original is graded by the neighbours' / global transform
   instead of producing ghosts. Pixels outside the AI frame (AI crop) get the same fallback. The share
   of changed area is shown as a warning and as an overlay.
5. **I/O** (`grading.py`). AI image converted into the original's ICC space (LittleCMS) before fitting.
   Output has the original's size: JPG (EXIF + ICC kept, no sharpening) or 16-bit TIFF (ICC kept; NEF is
   developed at 16 bit for this path). Originals are never written. Batch: pair files by name
   (`IMG_1.NEF` ↔ `IMG_1_ai.png`, `IMG_1-gemini.jpg`, …).
6. **GUI** (`gui/transfer_view.py`). Window "Przenieś grading z AI": pick Oryginał + Wersja AI, preview
   with modes Wynik / Oryginał / AI / Miganie / Podział, sliders Kolor and Jasność, Pędzel Jasności,
   "Chroń miejsca zmienione przez AI", status line (błąd dopasowania px, zmieniony obszar %), Zapisz
   JPG / TIFF 16-bit, Seria (folder oryginałów + folder AI). Polish UI.

**Tech Stack:** Python 3.11+, NumPy, OpenCV (SIFT, ECC, DIS, warp), numba, Pillow ImageCms, tifffile (BSD),
rawpy, CustomTkinter, pytest.

**Spec:** Jack4B's request t53u (chat, 2026-11-25) + parent brief of 2026-11-25 10:57 (Europe/Warsaw).

## Global Constraints
- Name in code/commits/docs: `Jack4B` only. No private photos in the repo (tests use synthetic images).
- Algorithms from papers / MIT-BSD-Apache code only; no GPL code; nothing from Retouch4me.
- Originals are read-only. Output size == original size.
- TDD: failing test first; whole suite green before every push; quality gate green before release.
- Polish UI text; layout must not jump when sliders move.

## Quality gate (`tools/quality_gate.py --transfer OUTDIR [photo ...]`)
Synthetic cases built from a full-resolution image (procedural in tests; any photo / NEF on the box):
ground truth `GT = grade(original)` (tone curve + colour matrix + smooth relight field + one sharp-edged
local grade), AI = `GT` downscaled 2–4×, cropped, shifted, rotated ≤ 2°, JPEG q90; one case with a
non-rigid 1–2 px drift; one with a removed object; identity case AI = downscaled original.

| Metric | Threshold |
|---|---|
| alignment error vs the true transform, full-res px (mean over the frame) | < 0.5 |
| ΔE2000 mean, result vs GT (inside the AI frame) | < 2.0 |
| sharpness: high-frequency energy (Laplacian variance) result / GT | ≥ 0.95 |
| naive AI upscale sharpness / GT (sanity, must be clearly worse) | reported |
| halo: ΔE2000 mean in a 12 px band along strong edges | < 3.0 |
| identity: ΔE2000 mean result vs original | < 0.6 |
| changed-content case: ΔE2000 of the removed-object area vs graded original (no ghost) | < 4.0 |
| output size == original size; original file bytes unchanged | exact |

Sheets: `original 100 % crop | naive AI upscale | result` (+ GT) written to `/workspace/notes/transfer-gate/`.

## Review Focus
- AI image with a different aspect ratio (square crop of a 3:2 photo) – alignment must still converge.
- AI image much smaller (1024 px vs 6000 px) – 6× scale; working size must follow the AI.
- AI image unrelated to the original – must refuse with a clear message, not produce garbage.
- Portrait NEF (orientation flip) – both sides upright before matching.
- 24 Mpx originals – peak memory bounded (strips), no float copies of the full frame per coefficient.

## Files
- Create `retusz/engine/align.py` – `register(orig_small, ai) -> Alignment` (affine 2×3 orig→AI, optional flow, residual px).
- Create `retusz/engine/bgu.py` – `fit(guide, target, weight, cell, bins, lam) -> Grid`, `slice_(img, grid, scale) -> img` (numba + numpy fallback).
- Create `retusz/engine/transfer.py` – `TransferParams`, `prepare(orig_full_float, ai_rgb) -> Prepared`, `render(prepared, orig, params, mask=None) -> float img`, `confidence`.
- Create `retusz/grading.py` – load pair (ICC), run, save JPG / TIFF16, `pair_files(orig_dir, ai_dir)`, batch.
- Create `retusz/gui/transfer_view.py`; modify `retusz/gui/app.py` (button "Grading z AI").
- Modify `retusz/cli.py`, `retusz/__main__.py` – `python -m retusz transfer ORIG AI [--out] [--kolor] [--jasnosc] [--tiff]`.
- Modify `retusz/rawio.py` – `load16(path)` (developed 16-bit sRGB).
- Modify `tools/quality_gate.py` (+ `tools/transfer_gate.py`), `tools/gui_check.py` (screenshots of the new window).
- Tests: `tests/test_align.py`, `tests/test_bgu.py`, `tests/test_transfer.py`, `tests/test_grading_io.py`.

## Tasks
- [ ] **T1 Registration** – tests: known affine (scale 1/3, rot 1.5°, shift, crop) recovered with < 0.5 px full-res corner error on a graded+relit target; unrelated image → `AlignmentError`; non-square aspect OK.
- [ ] **T2 BGU fit/slice** – tests: global affine colour change reproduced exactly (ΔE < 0.3); slice keeps full-res detail (HF ratio ≥ 0.97); numba == numpy fallback; constant region stable (no noise gain).
- [ ] **T3 Transfer + Kolor/Jasność + confidence** – tests: kolor=0,jasność=0 → original; kolor only → L* unchanged; identity; removed object → no ghost; mask scales Jasność.
- [ ] **T4 I/O + batch + CLI** – tests: output size == original, EXIF/ICC copied, original bytes unchanged, 16-bit TIFF round-trip, AdobeRGB original + sRGB AI handled, `pair_files` name rules, NEF path (if `RETUSZ_TEST_NEF`).
- [ ] **T5 Quality gate + sheets** – `tools/quality_gate.py --transfer`; all thresholds above pass on synthetic + box photos + NEF; sheets checked by eye.
- [ ] **T6 GUI** – window, preview modes, sliders, brush, warnings, save/batch; `tools/gui_check.py` screenshots; layout stable.
- [ ] **T7 Release** – version 0.1.4, README section, THIRD_PARTY_LICENSES (tifffile), CI build-windows green, tag v0.1.4 with the zip.

## How to run
```bash
python -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
python -m pytest -q                                     # whole suite
python tools/quality_gate.py --transfer /tmp/tg [photo.jpg|photo.NEF ...]   # transfer gate + sheets
python -m retusz transfer oryginal.NEF wersja_ai.png --out wynik.jpg
```
