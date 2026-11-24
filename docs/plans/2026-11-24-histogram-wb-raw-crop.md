# Retusz 0.1.3 – histogram, balans bieli z pipety, RAW NEF, kadrowanie i prostowanie – plan

> Implementation should proceed task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four features chosen by Jack4B, in this order: histogram with clipping warning, white-balance eyedropper with temp/tint, Nikon NEF (D750/D700) input, non-destructive crop and straighten.

**Architecture:** Each feature is a small pure module with its own tests (`engine/histo.py`, `engine/wb.py`, `rawio.py`, `engine/geometry.py`); the engine and GUI only call them. White balance is baked into the existing 33³ LUTs (no render cost; preview and export identical by construction). Geometry runs after the colour render on the full frame, at any resolution, from normalised parameters stored in the sidecar, so preview and export use one function. NEF is decoded by rawpy/LibRaw to linear 16-bit sRGB, then exposure and highlight shoulder are applied in float and quantised once to 8-bit for the engine.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Pillow, rawpy 0.28 (MIT) + LibRaw 0.22 (used under CDDL-1.0 of its LGPL-2.1/CDDL dual licence, unmodified), CustomTkinter, pytest.

**Spec:** Jack4B's choice in chat (t52s339 answers) + parent brief of 2026-11-24 22:12 (Europe/Warsaw).

## Global Constraints
- Author/name in code, commits and docs: `Jack4B` only. No private photos in the repo.
- Quality gate must pass before release: preview vs export ΔE < 1, saturation ≤ reference +5%, no seams, no clipping regressions.
- GUI: no layout jumps (fixed-size cells/value boxes), screenshots checked by eye.
- Originals are never modified (NEF read-only; sidecars only).
- Open source only MIT/BSD/Apache (or LGPL/CDDL libraries used unmodified as libraries); no GPL code.
- Polish UI text.

## Review Focus
- RAW+JPEG pairs with the same name in one folder (D750 shoots both) – export names would collide.
- Eyedropper on a clipped or very dark pixel – must refuse, not produce wild WB.
- Rotation of the D&B brush coordinates – strokes painted on a straightened view must land on the same spot in export.
- Crop aspect after rotation and switching portrait/landscape – crop must stay inside the frame.
- NEF portrait orientation (flip 5/6) – preview, export and thumbnail must all be upright.

## Files
- Create `retusz/engine/histo.py` – histogram + clipping stats/masks/overlay.
- Create `retusz/engine/wb.py` – temp/tint → linear gains, apply on LUT, eyedropper solve.
- Create `retusz/rawio.py` – NEF/NRW decode (full/half), thumbnail, EXIF, size.
- Create `retusz/engine/geometry.py` – straighten scale, apply(rgb, angle, crop), point maps, aspect presets, horizon angle.
- Modify `retusz/engine/pipeline.py` – Params: `temp`, `tint`, `angle`, `crop`, `aspect`; WB in `_bake_one`; `colour_key`.
- Modify `retusz/imageio.py` – dispatch NEF; list NEF; skip same-stem JPG when a NEF exists.
- Modify `retusz/batch.py` – geometry after render (image + skin mask).
- Modify `retusz/sidecar.py` – crop/angle are per photo; `Series.sync_fields()` for WB.
- Modify `retusz/gui/app.py` – histogram widget + clip overlay, WB row + eyedropper, tool tabs Pędzel/Kadr, crop overlay/handles, horizon tool, keys W/C/J/Enter/Esc.
- Create `retusz/gui/histview.py` – histogram canvas.
- Modify `tools/quality_gate.py` – WB/crop/NEF preview-vs-export cases.
- Modify `tools/gui_check.py` – screenshots of crop mode, histogram/clip overlay, eyedropper.
- Modify `retusz.spec`, `requirements*.txt`, `THIRD_PARTY_LICENSES.md`, `README.md`, version 0.1.3.
- Create `docs/android-architecture.md` – notes only.

## Task 1: Histogram + clipping (engine/histo.py)
- [ ] Test: `histogram(rgb)` returns r,g,b,l arrays of 256 bins summing to 1; `clipping(rgb)` on an image with 10% pixels at 255 and 5% at 0 returns hi≈0.10, lo≈0.05; `overlay()` paints clipped highlights red (255,40,40) and shadows blue (40,90,255) and leaves the rest unchanged.
- [ ] Run, see it fail (module missing).
- [ ] Implement: highlights = any channel ≥ 254; shadows = all channels ≤ 2; histogram via `np.bincount`.
- [ ] Run, green. Commit.

## Task 2: White balance (engine/wb.py + Params)
- [ ] Tests: `gains(0,0) == (1,1,1)`; temp>0 raises R/B; luminance (Rec.709) of a neutral stays 1; `solve(patch)` on a blue-cast grey patch then applying the returned deltas makes it neutral (|a|,|b| < 1 LAB); patch with a clipped channel or very dark → `None`; engine render with `temp=0.3` warms a grey image; `Params` clamps temp/tint to [-1,1] and both are in `colour_key`.
- [ ] Run, red.
- [ ] Implement: ln gains R=+0.35·temp, B=−0.35·temp, G=−0.25·tint; normalise by Rec.709 luminance; apply in linear light inside `_bake_one` after the grade, before `wide`. Eyedropper: Δtemp = ln(B/R)/(2·0.35), Δtint = −ln(√(RB)/G)/0.25 on a 7×7 linear mean; refuse if any channel > 250 or max < 12 (8-bit).
- [ ] Green. Commit.

## Task 3: Series sync for WB (sidecar.py)
- [ ] Test: `Series.sync_fields(photos, {"temp":0.2,"tint":-0.1})` updates series defaults and every existing sidecar, leaves strokes/crop of those photos intact, creates no new sidecars.
- [ ] Red → implement → green. Commit.

## Task 4: Geometry (engine/geometry.py)
- [ ] Tests: `apply(rgb, 0, None)` is identity (same object values); crop slice exact at angle 0; `straighten_scale` covers the frame (no black corners after `apply` with angle 7°); `frame_to_src(src_to_frame(p)) == p`; `horizon_angle((0,0),(100,10))` levels the line (rotating the two points by the returned delta gives equal y); `aspect_rect("2:3", w,h, portrait)` gives ratio 2:3 within 1 px, centred, inside [0,1]; `clamp_crop` keeps min size and order; Params clamps `angle` to [-45,45] and `crop`.
- [ ] Red → implement (cv2.getRotationMatrix2D + one warpAffine straight into the crop window; INTER_CUBIC) → green. Commit.

## Task 5: Export path uses geometry (batch.py, sidecar.py)
- [ ] Test: export of a photo with angle 5° and crop 2:3 has the crop's pixel size (±1) and matches `geometry.apply` of the full render; skin mask transformed too; series defaults never store crop/angle (like strokes).
- [ ] Red → implement → green. Commit.

## Task 6: NEF input (rawio.py, imageio.py)
- [ ] Tests (skip when `RETUSZ_TEST_NEF` sample absent; synthetic tests run always with a fake rawpy object): `imageio.load(nef)` returns uint8 HxWx3 upright, meta icc = sRGB profile, EXIF has Model and DateTimeOriginal, no strip offsets; `load_preview(nef, 1600)` matches the downscaled full decode with ΔE76 mean < 1; `image_size` equals full decode size; `list_images` lists `.NEF` and hides `X.JPG` when `X.NEF` exists; thumbnail from the embedded JPEG is upright; batch export of NEF writes JPG.
- [ ] Red → implement: `postprocess(use_camera_wb=True, output_color=sRGB, output_bps=16, gamma=(1,1), no_auto_bright=True, half_size=…)`; exposure gain from embedded JPEG median luminance (bisection on 512 px, clamp 0.25–8, cached per file+mtime); shoulder above 0.8 linear; sRGB encode; quantise with ±0.5 LSB ordered dither.
- [ ] Green. Commit.

## Task 7: GUI (app.py, histview.py)
- [ ] Histogram (220×72) in the right column, clip triangles light up, click or J toggles the overlay on the result view; label "Cienie x% · Światła y%".
- [ ] Row 4 of the slider grid: Temperatura, Odcień (−100..100), cell spanning 2 columns with "Pipeta (W)" and "Balans bieli → seria".
- [ ] Tool tabs "Pędzel | Kadr" in one fixed-size cell; Kadr: aspect (Dowolny, Oryginał, 1:1, 2:3, 4:5, 16:9), "Pion/Poziom", angle slider −45..45 with fixed value box, "Linia horyzontu", "Resetuj kadr". Crop overlay: dim outside, thirds grid in crop mode, corner handles, drag to move/resize/new.
- [ ] Brush/eyedropper map display (straightened) coordinates back to source via geometry.
- [ ] `tools/gui_check.py`: layout stable with all new widgets, screenshots for crop, histogram+overlay, eyedropper; crop drag round-trip; brush point under rotation.
- [ ] Look at the screenshots. Commit.

## Task 8: Gate, docs, release
- [ ] Gate: add cases (WB temp 0.3, angle 4° + crop 4:5) for preview vs export; NEF sample → reference; must PASS.
- [ ] PyInstaller spec includes rawpy (hidden import + binaries), THIRD_PARTY (rawpy MIT, LibRaw CDDL-1.0, sample files not shipped), README features, version 0.1.3, `docs/android-architecture.md`.
- [ ] Full pytest, gate, tag v0.1.3 → CI release; PC: pull, pytest, build, NEF test copy into `D:\AplikacjaRetusz\test`, gate, timings, shortcut check.
