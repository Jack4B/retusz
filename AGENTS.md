# Retusz – zasady dla współpracowników i agentów AI

Read this before changing anything. (Claude Code: `CLAUDE.md` points here.)

## Project
Retusz is a free MIT-licensed Windows desktop app for portrait photographers (Polish UI): Lightroom Classic
1-column UX, standalone OKLab/OKLCh studio develop, ColorChecker 24 calibration, colour match to single or
pooled references (wzorce), Dodge & Burn v2, 2-axis Plastyka 3D, mixed-light LED skin unification,
pore-preserving healing brush, batch with per-photo overrides, 16-bit TIFF/PNG & ICC export, RAW, and
transfer of grading from a low-res AI version onto the full-res original.
Stack: Python 3.11+ (CustomTkinter, OpenCV + OpenCL `cv2.ocl`, NumPy, numba, onnxruntime(-directml),
rawpy, Pillow, tifffile, PyInstaller, pytest) + native Rust workspace (`wgpu` + `egui` + `rayon`, Inno Setup).

## Hard rules
1. **Naming:** the only author/person name allowed in code, comments, commits, docs and releases is `Jack4B`.
   Git identity: `Jack4B <Jack4B@users.noreply.github.com>`.
2. **No private photos** in the repo (not in tests, docs, fixtures or screenshots of real clients).
   Tests use synthetic images generated in code; real photos stay on local disks only.
3. **No Retouch4me** code, models, binaries or extracted data – not decompiled, not copied, not imitated
   from internals. It may only be used as a black box to compare outputs.
4. **Licences:** take ideas from papers and MIT / BSD / Apache-2.0 code & weights only (cite them with
   SHA-256 in `THIRD_PARTY_LICENSES.md`). No GPL/AGPL/NC code or weights.
   LGPL/CDDL libraries only as unmodified dependencies.
5. **Originals are read-only.** The app writes only exports and sidecars (`*.retusz.json`).
6. **UI language is Polish**; keep layout stable (fixed-width value boxes, no jumping widgets).

## How we work
- Plan first (`docs/plans/*.md`), then TDD: write the failing test, watch it fail, implement, run the
  **whole** suite (`python -m pytest -q` and `cargo test --workspace --release`).
- Quality gate before merge/release: `python tools/quality_gate.py OUTDIR SRC REF ...`,
  `python tools/quality_gate.py --colorchecker OUTDIR`, and `python tools/quality_gate.py --transfer OUTDIR [photos]`.
- GUI changes: `xvfb-run -a -s "-screen 0 1600x1000x24" python tools/gui_check.py <folder> <reference> <outdir>` and look at screenshots.
- Feature branches + PRs into `main`; small commits; CI `build-windows` builds installer & zip on tags `v*`.
- Preview and export must use the same code path (preview vs export ΔE < 1).
- Memory: originals can be 24 Mpx+; process full-res data in strips, never 12 float copies of a frame.

## Layout
- `retusz/engine/` – algorithms (colour, tone, detail, skin_physics, llf, colorchecker, faces, dnb, relight, render, geometry, wb, histo, align, bgu, transfer, device)
- `retusz/gui/` – CustomTkinter windows (`app.py`, `library.py`, `histview.py`, `imageview.py`, `dialogs.py`)
- `retusz/*.py` – I/O, RAW, colour management, export, batch, presets, sidecars, CLI
- `crates/` – native Rust workspace (`retusz-core`, `retusz-gpu`, `retusz-app`)
- `tools/` – quality gate, GUI check · `tests/` – pytest · `docs/plans/` – implementation plans
