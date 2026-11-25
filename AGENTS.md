# Retusz – zasady dla współpracowników i agentów AI

Read this before changing anything. (Claude Code: `CLAUDE.md` points here.)

## Project
Retusz is a Windows desktop app for portrait photographers (Polish UI): colour match to a reference
(wzorzec), dodge & burn, Plastyka 3D, batch with per-photo overrides, RAW NEF, crop/straighten, export,
and (0.1.4) transfer of grading from a low-res AI version onto the full-res original.
Stack: Python 3.11+, CustomTkinter, OpenCV, NumPy, numba, onnxruntime(-directml), rawpy, Pillow, PyInstaller, pytest.

## Hard rules
1. **Naming:** the only author/person name allowed in code, comments, commits, docs and releases is `Jack4B`.
   Git identity: `Jack4B <Jack4B@users.noreply.github.com>`.
2. **No private photos** in the repo (not in tests, docs, fixtures or screenshots of real clients).
   Tests use synthetic images generated in code; real photos stay on local disks only.
3. **No Retouch4me** code, models, binaries or extracted data – not decompiled, not copied, not imitated
   from internals. It may only be used as a black box to compare outputs.
4. **Licences:** take ideas from papers and MIT / BSD / Apache-2.0 code only (cite them in comments or
   THIRD_PARTY_LICENSES.md). No GPL/AGPL code (e.g. darktable, color-matcher are off limits).
   LGPL/CDDL libraries only as unmodified dependencies.
5. **Originals are read-only.** The app writes only exports and sidecars (`*.retusz.json`).
6. **UI language is Polish**; keep layout stable (fixed-width value boxes, no jumping widgets).

## How we work
- Plan first (`docs/plans/*.md`), then TDD: write the failing test, watch it fail, implement, run the
  **whole** suite (`python -m pytest -q`).
- Quality gate before merge/release: `python tools/quality_gate.py OUTDIR SRC REF ...` (colour engine) and
  `python tools/quality_gate.py --transfer OUTDIR [photos]` (AI grading transfer). Look at the sheets.
- GUI changes: `xvfb-run python tools/gui_check.py <folder> <reference> <outdir>` and look at screenshots.
- Feature branches + PRs into `main`; small commits; CI `build-windows` builds the zip on tags `v*`.
- Preview and export must use the same code path (preview vs export ΔE < 1).
- Memory: originals can be 24 Mpx+; process full-res data in strips, never 12 float copies of a frame.

## Layout
- `retusz/engine/` – algorithms (colour, faces, relight, render, geometry, wb, histo, align, bgu, transfer)
- `retusz/gui/` – CustomTkinter windows
- `retusz/*.py` – I/O, RAW, colour management, export, batch, sidecars, CLI
- `tools/` – quality gate, GUI check · `tests/` – pytest · `docs/plans/` – implementation plans
