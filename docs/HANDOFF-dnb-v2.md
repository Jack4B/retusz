# HANDOFF – Dodge & Burn v2 (branch feature/dnb-v2)

## Where it stopped
- Done: v1 failure modes reproduced on box portraits (D&B alone at 0/25/50/75/100 %), root causes traced,
  plan `docs/plans/2026-11-25-dnb-v2.md`, study `docs/plans/dnb-v2-research.md`.
- Baseline: `python -m pytest -q` → 111 passed (main 9c6c3b6).
- NOT done: no production code changed yet; candidate measurements and comparison sheet not run; no release.
- Another worker fixes AI-grading alignment on main (v0.1.5). Rebase on origin/main before finishing;
  release as next tag after main's latest (v0.1.6 if v0.1.5 exists). Push with git only (no web merges),
  author `Jack4B <Jack4B@users.noreply.github.com>`.

## Prompt for the next agent (Claude)
Read AGENTS.md, then docs/plans/2026-11-25-dnb-v2.md and docs/plans/dnb-v2-research.md. User complaint:
D&B at high slider looks terrible, glow (halo) around people and a flat image. TDD:
1. `tests/test_dnb.py` on a synthetic face (skin ellipse + mid-frequency blotches + pore noise + dark
   eyes/glasses frame, landmarks given): blotch variance drops, HF texture ≥ 90 %, a/b ΔE < 2, no |ΔL|
   outside skin, monotonic in strength, capped, zero strength = identity, preview (1/4 size) == export.
2. New `retusz/engine/dnb.py` per the research recommendation; wire into `pipeline._dodge_burn_region`
   (keep `color.dodge_burn` as v1 for comparison). YuNet landmarks: extend `faces.detect` (rows 4..13 of
   the YuNet output) and add `Fit.lms_n` (default None, bbox fallback).
3. Plastyka 3D halo: `relight.Relight.gain` – drop/limit `-0.08*(1-fg)`, guided-refine `fg` with the image,
   guided-upsample the gain map.
4. Gate: `tools/quality_gate.py --dnb OUTDIR photos…` with halo profile + multi-scale RMS contrast; sheets
   old vs new at 0/50/100 %, plus a 2–3 candidate comparison sheet at 100 %. Look at the sheets.
5. Full suite, CHANGELOG, version bump, tag → CI build-windows; PC build in D:\AplikacjaRetusz\retusz.
Never commit real photos (private ones live in /workspace/private-tests or D:\AplikacjaRetusz\test).

## Commands
```
git clone https://github.com/Jack4B/retusz && cd retusz && git checkout feature/dnb-v2
pip install -r requirements-dev.txt && python -m pytest -q
# box only: private repro of v1 at 0..100 %:  python /workspace/private-tests/dnb/repro.py OUT tag photo.jpg ...
git fetch origin && git rebase origin/main && git push origin feature/dnb-v2
```
