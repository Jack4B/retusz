"""TDD test suite for Retusz v0.9.1:
1. Skin Fill Light ('Dopełnienie skóry', skin_fill): studio reflector fill on shadowed/midtone skin in OKLCh,
   negative flash-dimming, Auto Ton AI v2 backlit detection, and faceless close-up fallback.
2. Manual AI Mask Brush ('Maska AI: + Dodaj / − Usuń', mask_strokes): non-destructively adds missed skin
   areas and erases unwanted background/clothing areas from mq_raw (updating both overlay & skin edits).
3. Vertical Collapsible Accordion Modules stacked one under another (open_panels, toggle_section).
4. Full Intel iGPU / GPU OpenCL UMat acceleration across blur, boxFilter, bilateralFilter, warpAffine,
   3-channel fast guided filter, chained RL deconvolution, and live ocl_benchmark().
5. Configurable default_view & sticky view mode when stepping across photos with references.
6. Clean standalone branding: zero competitor product names in GUI modules, AGENTS.md, and README.md.

Author: Jack4B.
"""
from __future__ import annotations
from pathlib import Path
import tomllib
import numpy as np
import cv2

import retusz
from retusz import settings as S
from retusz.engine import pipeline as P, tone as TN, device as DEV, geometry as GEO, color as COL, detail as DET
from retusz.gui import app as APP, dialogs as DLG


REPO_ROOT = Path(__file__).resolve().parents[1]


def _backlit_portrait_with_warm_bg_object(h: int = 260, w: int = 340) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(91)
    pores = rng.normal(0.0, 3.8, size=(h, w, 1)).astype(np.float32)

    img = np.full((h, w, 3), [214.0, 224.0, 238.0], np.float32)
    img[60:220, 10:65] = [182.0, 138.0, 112.0]   # skin-like warm wooden door in background
    img[205:, 95:255] = [38.0, 52.0, 78.0]
    img[55:205, 120:230] = np.clip(np.array([96.0, 70.0, 56.0], np.float32) + pores[55:205, 120:230], 0, 255)
    img[62:78, 155:195] = np.clip(np.array([232.0, 204, 188.0], np.float32) + pores[62:78, 155:195] * 0.5, 0, 255)
    img[102:110, 145:157] = [18.0, 16.0, 15.0]
    img[102:110, 193:205] = [18.0, 16.0, 15.0]

    faces_n = [np.array([120.0 / w, 55.0 / w, 110.0 / w, 150.0 / w], np.float32)]
    return img.astype(np.uint8), faces_n


def test_skin_fill_brightens_shadowed_skin_preserving_chroma_highlights_eyes_and_background():
    src, faces_n = _backlit_portrait_with_warm_bg_object()
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(src, None)
    fit.faces_n = faces_n

    p_fill = P.Params(skin_fill=0.80, db=0.0, plastic=0.0, skin_even=0.0, vignette=0.0)
    out = eng.render(src, fit, p_fill, db_enabled=False)

    cheek_before = src[125:175, 138:212]
    cheek_after = out[125:175, 138:212]
    assert float(cheek_after.mean()) > float(cheek_before.mean()) + 22.0

    lab_b = cv2.cvtColor(cheek_before.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    lab_a = cv2.cvtColor(cheek_after.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    assert float(lab_a[..., 0].mean()) > float(lab_b[..., 0].mean()) + 9.0
    c_before = float(np.hypot(lab_b[..., 1], lab_b[..., 2]).mean())
    c_after = float(np.hypot(lab_a[..., 1], lab_a[..., 2]).mean())
    assert c_before * 0.96 <= c_after <= c_before * 1.32, f"Expected healthy skin chroma {c_before:.1f} -> {c_after:.1f}"

    assert np.abs(out[10:45, 250:330].astype(int) - src[10:45, 250:330].astype(int)).max() <= 1
    assert np.abs(out[80:200, 15:58].astype(int) - src[80:200, 15:58].astype(int)).max() <= 1

    pupil_before = float(src[104:108, 148:154].mean())
    pupil_after = float(out[104:108, 148:154].mean())
    assert pupil_after < pupil_before + 4.5, f"Pupil should stay dark: {pupil_before:.1f} -> {pupil_after:.1f}"
    assert int(out[64:76, 160:190].max()) <= 248

    hp_b = (cheek_before[..., 1].astype(np.float32) - cv2.GaussianBlur(cheek_before[..., 1].astype(np.float32), (0, 0), 1.6)).std()
    hp_a = (cheek_after[..., 1].astype(np.float32) - cv2.GaussianBlur(cheek_after[..., 1].astype(np.float32), (0, 0), 1.6)).std()
    assert hp_a >= 0.85 * hp_b

    flash_img = np.full((220, 260, 3), [34, 36, 42], np.uint8)
    flash_img[45:175, 75:185] = [226, 188, 168]
    fit_fl = eng.prepare(flash_img, None)
    fit_fl.faces_n = [np.array([75.0 / 260, 45.0 / 260, 110.0 / 260, 130.0 / 260], np.float32)]
    out_calm = eng.render(flash_img, fit_fl, P.Params(skin_fill=-0.80, db=0.0, plastic=0.0, skin_even=0.0, vignette=0.0), db_enabled=False)
    assert float(out_calm[70:150, 95:165].mean()) < float(flash_img[70:150, 95:165].mean()) - 14.0
    assert np.abs(out_calm[10:35, 10:60].astype(int) - flash_img[10:35, 10:60].astype(int)).max() <= 1


def test_skin_fill_auto_tone_backlit_detection_and_faceless_closeup_fallback():
    src, faces_n = _backlit_portrait_with_warm_bg_object()
    auto = TN.auto_tone_params(src, faces_n=faces_n)
    assert auto.get("skin_fill", 0.0) >= 0.20, f"Expected Auto Ton AI v2 to add skin_fill on backlit portrait, got {auto}"

    hand_img = np.full((180, 220, 3), [42, 68, 112], np.uint8)
    hand_img[50:135, 65:155] = [118, 86, 68]
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(hand_img, None)
    fit.faces_n = []
    out_hand = eng.render(hand_img, fit, P.Params(skin_fill=0.75, db=0.0, plastic=0.0, skin_even=0.0, vignette=0.0), db_enabled=False)
    assert float(out_hand[70:115, 85:135].mean()) > float(hand_img[70:115, 85:135].mean()) + 16.0
    assert np.abs(out_hand[10:35, 10:50].astype(int) - hand_img[10:35, 10:50].astype(int)).max() <= 1


def test_ai_mask_manual_add_and_subtract_strokes():
    src, faces_n = _backlit_portrait_with_warm_bg_object()
    H, W = src.shape[:2]
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(src, None)
    fit.faces_n = faces_n

    # 1. Subtract stroke ('sub') at center of right cheek (x=195, y=150) removes it from AI mask and spares it from skin_fill
    p_sub = P.Params(
        skin_fill=0.85, db=0.0, plastic=0.0, skin_even=0.0, vignette=0.0,
        mask_strokes=[{"mode": "sub", "size": 24.0 / W, "softness": 0.2, "points": [[195.0 / W, 150.0 / H]]}],
    )
    out_sub, mq_sub = eng.render(src, fit, p_sub, db_enabled=False, return_mask=True)
    qh, qw = mq_sub.shape[:2]
    assert float(mq_sub[int(150 * qh / H), int(195 * qw / W)]) < 0.05
    # Erased cheek spot stays near original brightness while un-erased left cheek (x=145, y=150) is brightened!
    assert abs(float(out_sub[146:154, 191:199].mean()) - float(src[146:154, 191:199].mean())) < 4.0
    assert float(out_sub[146:154, 141:149].mean()) > float(src[146:154, 141:149].mean()) + 20.0

    # 2. Add stroke ('add') on the warm wooden panel on the left (x=36, y=140) includes it in mq_raw
    p_add = P.Params(
        skin_fill=0.80, db=0.0, plastic=0.0, skin_even=0.0, vignette=0.0,
        mask_strokes=[{"mode": "add", "size": 24.0 / W, "softness": 0.3, "points": [[36.0 / W, 140.0 / H]]}],
    )
    out_add, mq_add = eng.render(src, fit, p_add, db_enabled=False, return_mask=True)
    assert float(mq_add[int(140 * qh / H), int(36 * qw / W)]) > 0.75
    assert float(out_add[136:144, 32:40].mean()) > float(src[136:144, 32:40].mean()) + 10.0

    # Roundtrip serialization in sidecar dict
    rt = P.Params.from_dict(p_sub.to_dict())
    assert len(rt.mask_strokes) == 1 and rt.mask_strokes[0]["mode"] == "sub"


def test_full_opencl_umat_pipeline_and_benchmark():
    assert DEV.UMAT_MIN_PIXELS <= 180_000
    DEV.ocl_configure("ocl")
    rng = np.random.default_rng(19)
    frame_f = rng.uniform(0, 255, size=(480, 640, 3)).astype(np.float32)
    frame_u8 = frame_f.astype(np.uint8)

    b_box = DEV.ocl_box_filter(frame_f[..., 0], 5)
    assert isinstance(b_box, np.ndarray) and b_box.shape == (480, 640)

    b_bil = DEV.ocl_bilateral_filter(frame_f[..., 0], 5, 15.0, 4.0)
    assert isinstance(b_bil, np.ndarray) and b_bil.shape == (480, 640)

    M = GEO.matrix(2.5, 640, 480)
    warped = DEV.ocl_warp_affine(frame_u8, M, (640, 480), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    assert isinstance(warped, np.ndarray) and warped.shape == (480, 640, 3) and warped.dtype == np.uint8

    g_rgb = COL.fast_guided_rgb(frame_f / 255.0, (frame_f[..., 0] > 128).astype(np.float32), r=6, eps=1e-3, s=2.0)
    assert isinstance(g_rgb, np.ndarray) and g_rgb.shape == (480, 640) and g_rgb.dtype == np.float32

    det = DET.apply_detail_pass(frame_u8, sharpen_amount=0.6, sharpen_mode="rl", texture=0.3, denoise=0.4)
    assert isinstance(det, np.ndarray) and det.shape == frame_u8.shape and det.dtype == np.uint8

    bench = DEV.ocl_benchmark(h=360, w=480)
    assert isinstance(bench, dict) and "cpu_ms" in bench and "available" in bench and "status_pl" in bench
    DEV.ocl_configure("auto")


def test_default_view_setting_and_sticky_canvas_mode_logic():
    assert hasattr(S, "DEFAULT_VIEWS")
    for mode in ("Wynik", "Przed | Po", "Z wzorcem", "Podział", "auto"):
        assert mode in S.DEFAULT_VIEWS
    st = S.Settings()
    assert hasattr(st, "default_view") and hasattr(st, "open_panels")
    assert APP.RetuszApp.resolve_canvas_view("Wynik", ref_enabled=True, user_picked=False) == "Wynik"
    assert APP.RetuszApp.resolve_canvas_view("Przed | Po", ref_enabled=True, user_picked=False) == "Przed | Po"
    assert APP.RetuszApp.resolve_canvas_view("Z wzorcem", ref_enabled=False, user_picked=False) == "Z wzorcem"
    assert APP.RetuszApp.resolve_canvas_view("auto", ref_enabled=True, user_picked=True, current="Wynik") == "Wynik"


def test_clean_original_fast_reset_and_named_history_inference():
    dirty = P.Params(
        strength=0.95, exposure=0.65, db=0.80, plastic=0.70, skin_fill=0.45,
        vignette=0.50, angle=4.2, crop=[0.1, 0.1, 0.9, 0.9],
        strokes=[{"mode": "dodge", "size": 0.05, "strength": 0.4, "softness": 0.5, "points": [[0.5, 0.5]]}],
        heal_strokes=[{"size": 0.03, "points": [[0.4, 0.4]]}],
        mask_strokes=[{"mode": "sub", "size": 0.08, "softness": 0.3, "points": [[0.2, 0.2]]}],
        rating=5, flag=1, reference_path="__none__",
    )
    clean = P.Params.clean_original(dirty)
    assert clean.strength == 0.0
    assert clean.exposure == 0.0 and clean.db == 0.0 and clean.plastic == 0.0 and clean.skin_fill == 0.0
    assert clean.skin_even == 0.0 and clean.vignette == 0.0 and clean.angle == 0.0 and clean.crop is None
    assert clean.strokes == [] and clean.heal_strokes == [] and clean.mask_strokes == []
    assert clean.rating == 5 and clean.flag == 1 and clean.reference_path == "__none__"

    src, faces_n = _backlit_portrait_with_warm_bg_object()
    eng = P.Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(src, None)
    fit.faces_n = faces_n
    out_clean = eng.render(src, fit, clean)
    assert np.abs(out_clean.astype(int) - src.astype(int)).max() <= 1

    d0 = P.Params().to_dict()
    d1 = P.Params(skin_fill=0.35).to_dict()
    lbl1 = APP.RetuszApp._infer_history_label(d0, d1)
    assert "Dopełnienie skóry" in lbl1 and "+35" in lbl1

    d2 = P.Params(skin_fill=0.35, mask_strokes=[{"mode": "sub", "size": 0.05, "softness": 0.3, "points": [[0.3, 0.3]]}]).to_dict()
    lbl2 = APP.RetuszApp._infer_history_label(d1, d2)
    assert "Maska AI" in lbl2 and "Usuń" in lbl2


def test_no_competitor_brand_mentions_and_version_091():
    forbidden = ("lightroom", "capture one", "photoshop")
    check_files = (
        list((REPO_ROOT / "retusz").glob("*.py"))
        + list((REPO_ROOT / "retusz" / "gui").glob("*.py"))
        + list((REPO_ROOT / "retusz" / "engine").glob("*.py"))
        + list((REPO_ROOT / "crates").rglob("*.rs"))
        + [REPO_ROOT / "README.md", REPO_ROOT / "AGENTS.md"]
    )
    for fp in check_files:
        txt = fp.read_text(encoding="utf-8").lower()
        for word in forbidden:
            assert word not in txt, f"Found forbidden competitor brand '{word}' in {fp.relative_to(REPO_ROOT)}"

    p0 = P.Params()
    p1 = P.Params(skin_fill=0.55)
    assert p0.colour_key() != p1.colour_key()
    rt = P.Params.from_dict(p1.to_dict())
    assert abs(rt.skin_fill - 0.55) < 1e-5

    portrait_keys = [row[0] for row in APP.SLIDER_GROUPS["Portret 3D"]]
    assert "skin_fill" in portrait_keys
    assert "skin_fill" in APP.SLIDER_HINTS and "skin_fill" in APP.PRO_TIPS and "skin_fill" in APP._COPY_KEYS
    copy_all = [k for _, _, gkeys, _ in DLG.COPY_GROUPS for k in gkeys]
    assert "skin_fill" in copy_all

    for method_name in (
        "toggle_section", "jump_to_history", "history_labels", "reset_to_clean_original",
        "toggle_dim_panels", "toggle_distraction_free", "toggle_mask_tool",
        "_show_edit_menu", "_show_photo_menu", "_show_view_menu",
    ):
        assert hasattr(APP.RetuszApp, method_name), f"Missing {method_name} on RetuszApp"

    pyproj = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((REPO_ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    assert retusz.__version__ == "0.9.1"
    assert pyproj["project"]["version"] == "0.9.1"
    assert cargo["workspace"]["package"]["version"] == "0.9.1"

