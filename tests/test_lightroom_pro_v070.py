"""Testy TDD dla Retusz v0.7.0 Lightroom Pro:
- Dwuosiowe Światło 3D: Lewo ↔ Prawo (light_x) oraz Dół ↔ Góra (light_y) na mapie normalnych 3D
- Separacja tła 3D (bg_exposure): przyciemnianie / rozjaśnianie tła z pełną ochroną twarzy i 0% halo
- Kalibracja kierunków Odcienia skóry (◀ Róż/Czerwień vs Złoty/Żółty ▶) oraz podpowiedzi kierunkowych UI (SLIDER_HINTS / PRO_TIPS).

Author: Jack4B.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from retusz.engine.pipeline import Engine, Params
from retusz.engine import relight as RL, color as C
from tests.synthetic import portrait_scene


def test_3d_relight_two_axis_left_right_and_bottom_top():
    """Dwa suwaki Światło 3D: light_x (-1..+1: Lewo ↔ Prawo) i light_y (-1..+1: Dół ↔ Góra)
    sterują kierunkiem wirtualnego softboxa 3D na twarzy z zachowaniem średniej ekspozycji."""
    v_left = RL.light_vec_xy(-0.8, 0.0)
    v_right = RL.light_vec_xy(0.8, 0.0)
    v_top = RL.light_vec_xy(0.0, 0.8)
    v_bot = RL.light_vec_xy(0.0, -0.8)
    assert v_left[0] < -0.5 and v_right[0] > 0.5
    assert v_top[1] < -0.5 and v_bot[1] > 0.5  # image coords: y down -> top light has negative y component
    assert np.isclose(np.linalg.norm(v_left), 1.0, atol=1e-4)

    img, _ = portrait_scene(600, 450, seed=7)
    eng = Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)
    assert fit.faces_n, "Syntetyczny portret powinien zawierać twarz"
    fx, fy, fw, fh = (fit.faces_n[0] * img.shape[1]).astype(int)
    cx, cy = fx + fw // 2, fy + fh // 2

    out_left = eng.render(img, fit, Params(strength=0.0, db=0.0, vignette=0.0, skin_even=0.0,
                                           plastic=0.9, light_x=-0.9, light_y=0.0))
    out_right = eng.render(img, fit, Params(strength=0.0, db=0.0, vignette=0.0, skin_even=0.0,
                                            plastic=0.9, light_x=0.9, light_y=0.0))
    L_left_l = float(out_left[cy - 20:cy + 20, fx + 8:cx - 4].mean())
    L_left_r = float(out_left[cy - 20:cy + 20, cx + 4:fx + fw - 8].mean())
    L_right_l = float(out_right[cy - 20:cy + 20, fx + 8:cx - 4].mean())
    L_right_r = float(out_right[cy - 20:cy + 20, cx + 4:fx + fw - 8].mean())
    assert (L_left_l - L_left_r) > (L_right_l - L_right_r) + 1.5

    out_top = eng.render(img, fit, Params(strength=0.0, db=0.0, vignette=0.0, skin_even=0.0,
                                          plastic=0.9, light_x=0.0, light_y=0.9))
    out_bot = eng.render(img, fit, Params(strength=0.0, db=0.0, vignette=0.0, skin_even=0.0,
                                          plastic=0.9, light_x=0.0, light_y=-0.9))
    L_top_u = float(out_top[fy + 10:cy - 4, cx - 20:cx + 20].mean())
    L_top_d = float(out_top[cy + 4:fy + fh - 10, cx - 20:cx + 20].mean())
    L_bot_u = float(out_bot[fy + 10:cy - 4, cx - 20:cx + 20].mean())
    L_bot_d = float(out_bot[cy + 4:fy + fh - 10, cx - 20:cx + 20].mean())
    assert (L_top_u - L_top_d) > (L_bot_u - L_bot_d) + 1.0


def test_3d_background_separation_bg_exposure_protects_subject_face():
    """Separacja tła 3D (bg_exposure: -1..+1) przyciemnia lub rozjaśnia tło bez zmiany jasności twarzy."""
    img, _ = portrait_scene(600, 450, seed=12)
    eng = Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)
    fx, fy, fw, fh = (fit.faces_n[0] * img.shape[1]).astype(int)
    face_box = (slice(fy + fh // 4, fy + 3 * fh // 4), slice(fx + fw // 4, fx + 3 * fw // 4))
    bg_box = (slice(10, 70), slice(img.shape[1] - 90, img.shape[1] - 15))

    p_base = Params(strength=0.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0, bg_exposure=0.0)
    p_dark = Params(strength=0.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0, bg_exposure=-0.85)
    p_bright = Params(strength=0.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0, bg_exposure=0.85)

    out_0 = eng.render(img, fit, p_base)
    out_d = eng.render(img, fit, p_dark)
    out_b = eng.render(img, fit, p_bright)

    assert float(out_d[bg_box].mean()) < float(out_0[bg_box].mean()) - 8.0
    assert float(out_b[bg_box].mean()) > float(out_0[bg_box].mean()) + 8.0
    # Twarz pozostaje chroniona (< 0.8 zmiany jasności)
    assert abs(float(out_d[face_box].mean()) - float(out_0[face_box].mean())) < 0.8
    assert abs(float(out_b[face_box].mean()) - float(out_0[face_box].mean())) < 0.8


def test_skin_hue_rose_red_vs_golden_yellow_direction_and_pro_hints():
    """Odcień skóry (skin_hue): lewo (-) = Róż / Czerwień, prawo (+) = Złoty / Żółty,
    a UI udostępnia podpowiedzi kierunkowe SLIDER_HINTS oraz opisy PRO_TIPS."""
    from retusz.gui import app as APP
    for key in ("skin_hue", "temp", "tint", "light_x", "light_y", "bg_exposure"):
        assert key in APP.SLIDER_HINTS, f"Brak podpowiedzi kierunkowej dla {key}"
        left_txt, right_txt, left_col, right_col = APP.SLIDER_HINTS[key]
        assert left_txt and right_txt and left_col.startswith("#") and right_col.startswith("#")
        assert key in APP.PRO_TIPS and len(APP.PRO_TIPS[key]) > 20

    img, _ = portrait_scene(600, 450, seed=5)
    eng = Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)
    fx, fy, fw, fh = (fit.faces_n[0] * img.shape[1]).astype(int)
    face_box = (slice(fy + fh // 4, fy + 3 * fh // 4), slice(fx + fw // 4, fx + 3 * fw // 4))

    out_rose = eng.render(img, fit, Params(strength=0.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0, skin_hue=-0.8))
    out_gold = eng.render(img, fit, Params(strength=0.0, db=0.0, plastic=0.0, vignette=0.0, skin_even=0.0, skin_hue=0.8))
    lab_rose = C.to_lab(out_rose[face_box])
    lab_gold = C.to_lab(out_gold[face_box])
    # W prawo (+, Złoty/Żółty) b* rośnie względem a*, w lewo (-, Róż/Czerwień) a* rośnie względem b*
    assert float((lab_gold[..., 2] - lab_gold[..., 1]).mean()) > float((lab_rose[..., 2] - lab_rose[..., 1]).mean()) + 3.0


def test_ai_face_landmarks_and_skin_mask_overlay_visualization():
    """Wizualizacja pracy AI (Maska AI - klawisz M / przełącznik Maska AI):
    nanosi na podgląd maskę skóry, obrys sylwetki 3D, ramki wykrytych twarzy oraz 5 punktów anatomicznych (oczy, nos, usta)."""
    from retusz.engine import histo as HI
    img, _ = portrait_scene(600, 450, seed=9)
    eng = Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)
    out, mask_q = eng.render(img, fit, Params(), return_mask=True)

    ov = HI.ai_mask_overlay(
        out,
        mask_q,
        faces_n=fit.faces_n,
        lms_n=getattr(fit, "lms_n", None),
        subject_fg=fit.relight.fg if fit.relight is not None else None,
    )
    assert ov.shape == out.shape and ov.dtype == np.uint8
    # Nakładka musi różnić się od obrazu wynikowego w obszarze twarzy (maska skóry + ramka twarzy + punkty oczu/ust)
    diff = np.abs(ov.astype(np.int32) - out.astype(np.int32)).mean()
    assert diff > 2.0


def test_multi_photo_subset_import_from_folder(tmp_path):
    """Import wybranych kilku zdjęć z folderu (np. 2 z 4 plików): Series.from_selection ładuje tylko wybrane pliki."""
    import cv2
    from retusz.sidecar import Series
    img, _ = portrait_scene(480, 360, seed=3)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    files = []
    for i in range(4):
        p = tmp_path / f"portret_{i+1}.jpg"
        cv2.imwrite(str(p), bgr)
        files.append(p)

    # Użytkownik wybiera w oknie dialogowym tylko 2 zdjęcia z 4 znajdujących się w folderze
    chosen = [files[0], files[2]]
    ser, loaded = Series.from_selection(chosen)
    assert ser.folder == tmp_path
    assert loaded == [files[0], files[2]]
    assert len(loaded) == 2


def test_pro_ai_skin_mask_hue_tolerance_and_anatomical_feature_protection():
    """Korekta maski twarzy AI (mask_hue_tol: -1..+1, mask_protect: 0..1):
    - mask_hue_tol < 0 zawęża maskę po odcieniu Mahalanobisa (odrzuca tło/włosy), > 0 rozszerza na szyję/cienie
    - mask_protect > 0 chroni oczy i usta (punkty anatomiczne YuNet) przy zachowaniu pełnej maski na policzkach."""
    img, _ = portrait_scene(600, 450, seed=14)
    eng = Engine(None, "classic", depth_ai=False)
    fit = eng.prepare(img, None)

    _, m_narrow = eng.render(img, fit, Params(mask_hue_tol=-0.85, mask_protect=0.0), return_mask=True)
    _, m_default = eng.render(img, fit, Params(mask_hue_tol=0.0, mask_protect=0.0), return_mask=True)
    _, m_wide = eng.render(img, fit, Params(mask_hue_tol=0.85, mask_protect=0.0), return_mask=True)

    assert float(m_narrow.mean()) < float(m_default.mean()) < float(m_wide.mean())

    # Ochrona oczu i ust (mask_protect = 0.9 vs 0.0)
    _, m_no_prot = eng.render(img, fit, Params(mask_hue_tol=0.0, mask_protect=0.0), return_mask=True)
    _, m_prot = eng.render(img, fit, Params(mask_hue_tol=0.0, mask_protect=0.95), return_mask=True)
    qh, qw = m_prot.shape[:2]
    fx, fy, fw, fh = (fit.faces_n[0] * qw).astype(int)
    # Punkt lewego oka (~0.32 fw, 0.38 fh) vs środek policzka (~0.28 fw, 0.58 fh)
    ex, ey = int(fx + 0.32 * fw), int(fy + 0.38 * fh)
    assert float(m_prot[max(0, ey-2):ey+3, max(0, ex-2):ex+3].mean()) < float(m_no_prot[max(0, ey-2):ey+3, max(0, ex-2):ex+3].mean()) - 0.15


def test_auto_match_references_pairs_each_photo_in_series_with_closest_reference_from_pool(tmp_path):
    """Automatyczne parowanie wzorców z puli (Series.auto_match_references):
    każde zdjęcie w serii otrzymuje najdokładniej pasujący wzorzec z Banku Wzorców (po sygnaturze OKLab)."""
    import cv2
    from retusz.sidecar import Series
    base, _ = portrait_scene(480, 360, seed=21)

    # 3 różne warianty oświetlenia/kolorystyki zdjęć i odpowiadające im 3 wzorce w puli
    warm = np.clip(base.astype(np.float32) * np.array([1.18, 1.02, 0.80], np.float32), 0, 255).astype(np.uint8)
    cool = np.clip(base.astype(np.float32) * np.array([0.80, 0.98, 1.22], np.float32), 0, 255).astype(np.uint8)
    dark = np.clip(base.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)

    p_warm = tmp_path / "ujecie_cieple.jpg"
    p_cool = tmp_path / "ujecie_chlodne.jpg"
    p_dark = tmp_path / "ujecie_ciemne.jpg"
    r_warm = tmp_path / "wzorzec_cieply.jpg"
    r_cool = tmp_path / "wzorzec_chlodny.jpg"
    r_dark = tmp_path / "wzorzec_ciemny.jpg"

    for path, arr in ((p_warm, warm), (p_cool, cool), (p_dark, dark), (r_warm, warm), (r_cool, cool), (r_dark, dark)):
        cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    ser = Series(tmp_path)
    for rp in (r_warm, r_cool, r_dark):
        ser.add_to_reference_bank(rp)

    photos = [p_warm, p_cool, p_dark]
    mapping = ser.auto_match_references(photos)
    assert Path(mapping[p_warm]).name == "wzorzec_cieply.jpg"
    assert Path(mapping[p_cool]).name == "wzorzec_chlodny.jpg"
    assert Path(mapping[p_dark]).name == "wzorzec_ciemny.jpg"
    assert ser.effective_reference_for(p_warm)[0].name == "wzorzec_cieply.jpg"
    assert ser.effective_reference_for(p_cool)[0].name == "wzorzec_chlodny.jpg"
    assert ser.effective_reference_for(p_dark)[0].name == "wzorzec_ciemny.jpg"


def test_library_folder_tree_discovery(tmp_path):
    """Drzewo folderów w module Biblioteka (G) wykrywa bieżący folder, podfoldery i sąsiednie foldery ze zdjęciami,
    pomijając podfolder eksportu 'retusz'."""
    import cv2
    from retusz.sidecar import Series
    img, _ = portrait_scene(400, 300, seed=4)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    f_main = tmp_path / "sesja_slubna"
    f_sub = f_main / "plener"
    f_sib = tmp_path / "studio_portret"
    f_exp = f_main / "retusz"
    for d in (f_main, f_sub, f_sib, f_exp):
        d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(f_main / "a.jpg"), bgr)
    cv2.imwrite(str(f_main / "b.jpg"), bgr)
    cv2.imwrite(str(f_sub / "c.jpg"), bgr)
    cv2.imwrite(str(f_sib / "d.jpg"), bgr)
    cv2.imwrite(str(f_exp / "exp.jpg"), bgr)

    folders = Series.discover_workspace_folders(f_main, output_subfolder="retusz")
    names = {p.name: count for p, count in folders}
    assert names.get("sesja_slubna") == 2
    assert names.get("plener") == 1
    assert names.get("studio_portret") == 1
    assert "retusz" not in names


