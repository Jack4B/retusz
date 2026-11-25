"""TDD tests for retusz.engine.faces.skin_mask: reject dark brown hair and skin-like background patches,
keep true skin across light and dark complexions plus hands matching face skin."""
import numpy as np, cv2
from synthetic import portrait_scene
from retusz.engine import faces as F


def test_empty_faces_gives_zero_mask():
    rgb, _ = portrait_scene(600, 400, seed=1)
    m = F.skin_mask(rgb, [])
    assert m.shape == rgb.shape[:2] and m.max() == 0


def test_rejects_dark_brown_hair_inside_face_ellipse():
    rgb, info = portrait_scene(900, 600, seed=4, grain=2)
    m = F.skin_mask(rgb, [info["face"]])
    # true hair: inside portrait_scene person, outside skin and torso (above neck)
    x, y, fw, fh = info["face"]
    hair = (info["person"] > 0.5) & (info["skin"] < 0.05) & (info["features"] < 0.05)
    hair[int(y + fh * 0.9):, :] = False  # keep head hair only (not shirt)
    hair = cv2.erode(hair.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    core_skin = cv2.erode((info["skin"] > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    assert hair.sum() > 500
    assert float(m[hair].mean()) < 0.08, f"dark brown hair leaked into skin mask: {m[hair].mean():.3f}"
    assert float(m[core_skin].mean()) > 0.80, f"true skin dropped too low: {m[core_skin].mean():.3f}"


def test_rejects_skin_coloured_background_patches():
    rgb, info = portrait_scene(900, 600, seed=5, grain=2)
    rgb = rgb.copy()
    H, W = rgb.shape[:2]
    x, y, fw, fh = info["face"]
    # 1. Warm skin-like patch on the wall right next to the hair (inside the oversized 0.75w face ellipse!)
    p1 = np.zeros((H, W), bool)
    p1[max(0, y):int(y + 0.5 * fh), max(0, int(x - 0.25 * fw)):max(1, int(x - 0.02 * fw))] = True
    p1 &= info["person"] < 0.05
    rgb[p1] = (195, 148, 122)
    # 2. Large warm wooden sofa/wall patch in the lower-left background (within hand search gate!)
    p2 = np.zeros((H, W), bool)
    p2[int(0.55 * H):int(0.85 * H), int(0.05 * W):int(0.28 * W)] = True
    p2 &= info["person"] < 0.05
    rgb[p2] = (178, 128, 100)
    # 3. Dark brown furniture patch in lower-right background
    p3 = np.zeros((H, W), bool)
    p3[int(0.58 * H):int(0.75 * H), int(0.78 * W):int(0.95 * W)] = True
    p3 &= info["person"] < 0.05
    rgb[p3] = (62, 42, 30)

    m = F.skin_mask(rgb, [info["face"]], hands=True)
    assert p1.sum() > 200 and p2.sum() > 500 and p3.sum() > 500
    assert float(m[p1].mean()) < 0.08, f"adjacent warm wall leaked: {m[p1].mean():.3f}"
    assert float(m[p2].mean()) < 0.12, f"background sofa leaked: {m[p2].mean():.3f}"
    assert float(m[p3].mean()) < 0.08, f"dark brown furniture leaked: {m[p3].mean():.3f}"


def test_keeps_dark_skin_tone_and_rejects_darker_hair():
    rgb, info = portrait_scene(900, 600, seed=6, grain=1)
    rgb = rgb.copy()
    sk = info["skin"] > 0.5
    x, y, fw, fh = info["face"]
    hair = (info["person"] > 0.5) & (info["skin"] < 0.05) & (info["features"] < 0.05)
    hair[int(y + fh * 0.9):, :] = False
    # Shift skin to rich dark brown complexion, hair even darker
    rgb[sk] = np.clip(rgb[sk].astype(np.float32) * np.array([0.55, 0.48, 0.44]), 0, 255).astype(np.uint8)
    rgb[hair] = np.clip(rgb[hair].astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    m = F.skin_mask(rgb, [info["face"]])
    core_skin = cv2.erode((info["skin"] > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    core_hair = cv2.erode(hair.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    assert float(m[core_skin].mean()) > 0.70, f"dark skin missed: {m[core_skin].mean():.3f}"
    assert float(m[core_hair].mean()) < 0.12, f"hair leaked on dark-skin portrait: {m[core_hair].mean():.3f}"


def test_detects_hand_matching_face_skin():
    rgb, info = portrait_scene(900, 600, seed=7, grain=1)
    rgb = rgb.copy()
    x, y, fw, fh = info["face"]
    # Draw a realistic hand on the shirt matching the face's exact skin colour
    core_skin = cv2.erode((info["skin"] > 0.9).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    skin_col = tuple(int(v) for v in np.median(rgb[core_skin], axis=0))
    hx, hy = int(x + 0.15 * fw), int(y + 1.85 * fh)
    hand = np.zeros(rgb.shape[:2], np.uint8)
    cv2.ellipse(hand, (hx, hy), (int(0.24 * fw), int(0.32 * fw)), 15, 0, 360, 1, -1)
    rgb[hand > 0] = skin_col
    m = F.skin_mask(rgb, [info["face"]], hands=True)
    core_hand = cv2.erode(hand, np.ones((9, 9), np.uint8)) > 0
    assert float(m[core_hand].mean()) > 0.55, f"hand missed: {m[core_hand].mean():.3f}"
