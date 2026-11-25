"""ColorChecker Classic (24-patch) automatic detection, Root-Polynomial + Bradford colour calibration
(Finlayson, Mackiewicz & Hurlbert 2015, IEEE Trans. Image Processing), CIEDE2000 reporting (Sharma 2005),
and visual diagnostic overlay."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np, cv2
from . import color as C, tone as TN

CC24_NAMES = (
    "1. Ciemna skóra", "2. Jasna skóra", "3. Błękit nieba", "4. Zieleń liści", "5. Niebieski kwiat", "6. Turkus",
    "7. Pomarańczowy", "8. Fioletowoniebieski", "9. Czerwony", "10. Fioletowy", "11. Żółtozielony", "12. Pomarańczowożółty",
    "13. Niebieski", "14. Zielony", "15. Czerwony (podst.)", "16. Żółty", "17. Magenta", "18. Cyjan",
    "19. Biały 9.5", "20. Szary 8", "21. Szary 6.5", "22. Szary 5", "23. Szary 3.5", "24. Czarny 2",
)

# Standard 24-patch ColorChecker Classic sRGB D65 reference values (BabelColor / X-Rite D65):
CC24_SRGB = np.array([
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67], [133, 128, 177], [103, 189, 170],
    [214, 126,  44], [ 80,  91, 166], [193,  90,  99], [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31], [187,  86, 149], [  8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52],
], dtype=np.uint8)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Vectorised CIEDE2000 colour difference (Sharma, Wu & Dalal 2005). Inputs: (..., 3) float LAB."""
    a = np.asarray(lab1, dtype=np.float64)
    b = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = a[..., 0], a[..., 1], a[..., 2]
    L2, a2, b2 = b[..., 0], b[..., 1], b[..., 2]
    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    Cb = 0.5 * (C1 + C2)
    Cb7 = Cb ** 7
    G = 0.5 * (1.0 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dhp = np.where(C1p * C2p == 0.0, 0.0, np.where(np.abs(dh) <= 180.0, dh, np.where(dh > 180.0, dh - 360.0, dh + 360.0)))
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(0.5 * dhp))

    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hbp = np.where(
        C1p * C2p == 0.0,
        h1p + h2p,
        np.where(np.abs(h1p - h2p) <= 180.0, 0.5 * (h1p + h2p),
                 np.where(h1p + h2p < 360.0, 0.5 * (h1p + h2p + 360.0), 0.5 * (h1p + h2p - 360.0)))
    )
    T = (1.0
         - 0.17 * np.cos(np.radians(hbp - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * hbp))
         + 0.32 * np.cos(np.radians(3.0 * hbp + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * hbp - 63.0)))
    dtheta = 30.0 * np.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    RC = 2.0 * np.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    Lm50 = (Lbp - 50.0) ** 2
    SL = 1.0 + (0.015 * Lm50) / np.sqrt(20.0 + Lm50)
    SC = 1.0 + 0.045 * Cbp
    SH = 1.0 + 0.015 * Cbp * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * RC
    return np.sqrt((dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)).astype(np.float32)


def mean_de00_rgb(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    """Mean CIEDE2000 between two uint8 RGB arrays of identical shape."""
    a = C.to_lab(rgb1.reshape(-1, 1, 3)).reshape(-1, 3)
    b = C.to_lab(rgb2.reshape(-1, 1, 3)).reshape(-1, 3)
    return float(ciede2000(a, b).mean())


@dataclass
class ChartDetection:
    centres: np.ndarray          # (24, 2) float32 pixel coordinates in the full input image, order #1..#24
    patch_radius: float          # half-side of a single patch in pixels
    corners: np.ndarray          # (4, 2) float32 outer corners of the 6x4 patch centres grid


@dataclass
class ChartReport:
    de00_before: float
    de00_after: float
    max_de00_before: float
    max_de00_after: float
    skin_de00: float             # mean ΔE00 on patches #1 (Dark Skin) and #2 (Light Skin)
    gray_de00: float             # mean ΔE00 on neutral patches #19..#24
    patch_de00_before: list[float]
    patch_de00_after: list[float]

    def summary_pl(self) -> str:
        return (
            f"ColorChecker 24: ΔE₀₀ {self.de00_before:.1f} → {self.de00_after:.2f} "
            f"(skóra: {self.skin_de00:.2f} · szarości: {self.gray_de00:.2f} · max: {self.max_de00_after:.2f})"
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChartCalibration:
    detection: ChartDetection
    measured_srgb: np.ndarray    # (24, 3) float32 0..255
    calibrated_srgb: np.ndarray  # (24, 3) uint8
    target_srgb: np.ndarray      # (24, 3) uint8
    lut_srgb: np.ndarray         # (33, 33, 33, 3) float32 0..1 calibrated sRGB 3D LUT
    lut_lab: np.ndarray          # (33, 33, 33, 3) float32 calibrated CIELAB 3D LUT
    report: ChartReport


def _sample_patch_rgb(rgb: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    H, W = rgb.shape[:2]
    r = max(2, int(round(radius * 0.45)))
    cx, cy = int(round(float(centre[0]))), int(round(float(centre[1])))
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    patch = rgb[y0:y1, x0:x1].reshape(-1, 3)
    if len(patch) == 0:
        return rgb[np.clip(cy, 0, H - 1), np.clip(cx, 0, W - 1)].astype(np.float32)
    return np.median(patch, axis=0).astype(np.float32)


def _fit_6x4_lattice(pts: np.ndarray, patch_side: float, img_rgb: np.ndarray) -> ChartDetection | None:
    """Fit a right-handed 6x4 regular grid to candidate square patch centroids and orient via the neutral ramp."""
    if len(pts) < 12:
        return None
    rect = cv2.minAreaRect(pts.astype(np.float32))
    (rcx, rcy), (rw, rh), ang_deg = rect
    if min(rw, rh) < patch_side * 1.8:
        return None
    # Build a right-handed orthonormal frame (e1 along the 6-column long axis, e2 = (-e1_y, e1_x) along 4-row short axis)
    box = cv2.boxPoints(rect)
    v01 = box[1] - box[0]
    v12 = box[2] - box[1]
    if np.linalg.norm(v01) >= np.linalg.norm(v12):
        e1 = v01 / (np.linalg.norm(v01) + 1e-9)
    else:
        e1 = v12 / (np.linalg.norm(v12) + 1e-9)
    e2 = np.array([-e1[1], e1[0]], dtype=np.float64)

    rel = (pts.astype(np.float64) - np.array([rcx, rcy], dtype=np.float64))
    u = rel @ e1
    v = rel @ e2
    u_min, u_max = np.percentile(u, 2), np.percentile(u, 98)
    v_min, v_max = np.percentile(v, 2), np.percentile(v, 98)
    du = (u_max - u_min) / 5.0
    dv = (v_max - v_min) / 3.0
    if du < patch_side * 0.7 or dv < patch_side * 0.7:
        return None
    ratio = du / (dv + 1e-9)
    if not (0.72 <= ratio <= 1.38):
        return None

    cols = np.clip(np.round((u - u_min) / du).astype(np.int32), 0, 5)
    rows = np.clip(np.round((v - v_min) / dv).astype(np.int32), 0, 3)

    # Deduplicate grid cells: keep the point closest to each integer cell center
    cell_map: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for k in range(len(pts)):
        c_idx, r_idx = int(cols[k]), int(rows[k])
        u_exp = u_min + c_idx * du
        v_exp = v_min + r_idx * dv
        dist = float(np.hypot(u[k] - u_exp, v[k] - v_exp))
        if dist > 0.45 * max(du, dv):
            continue
        key = (r_idx, c_idx)
        if key not in cell_map or dist < cell_map[key][0]:
            cell_map[key] = (dist, pts[k])

    if len(cell_map) < 12:
        return None

    A = []
    B = []
    for (r_idx, c_idx), (_d, p_xy) in cell_map.items():
        A.append([float(c_idx), float(r_idx), 1.0])
        B.append([float(p_xy[0]), float(p_xy[1])])
    A_arr = np.array(A, dtype=np.float64)
    B_arr = np.array(B, dtype=np.float64)
    T, _, _, _ = np.linalg.lstsq(A_arr, B_arr, rcond=None)  # 3x2: [c, r, 1] @ T -> [x, y]

    pred = A_arr @ T
    res = np.linalg.norm(pred - B_arr, axis=1)
    step = float(0.5 * (np.linalg.norm(T[0]) + np.linalg.norm(T[1])))
    if res.mean() > 0.22 * step:
        return None

    # Generate all 24 grid centres in both right-handed orientations (0 deg vs 180 deg)
    grid_0 = np.array([[c, r, 1.0] for r in range(4) for c in range(6)], dtype=np.float64) @ T
    grid_180 = np.array([[5 - c, 3 - r, 1.0] for r in range(4) for c in range(6)], dtype=np.float64) @ T

    rad = 0.35 * step

    def neutral_score(centres_24: np.ndarray) -> float:
        # Row 3 (indices 18..23) should be the neutral ramp: #19 White (brightest) -> #24 Black (darkest)
        s_rgb = np.array([_sample_patch_rgb(img_rgb, centres_24[i], rad) for i in range(24)], dtype=np.float32)
        lab = C.to_lab(s_rgb.reshape(24, 1, 3).astype(np.uint8)).reshape(24, 3)
        L_bot = lab[18:24, 0]
        C_bot = np.hypot(lab[18:24, 1], lab[18:24, 2]).mean()
        C_top = np.hypot(lab[0:6, 1], lab[0:6, 2]).mean()
        # Monotonic decrease from #19 to #24 + lower chroma on bottom row than top row
        diffs = np.diff(L_bot)
        mono = float(np.sum(diffs < 0)) * 12.0 + float(L_bot[0] - L_bot[-1])
        return mono + float(C_top - C_bot) * 1.5

    centres = grid_0 if neutral_score(grid_0) >= neutral_score(grid_180) else grid_180
    corners = np.array([centres[0], centres[5], centres[23], centres[18]], dtype=np.float32)
    return ChartDetection(centres=centres.astype(np.float32), patch_radius=float(0.36 * step), corners=corners)


def detect_chart(rgb: np.ndarray) -> ChartDetection | None:
    """Automatically detect a 24-patch ColorChecker Classic chart in an RGB image.
    Invariant to chart scale, translation, and 2D rotation (0..360 deg)."""
    H, W = rgb.shape[:2]
    work_side = 1000
    s = min(1.0, work_side / max(H, W))
    work = rgb if s >= 1.0 else cv2.resize(rgb, (max(2, round(W * s)), max(2, round(H * s))), interpolation=cv2.INTER_AREA)
    Y = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)

    # Candidate dark-frame masks: local adaptive thresholds + percentile thresholds
    masks = []
    for block in (31, 51, 75):
        b = min(block, (min(Y.shape[:2]) // 2) * 2 - 1)
        if b >= 11:
            m = cv2.adaptiveThreshold(Y, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, b, 6)
            masks.append(m)
    for q in (12, 18, 25, 32):
        thr = float(np.percentile(Y, q))
        masks.append(((Y < thr).astype(np.uint8)) * 255)

    best: ChartDetection | None = None
    best_count = 0

    for m in masks:
        for k_close in (1, 3):
            mc = m if k_close == 1 else cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
            contours, hier = cv2.findContours(mc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if hier is None:
                continue
            hier = hier[0]
            # Group child contours by parent contour
            children_by_parent: dict[int, list[int]] = {}
            for idx, (_nxt, _prv, _ch, parent) in enumerate(hier):
                if parent >= 0:
                    children_by_parent.setdefault(int(parent), []).append(idx)

            for _parent, ch_indices in children_by_parent.items():
                if len(ch_indices) < 12:
                    continue
                squares = []
                areas = []
                for c_idx in ch_indices:
                    cnt = contours[c_idx]
                    area = cv2.contourArea(cnt)
                    if area < 36:
                        continue
                    (_cx, _cy), (bw, bh), _ang = cv2.minAreaRect(cnt)
                    if min(bw, bh) < 5:
                        continue
                    aspect = min(bw, bh) / (max(bw, bh) + 1e-6)
                    extent = area / (bw * bh + 1e-6)
                    if aspect > 0.65 and extent > 0.65:
                        M = cv2.moments(cnt)
                        if M["m00"] > 0:
                            squares.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])
                            areas.append(area)
                if len(squares) < 12:
                    continue
                areas_arr = np.array(areas, dtype=np.float64)
                pts_arr = np.array(squares, dtype=np.float64)
                med_area = float(np.median(areas_arr))
                keep = (areas_arr >= 0.45 * med_area) & (areas_arr <= 1.80 * med_area)
                if keep.sum() < 12 or keep.sum() > 28:
                    continue
                pts_kept = pts_arr[keep]
                det = _fit_6x4_lattice(pts_kept, float(np.sqrt(med_area)), work)
                if det is not None and int(keep.sum()) > best_count:
                    best_count = int(keep.sum())
                    best = det

    if best is None:
        return None
    inv_s = 1.0 / s
    return ChartDetection(
        centres=(best.centres * inv_s).astype(np.float32),
        patch_radius=float(best.patch_radius * inv_s),
        corners=(best.corners * inv_s).astype(np.float32),
    )


def _root_poly_basis(lin: np.ndarray) -> np.ndarray:
    """Degree-2 root-polynomial expansion (Finlayson, Mackiewicz & Hurlbert 2015) + small offset anchor:
    [R, G, B, sqrt(R*G), sqrt(G*B), sqrt(B*R), 1]. Shape (..., 3) -> (..., 7)."""
    x = np.clip(lin, 0.0, None).astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    ones = np.ones_like(r)
    return np.stack([r, g, b, np.sqrt(r * g), np.sqrt(g * b), np.sqrt(b * r), ones], axis=-1)


def calibrate_chart(rgb: np.ndarray, target_srgb: np.ndarray | None = None,
                    detection: ChartDetection | None = None) -> ChartCalibration | None:
    """Detect and calibrate a 24-patch ColorChecker chart to D65 reference (or to another chart's target_srgb).
    Uses weighted Root-Polynomial Color Correction (Finlayson 2015) in linear light with elevated weights on
    skin patches (#1, #2) and neutral ramp (#19..#24), baked into a 33^3 tetrahedral LUT."""
    det = detection if detection is not None else detect_chart(rgb)
    if det is None:
        return None
    tgt_srgb = (CC24_SRGB if target_srgb is None else np.asarray(target_srgb, dtype=np.uint8)).copy()
    meas_srgb = np.array([_sample_patch_rgb(rgb, det.centres[i], det.patch_radius) for i in range(24)], dtype=np.float32)

    meas_lin = TN.srgb_to_lin(meas_srgb / 255.0).astype(np.float64)
    tgt_lin = TN.srgb_to_lin(tgt_srgb.astype(np.float32) / 255.0).astype(np.float64)

    # Weights: prioritise skin patches (#1 Dark Skin, #2 Light Skin) and the 6 neutral patches (#19..#24)
    w = np.ones(24, dtype=np.float64)
    w[0] = 3.5
    w[1] = 3.5
    w[18:24] = 3.0

    Phi = _root_poly_basis(meas_lin)  # (24, 7)
    W_sqrt = np.sqrt(w)[:, None]
    A = Phi * W_sqrt
    Y = tgt_lin * W_sqrt

    # Tikhonov regularisation towards identity [I_3 | 0_{3x4}] so out-of-hull colours extrapolate smoothly
    lam = 1e-4
    I_prior = np.zeros((7, 3), dtype=np.float64)
    I_prior[0, 0] = I_prior[1, 1] = I_prior[2, 2] = 1.0
    ATA = A.T @ A + lam * np.eye(7, dtype=np.float64)
    ATY = A.T @ Y + lam * I_prior
    M = np.linalg.solve(ATA, ATY)  # (7, 3)

    cal_lin = np.clip(Phi @ M, 0.0, 1.0).astype(np.float32)
    cal_srgb = (np.clip(TN.lin_to_srgb(cal_lin), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # Bake onto the 33^3 sRGB LUT grid with OKLab constant-hue gamut safety
    grid_srgb = C.rgb_grid(33)
    grid_lin = TN.srgb_to_lin(grid_srgb)
    grid_cal_lin = ( _root_poly_basis(grid_lin) @ M ).astype(np.float32)
    grid_oklab = TN.lin_to_oklab(np.clip(grid_cal_lin, 0.0, 1.5))
    grid_gamut_lin = TN.oklab_to_lin_gamut(grid_oklab)
    lut_srgb = np.clip(TN.lin_to_srgb(grid_gamut_lin), 0.0, 1.0).astype(np.float32)
    lut_lab = cv2.cvtColor(lut_srgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(33, 33, 33, 3).astype(np.float32)

    # Compute CIEDE2000 report before and after calibration
    lab_meas = C.to_lab(np.clip(meas_srgb + 0.5, 0, 255).astype(np.uint8).reshape(24, 1, 3)).reshape(24, 3)
    lab_cal = C.to_lab(cal_srgb.reshape(24, 1, 3)).reshape(24, 3)
    lab_tgt = C.to_lab(tgt_srgb.reshape(24, 1, 3)).reshape(24, 3)

    de_before = ciede2000(lab_meas, lab_tgt)
    de_after = ciede2000(lab_cal, lab_tgt)

    report = ChartReport(
        de00_before=float(de_before.mean()),
        de00_after=float(de_after.mean()),
        max_de00_before=float(de_before.max()),
        max_de00_after=float(de_after.max()),
        skin_de00=float(de_after[:2].mean()),
        gray_de00=float(de_after[18:24].mean()),
        patch_de00_before=[round(float(v), 2) for v in de_before],
        patch_de00_after=[round(float(v), 2) for v in de_after],
    )
    return ChartCalibration(
        detection=det,
        measured_srgb=meas_srgb,
        calibrated_srgb=cal_srgb,
        target_srgb=tgt_srgb,
        lut_srgb=lut_srgb,
        lut_lab=lut_lab,
        report=report,
    )


def draw_chart_overlay(rgb: np.ndarray, cal: ChartCalibration) -> np.ndarray:
    """Draw a studio diagnostic overlay on `rgb` showing the 24 detected patches, split target/calibrated
    inner swatches, and per-patch ΔE00 indicators."""
    out = rgb.copy()
    r = max(6, int(round(cal.detection.patch_radius * 0.55)))
    corners = cal.detection.corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [corners], True, (240, 175, 50), max(2, r // 5), cv2.LINE_AA)
    for i in range(24):
        cx, cy = int(round(float(cal.detection.centres[i, 0]))), int(round(float(cal.detection.centres[i, 1])))
        tgt = tuple(int(v) for v in cal.target_srgb[i])
        cal_c = tuple(int(v) for v in cal.calibrated_srgb[i])
        # Left half: D65 target, Right half: calibrated patch color
        cv2.rectangle(out, (cx - r, cy - r), (cx, cy + r), tgt, -1)
        cv2.rectangle(out, (cx, cy - r), (cx + r, cy + r), cal_c, -1)
        de = cal.report.patch_de00_after[i]
        border_col = (70, 220, 110) if de < 1.5 else ((240, 190, 60) if de < 3.0 else (235, 75, 75))
        cv2.rectangle(out, (cx - r, cy - r), (cx + r, cy + r), border_col, max(1, r // 6))
    return out
