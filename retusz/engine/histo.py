"""Histogram and clipping warning for the result (values as exported, in the working colour space).

Highlight clip = any channel >= HI (a blown channel loses detail even if the others survive);
shadow clip = all channels <= LO (pure black)."""
from __future__ import annotations
import numpy as np

LO, HI = 2, 254
HI_COLOR = (255, 40, 40)
LO_COLOR = (40, 90, 255)
_Y = np.array([0.2126, 0.7152, 0.0722], np.float32)

def histogram(rgb, bins=256):
    """Normalised histograms of R, G, B and luminance (each sums to 1)."""
    a = rgb.reshape(-1, 3); n = max(1, a.shape[0]); out = {}
    for i, k in enumerate("rgb"):
        out[k] = np.bincount(a[:, i], minlength=256)[:256].astype(np.float64)/n
    y = np.clip(a.astype(np.float32) @ _Y + 0.5, 0, 255).astype(np.uint8)
    out["l"] = np.bincount(y, minlength=256)[:256].astype(np.float64)/n
    if bins != 256:
        for k in out: out[k] = out[k].reshape(bins, -1).sum(1)
    return out

def clipping(rgb, lo=LO, hi=HI):
    """(shadow fraction, highlight fraction, shadow mask, highlight mask)."""
    mhi = rgb.max(-1) >= hi; mlo = rgb.max(-1) <= lo
    n = max(1, mhi.size)
    return float(mlo.sum())/n, float(mhi.sum())/n, mlo, mhi

def overlay(rgb, lo=LO, hi=HI):
    """Copy of rgb with clipped highlights red and clipped shadows blue."""
    _, _, mlo, mhi = clipping(rgb, lo, hi)
    out = rgb.copy(); out[mhi] = HI_COLOR; out[mlo] = LO_COLOR
    return out


SKIN_LINE_DEG = 52.0  # Universal portrait skin hue locus in OKLCh (~52° +- 8°)


def skin_vectorscope(rgb: np.ndarray, mask: np.ndarray | None = None, size: int = 64) -> dict:
    """Compute a 2D OKLab/OKLCh chroma vectorscope grid (size x size) and estimate the chroma-weighted
    mean skin hue angle in degrees relative to the universal 52° OKLCh Skin-Tone Line."""
    from . import tone as TN
    h, w = rgb.shape[:2]
    step = max(1, min(h, w) // 96)
    sub = rgb[::step, ::step].astype(np.float32) / 255.0
    ok = TN.lin_to_oklab(TN.srgb_to_lin(sub))
    a = ok[..., 1].ravel()
    b = ok[..., 2].ravel()
    C = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0

    if mask is not None and mask.size > 0:
        import cv2
        ms = cv2.resize(mask.astype(np.float32), (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_NEAREST).ravel()
        w_sk = np.clip(ms, 0.0, 1.0) * C
    else:
        # Weight pixels within the broad skin-adjacent sector (10°..110°) by their chroma
        dh_broad = np.abs(((hue - SKIN_LINE_DEG + 180.0) % 360.0) - 180.0)
        w_sk = np.clip(1.0 - dh_broad / 55.0, 0.05, 1.0) * C

    if float(w_sk.sum()) > 1e-6:
        mean_a = float((a * w_sk).sum() / w_sk.sum())
        mean_b = float((b * w_sk).sum() / w_sk.sum())
        hue_deg = float((np.degrees(np.arctan2(mean_b, mean_a)) + 360.0) % 360.0)
        mean_c = float(np.hypot(mean_a, mean_b))
    else:
        hue_deg = SKIN_LINE_DEG
        mean_c = 0.0
    delta_deg = float(((hue_deg - SKIN_LINE_DEG + 180.0) % 360.0) - 180.0)

    # Project (a, b) in [-0.18, 0.18] onto a size x size density grid (x = +a, y = -b so +b is up)
    r_max = 0.18
    gx = np.clip(((a / r_max * 0.5 + 0.5) * (size - 1)).astype(np.int32), 0, size - 1)
    gy = np.clip((((-b) / r_max * 0.5 + 0.5) * (size - 1)).astype(np.int32), 0, size - 1)
    grid = np.zeros((size, size), np.float32)
    np.add.at(grid, (gy, gx), 1.0)
    mx = float(grid.max())
    if mx > 0:
        grid = np.sqrt(grid / mx)
    return {
        "grid": grid,
        "hue_deg": hue_deg,
        "delta_deg": delta_deg,
        "mean_c": mean_c,
        "skin_line_deg": SKIN_LINE_DEG,
    }


def ai_mask_overlay(
    rgb: np.ndarray,
    mask_q: np.ndarray | None,
    faces_n: list | tuple = (),
    lms_n: list | None = None,
    subject_fg: np.ndarray | None = None,
) -> np.ndarray:
    """Diagnostic AI overlay for the preview canvas (never used in export):
    - Amber semi-transparent tint (#F5A33B) on the detected skin working mask
    - Cyan contour (#4FC3F7) around the 3D subject / background separation boundary
    - Corner-bracketed face detection boxes + 5 anatomical YuNet landmarks (eyes, nose, mouth)."""
    import cv2
    h, w = rgb.shape[:2]
    out = rgb.astype(np.float32).copy()

    # 1. Skin working mask tint (#F5A33B) + skin boundary contour
    if mask_q is not None and mask_q.size > 0:
        m = cv2.resize(mask_q.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        m_c = np.clip(m, 0.0, 1.0)[..., None]
        amber = np.array([245.0, 163.0, 59.0], np.float32)
        out = out * (1.0 - 0.42 * m_c) + amber * (0.42 * m_c)

    # 2. 3D Subject / Background separation contour in cyan (#4FC3F7)
    if subject_fg is not None and subject_fg.size > 0:
        fg = cv2.resize(subject_fg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        fg_bin = (fg > 0.5).astype(np.uint8)
        edge = cv2.morphologyEx(fg_bin, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        out[edge] = np.array([79.0, 195.0, 247.0], np.float32)

    canvas = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    lw = max(2, min(h, w) // 320)

    # 3. Detected AI Face bounding boxes + 5 anatomical landmarks (eyes, nose, mouth)
    for idx, f in enumerate(faces_n or ()):
        fx, fy, fw, fh = [int(round(float(v) * w)) for v in f]
        x0, y0 = max(0, fx), max(0, fy)
        x1, y1 = min(w - 1, fx + fw), min(h - 1, fy + fh)
        # Draw corner brackets in warm amber
        col_box = (245, 163, 59)
        clen = max(8, min(fw, fh) // 4)
        for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(canvas, (cx, cy), (cx + sx * clen, cy), col_box, lw, cv2.LINE_AA)
            cv2.line(canvas, (cx, cy), (cx, cy + sy * clen), col_box, lw, cv2.LINE_AA)
        cv2.putText(
            canvas, f"AI Twarz #{idx+1}", (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, max(0.45, min(h, w) / 1400.0), col_box, max(1, lw - 1), cv2.LINE_AA
        )

        # Draw 5 anatomical landmarks if available (left eye, right eye, nose, mouth left, mouth right)
        pts = None
        if lms_n is not None and idx < len(lms_n) and lms_n[idx] is not None:
            pts = (np.asarray(lms_n[idx], np.float32) * w).astype(np.int32)
        else:
            # Fallback canonical landmarks from face box
            pts = np.array([
                [fx + int(0.32 * fw), fy + int(0.38 * fh)],
                [fx + int(0.68 * fw), fy + int(0.38 * fh)],
                [fx + int(0.50 * fw), fy + int(0.56 * fh)],
                [fx + int(0.36 * fw), fy + int(0.75 * fh)],
                [fx + int(0.64 * fw), fy + int(0.75 * fh)],
            ], np.int32)
        if pts is not None and len(pts) >= 5:
            r_eye = max(4, int(round(fw * 0.08)))
            # Eyes (emerald green #66BB6A)
            for ep in (pts[0], pts[1]):
                cv2.circle(canvas, (int(ep[0]), int(ep[1])), r_eye, (102, 220, 110), max(1, lw - 1), cv2.LINE_AA)
                cv2.circle(canvas, (int(ep[0]), int(ep[1])), max(2, lw), (102, 220, 110), -1, cv2.LINE_AA)
            # Nose tip (amber #FFB74D)
            cv2.circle(canvas, (int(pts[2][0]), int(pts[2][1])), max(2, lw + 1), (255, 183, 77), -1, cv2.LINE_AA)
            # Mouth corners & lip line (rose #F06292)
            p_ml = (int(pts[3][0]), int(pts[3][1]))
            p_mr = (int(pts[4][0]), int(pts[4][1]))
            cv2.line(canvas, p_ml, p_mr, (240, 98, 146), max(1, lw - 1), cv2.LINE_AA)
            cv2.circle(canvas, p_ml, max(2, lw), (240, 98, 146), -1, cv2.LINE_AA)
            cv2.circle(canvas, p_mr, max(2, lw), (240, 98, 146), -1, cv2.LINE_AA)

    return canvas


