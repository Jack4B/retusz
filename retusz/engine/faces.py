"""Face detection (YuNet, opencv_zoo, MIT) and the skin mask (faces + neck + hands/arms)."""
import threading
import numpy as np, cv2
from ..paths import bundled_model
from .color import small, guided_refine_rgb

_det = None; _lock = threading.Lock()

def verify_face_landmarks(rgb: np.ndarray, box: np.ndarray, lm: np.ndarray) -> bool:
    """Two-stage anatomical landmark & core luminance plausibility check for low-threshold YuNet candidates:
    rejects false positives on background patterns/posters while keeping real faces wearing thick/novelty glasses."""
    x, y, w, h = (float(v) for v in box[:4])
    if w < 10.0 or h < 10.0 or lm is None or len(lm) < 5:
        return False
    pts = np.asarray(lm, np.float32).reshape(5, 2)
    e_r, e_l, nose, m_r, m_l = pts
    # Inter-ocular distance & roll angle check
    dx_e = float(abs(e_l[0] - e_r[0]))
    dy_e = float(abs(e_l[1] - e_r[1]))
    d_eyes = float(np.hypot(dx_e, dy_e))
    if not (0.14 * w <= d_eyes <= 0.72 * w):
        return False
    if dy_e / max(dx_e, 1e-3) > 0.72:
        return False
    e_mid_y = 0.5 * float(e_r[1] + e_l[1])
    m_mid_y = 0.5 * float(m_r[1] + m_l[1])
    if m_mid_y <= e_mid_y + 0.10 * h:
        return False
    if not (e_mid_y - 0.08 * h <= float(nose[1]) <= m_mid_y + 0.04 * h):
        return False
    # Check face core isn't pure black/blown-out solid patch
    H, W = rgb.shape[:2]
    x0, x1 = max(0, int(x + 0.2 * w)), min(W, int(x + 0.8 * w))
    y0, y1 = max(0, int(y + 0.2 * h)), min(H, int(y + 0.8 * h))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return False
    core_lum = float(rgb[y0:y1, x0:x1].mean())
    return 14.0 <= core_lum <= 249.0


def nms_boxes_iou(boxes: list, lms: list, scores: list | None = None, iou_thr: float = 0.42):
    """Cross-tile / multi-scale greedy IoU deduplication (YuNet already applies single-scale NMS internally)."""
    if not boxes:
        return [], []
    n = len(boxes)
    sc = [float(scores[i]) if scores is not None and i < len(scores) else 1.0 for i in range(n)]
    order = sorted(range(n), key=lambda i: (sc[i], float(boxes[i][2] * boxes[i][3])), reverse=True)
    kept_b, kept_l = [], []
    for i in order:
        bx = boxes[i]
        x1, y1, w1, h1 = (float(v) for v in bx[:4])
        a1 = max(1.0, w1 * h1)
        dup = False
        for kb in kept_b:
            x2, y2, w2, h2 = (float(v) for v in kb[:4])
            ix0, iy0 = max(x1, x2), max(y1, y2)
            ix1, iy1 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
            iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
            inter = iw * ih
            union = a1 + max(1.0, w2 * h2) - inter
            if inter / max(union, 1e-6) >= iou_thr:
                dup = True
                break
        if not dup:
            kept_b.append(bx)
            kept_l.append(lms[i] if i < len(lms) else None)
    return kept_b, kept_l


def _detect_raw(bgr_img: np.ndarray, score: float):
    global _det
    h, w = bgr_img.shape[:2]
    with _lock:
        if _det is None:
            _det = cv2.FaceDetectorYN.create(str(bundled_model("face_detection_yunet_2023mar.onnx")), "", (w, h), score)
        _det.setScoreThreshold(float(score))
        _det.setInputSize((w, h))
        _, f = _det.detect(bgr_img)
    return f


def detect(rgb, score=0.45, landmarks=False):
    """Faces as [x, y, w, h] in rgb pixel coordinates; with landmarks=True also a 5x2 array per face.
    Uses two-stage landmark verification for candidates in [0.45, 0.68) and overlapping horizontal tile scan
    on wide/high-res group shots so smaller faces in couples/groups are reliably detected."""
    sm = small(rgb, 1280); h, w = sm.shape[:2]
    bgr = cv2.cvtColor(sm, cv2.COLOR_RGB2BGR)
    f = _detect_raw(bgr, score)
    s = rgb.shape[1] / w

    boxes, lms, scs = [], [], []
    if f is not None:
        for row in f:
            bx = np.asarray(row[:4] * s, np.float32)
            lm = np.asarray(row[4:14] * s, np.float32).reshape(5, 2)
            conf = float(row[14]) if len(row) > 14 else 1.0
            if conf >= 0.68 or verify_face_landmarks(rgb, bx, lm):
                boxes.append(bx); lms.append(lm); scs.append(conf)

    # Tiled pass for wide/large group shots when full-frame pass found <= 1 small face
    H, W = rgb.shape[:2]
    if max(H, W) >= 1800 and W >= H and (not boxes or max(b[2] for b in boxes) < 0.14 * W):
        tw = int(round(0.62 * W))
        for x_off in (0, W - tw):
            crop = rgb[:, x_off:x_off + tw]
            c_sm = small(crop, 960); ch, cw = c_sm.shape[:2]
            cs = crop.shape[1] / cw
            cf = _detect_raw(cv2.cvtColor(c_sm, cv2.COLOR_RGB2BGR), max(score, 0.50))
            if cf is not None:
                for row in cf:
                    bx = np.asarray(row[:4] * cs, np.float32); bx[0] += x_off
                    lm = np.asarray(row[4:14] * cs, np.float32).reshape(5, 2); lm[:, 0] += x_off
                    conf = float(row[14]) if len(row) > 14 else 1.0
                    if conf >= 0.68 or verify_face_landmarks(rgb, bx, lm):
                        boxes.append(bx); lms.append(lm); scs.append(conf)
        boxes, lms = nms_boxes_iou(boxes, lms, scs, iou_thr=0.42)

    if not landmarks:
        return boxes
    return boxes, lms


def pick_faces(rgb, landmarks=False):
    """Main subjects only: drop faces narrower than 0.45x the widest (posters, distant background people),
    keeping secondary couple/group faces even when partially occluded by glasses or hats.
    landmarks=True returns (faces, lms) with the matching 5x2 landmark arrays."""
    f, l = detect(rgb, landmarks=True)
    if not f:
        return ([], []) if landmarks else []
    mw = max(x[2] for x in f)
    keep = [i for i, x in enumerate(f) if x[2] >= 0.45 * mw]
    fs = [f[i] for i in keep]
    return (fs, [l[i] for i in keep]) if landmarks else fs

def skin_mask(rgb, fcs, hands=True, subject=None, hue_tol: float = 0.0, protect_features: float = 0.0, lms=None):
    """Soft 0..1 mask. Per-face skin colour & luminance calibrated from the inner cheek/forehead core
    (rejects dark brown hair and warm background patches behind hair/ears), connected-component seeded
    from the face core; hands/arms matched tightly to the clean face skin model; edges snapped to the
    image's own RGB boundaries with a 3-channel guided filter (He, Sun, Tang 2010).
    - hue_tol (-1..+1): tightens (<0) or widens (>0) the Mahalanobis colour/luminance & spatial envelope.
    - protect_features (0..1): excludes eyes, eyelashes, and lip vermilion around the 5 YuNet landmarks."""
    H, W = rgb.shape[:2]
    if not len(fcs):
        return np.zeros((H, W), np.float32)
    ht = float(np.clip(hue_tol, -1.0, 1.0))
    tau = float(2.0 ** (1.35 * ht))
    sp_scale = float(np.clip(1.0 + (0.28 * ht if ht < 0 else 0.35 * ht), 0.65, 1.40))
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    Y = ycc[..., 0]; cc_all = ycc[..., 1:]
    cr_lo = max(124, int(round(133 - 10 * ht)))
    cr_hi = min(192, int(round(180 + 10 * ht)))
    cb_lo = max(65, int(round(77 - 10 * ht)))
    cb_hi = min(148, int(round(135 + 10 * ht)))
    skin_rng = cv2.inRange(ycc.astype(np.uint8), (16, cr_lo, cb_lo), (255, cr_hi, cb_hi)) > 0
    face_all = np.zeros((H, W), np.uint8)
    m = np.zeros((H, W), np.float32)
    for idx, (x, y, w, h) in enumerate(fcs):
        cx, cy = int(x+w/2), int(y+h*0.55)
        ell = np.zeros((H, W), np.uint8)
        cv2.ellipse(ell, (cx, cy), (max(1, int(w*0.72*sp_scale)), max(1, int(h*0.92*sp_scale))), 0, 0, 360, 1, -1)
        cv2.ellipse(ell, (cx, int(y+h*1.25)), (max(1, int(w*0.52*sp_scale)), max(1, int(h*0.48*sp_scale))), 0, 0, 360, 1, -1)
        face_all |= ell
        core = np.zeros((H, W), np.uint8)
        cv2.ellipse(core, (cx, int(y+h*0.52)), (max(1, int(w*0.36)), max(1, int(h*0.42))), 0, 0, 360, 1, -1)
        cm = (core > 0) & skin_rng
        if cm.sum() >= 30:
            yc = Y[cm]
            # drop darkest 25 % inside core (eyes, brows, glasses frames, nostrils) before calibrating skin level
            clean = cm & (Y >= np.percentile(yc, 25))
            if clean.sum() < 20: clean = cm
            y_med = float(np.median(Y[clean]))
            y_mad = float(1.4826*np.median(np.abs(Y[clean] - y_med)))
            y_lo = max(14.0, min(y_med - max(42.0, 3.2*y_mad)*tau, y_med*0.55), y_med*0.45) / max(0.7, 1.0 + 0.25*ht)
            y_hi = min(255.0, y_med + max(55.0, 3.5*y_mad)*tau)
            cc = cc_all[clean]; mu = cc.mean(0)
            ic = np.linalg.inv(np.cov(cc.T) + np.eye(2)*4.0)
            # evaluate only on the face+neck bbox for speed
            y0, y1 = max(0, int(y - 0.5*h)), min(H, int(y + 1.8*h))
            x0, x1 = max(0, int(x - 0.4*w)), min(W, int(x + 1.4*w))
            d = cc_all[y0:y1, x0:x1] - mu
            md = np.einsum("...i,ij,...j->...", d, ic, d)
            sub_y = Y[y0:y1, x0:x1]
            fm = (ell[y0:y1, x0:x1] > 0) & skin_rng[y0:y1, x0:x1] & (sub_y >= y_lo) & (sub_y <= y_hi) & (md < 10.0 * tau)
            # seed from the inner face core: warm backdrop patches separated by hair/shadow are dropped
            n, lab = cv2.connectedComponents(fm.astype(np.uint8), connectivity=8)
            seed_sub = clean[y0:y1, x0:x1]
            keep = np.zeros(n, bool); keep[np.unique(lab[seed_sub & (lab > 0)])] = True; keep[0] = False
            comp_m = (keep[lab] if keep.any() else fm).astype(np.float32)
            if abs(ht) > 1e-4:
                # Continuous Mahalanobis posterior weighting when user refines hue tolerance
                prob = np.exp(-np.maximum(0.0, md - 2.0 * tau) / (8.0 * tau + 1e-4)).astype(np.float32)
                comp_m = comp_m * (0.35 + 0.65 * prob) if ht > 0 else comp_m * prob
            m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], comp_m)
        else:
            m = np.maximum(m, ((ell > 0) & skin_rng).astype(np.float32))
    if hands and (m > 0.5).sum() > 150:
        m_bin = m > 0.5
        cc = cc_all[m_bin]; mu = cc.mean(0); ic = np.linalg.inv(np.cov(cc.T) + np.eye(2)*3.0)
        d = cc_all - mu; md = np.einsum("...i,ij,...j->...", d, ic, d)
        y_p05, y_p95 = np.percentile(Y[m_bin], (5, 95))
        y_lo = max(16.0, float(y_p05)*0.78 / max(0.75, 1.0 + 0.2*ht))
        y_hi = min(255.0, float(y_p95)*1.22 + 12.0)
        cand = (md < 4.5 * tau) & skin_rng & (Y >= y_lo) & (Y <= y_hi)
        gate = np.zeros((H, W), np.uint8)
        for (x, y, w, h) in fcs:
            cv2.rectangle(gate, (int(x-1.2*w), int(y)), (int(x+2.2*w), int(y+7*h)), 1, -1)
        cand &= (gate > 0) & ~(face_all > 0)
        if subject is not None:
            sub = cv2.resize(subject.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            ks = max(3, int(max(W, H)*0.015)) | 1
            cand &= cv2.dilate((sub > 0.35).astype(np.uint8), np.ones((ks, ks), np.uint8)) > 0
        fw = float(np.median([f[2] for f in fcs])); fa = float(np.median([f[2]*f[3] for f in fcs]))
        ko = max(3, int(fw*0.08)) | 1
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]; bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            if not ((0.02*fa < a < 0.65*fa) and a/(bw*bh+1e-6) > 0.28 and a/max(bw, bh) > 0.12*fw):
                continue
            comp = lab == i
            dm = np.median(cc_all[comp], axis=0) - mu
            ym = float(np.median(Y[comp]))
            if float(dm @ ic @ dm) < 2.5 * tau and (y_lo*1.02 <= ym <= y_hi):
                keep[i] = True
        m = np.maximum(m, keep[lab].astype(np.float32))

    # Anatomical feature exclusion (eyes, eyelashes, lips) when protect_features > 0
    pf = float(np.clip(protect_features, 0.0, 1.0))
    if pf > 1e-4:
        feat = np.zeros((H, W), np.float32)
        for idx, (x, y, w, h) in enumerate(fcs):
            pts = None
            if lms is not None and idx < len(lms) and lms[idx] is not None:
                pts = np.asarray(lms[idx], np.float32).reshape(-1, 2)
            else:
                pts = np.array([
                    [x + 0.32 * w, y + 0.38 * h],
                    [x + 0.68 * w, y + 0.38 * h],
                    [x + 0.50 * w, y + 0.56 * h],
                    [x + 0.36 * w, y + 0.75 * h],
                    [x + 0.64 * w, y + 0.75 * h],
                ], np.float32)
            if len(pts) >= 5:
                rx_e, ry_e = max(2, int(round(w * 0.13))), max(2, int(round(h * 0.09)))
                for ep in (pts[0], pts[1]):
                    cv2.ellipse(feat, (int(round(ep[0])), int(round(ep[1]))), (rx_e, ry_e), 0, 0, 360, 1.0, -1)
                mcx = int(round(0.5 * (pts[3][0] + pts[4][0])))
                mcy = int(round(0.5 * (pts[3][1] + pts[4][1])))
                rx_m = max(2, int(round(max(abs(pts[4][0] - pts[3][0]) * 0.62, w * 0.16))))
                ry_m = max(2, int(round(h * 0.08)))
                cv2.ellipse(feat, (mcx, mcy), (rx_m, ry_m), 0, 0, 360, 0.85, -1)
        feat = cv2.GaussianBlur(feat, (0, 0), max(1.0, max(W, H) * 0.006))
        m = m * (1.0 - pf * 0.88 * np.clip(feat, 0.0, 1.0))

    k = max(3, int(max(W, H)*0.003)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # snap to RGB edges so the skin LUT does not bleed into hair or backdrop, then feather lightly
    r = max(2, int(max(W, H)*0.008))
    m = np.clip(guided_refine_rgb(rgb.astype(np.float32)/255.0, m, r, 2e-3), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), max(0.5, max(W, H)*0.002)).astype(np.float32)
