"""Face detection (YuNet, opencv_zoo, MIT) and the skin mask (faces + neck + hands/arms)."""
import threading
import numpy as np, cv2
from ..paths import bundled_model
from .color import small, guided_refine_rgb

_det = None; _lock = threading.Lock()

def detect(rgb, score=0.45, landmarks=False):
    """Faces as [x, y, w, h] in rgb pixel coordinates; with landmarks=True also a 5x2 array per face
    (YuNet rows 4..13: right eye, left eye, nose tip, mouth right, mouth left), as (faces, lms)."""
    global _det
    sm = small(rgb, 1280); h, w = sm.shape[:2]
    with _lock:
        if _det is None:
            _det = cv2.FaceDetectorYN.create(str(bundled_model("face_detection_yunet_2023mar.onnx")), "", (w, h), score)
        _det.setScoreThreshold(float(score))
        _det.setInputSize((w, h)); _, f = _det.detect(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR))
    s = rgb.shape[1]/w
    boxes = [] if f is None else [np.asarray(x[:4]*s, np.float32) for x in f]
    if not landmarks: return boxes
    return boxes, ([] if f is None else [np.asarray(x[4:14]*s, np.float32).reshape(5, 2) for x in f])

def pick_faces(rgb, landmarks=False):
    """Main subjects only: drop faces narrower than 0.45x the widest (posters, distant background people),
    keeping secondary couple/group faces even when partially occluded by glasses or hats.
    landmarks=True returns (faces, lms) with the matching 5x2 landmark arrays."""
    f, l = detect(rgb, landmarks=True)
    if not f: return ([], []) if landmarks else []
    mw = max(x[2] for x in f); keep = [i for i, x in enumerate(f) if x[2] >= 0.45*mw]
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
