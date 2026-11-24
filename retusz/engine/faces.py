"""Face detection (YuNet, opencv_zoo, MIT) and the skin mask (faces + neck + hands/arms)."""
import threading
import numpy as np, cv2
from ..paths import bundled_model
from .color import small

_det = None; _lock = threading.Lock()

def detect(rgb, score=0.7):
    """Faces as [x, y, w, h] in rgb pixel coordinates."""
    global _det
    sm = small(rgb, 1280); h, w = sm.shape[:2]
    with _lock:
        if _det is None:
            _det = cv2.FaceDetectorYN.create(str(bundled_model("face_detection_yunet_2023mar.onnx")), "", (w, h), score)
        _det.setInputSize((w, h)); _, f = _det.detect(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR))
    s = rgb.shape[1]/w
    return [] if f is None else [np.asarray(x[:4]*s, np.float32) for x in f]

def pick_faces(rgb):
    """Main subjects only: drop faces narrower than 0.6x the widest (posters, background people)."""
    f = detect(rgb)
    if not f: return []
    mw = max(x[2] for x in f); return [x for x in f if x[2] >= 0.6*mw]

def skin_mask(rgb, fcs, hands=True):
    """Soft 0..1 mask. Faces+neck = ellipses ∩ YCrCb skin range. Hands/arms = pixels matching the
    detected faces' own skin colour (Mahalanobis distance in CrCb), near the person, size/thickness
    filtered so that sofas, walls and LED strips do not pass."""
    H, W = rgb.shape[:2]
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    skin_rng = cv2.inRange(ycc.astype(np.uint8), (0, 133, 77), (255, 180, 135)) > 0
    face = np.zeros((H, W), np.uint8)
    for (x, y, w, h) in fcs:
        cx, cy = int(x+w/2), int(y+h*0.55)
        cv2.ellipse(face, (cx, cy), (max(1, int(w*0.75)), max(1, int(h*0.95))), 0, 0, 360, 1, -1)
        cv2.ellipse(face, (cx, int(y+h*1.25)), (max(1, int(w*0.55)), max(1, int(h*0.5))), 0, 0, 360, 1, -1)
    m = (face > 0) & skin_rng
    if hands and len(fcs) and m.sum() > 200:
        cc = ycc[..., 1:][m]; mu = cc.mean(0); ic = np.linalg.inv(np.cov(cc.T)+np.eye(2)*2)
        d = ycc[..., 1:]-mu; md = np.einsum("...i,ij,...j->...", d, ic, d)
        ymin = np.percentile(ycc[..., 0][m], 3)
        cand = (md < 6.0) & skin_rng & (ycc[..., 0] > ymin*0.8)
        gate = np.zeros((H, W), np.uint8)
        for (x, y, w, h) in fcs:
            cv2.rectangle(gate, (int(x-2.5*w), int(y)), (int(x+3.5*w), int(y+7*h)), 1, -1)
        cand &= gate > 0; cand &= ~(face > 0)
        fw = float(np.median([f[2] for f in fcs])); fa = float(np.median([f[2]*f[3] for f in fcs]))
        ko = max(3, int(fw*0.08)) | 1
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]; bw, bh = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
            keep[i] = (0.02*fa < a < 1.2*fa) and a/(bw*bh+1e-6) > 0.25 and a/max(bw, bh) > 0.12*fw
        m = m | keep[lab]
    m = m.astype(np.float32)
    k = max(3, int(max(W, H)*0.003)) | 1
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), max(W, H)*0.004)
