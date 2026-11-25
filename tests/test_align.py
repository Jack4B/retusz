import numpy as np, cv2, pytest
from synthetic import textured, grade, make_ai, warp_error_px
from retusz.engine import align as A

@pytest.fixture(scope="module")
def scene():
    o, disc = textured(1200, 1800, seed=1)
    return o, grade(o, disc)

@pytest.mark.parametrize("k,angle,shift,crop", [(3, 1.5, (7.3, -4.6), 0.06), (2, -2.0, (-3.1, 5.2), 0.03), (4, 0.4, (2.2, 1.7), 0.10)])
def test_recovers_affine_on_regraded_relit_ai(scene, k, angle, shift, crop):
    o, gt = scene
    ai, T = make_ai(gt, k, angle, shift, crop)
    al = A.register(o, ai)
    err = warp_error_px(T, al.full_to_ai, o.shape, ai.shape)
    assert err < 0.5, err
    assert al.residual_px < 1.0
    assert abs(al.scale - 1/k) < 0.01

def test_square_crop_of_landscape(scene):
    o, gt = scene
    ai, T = make_ai(gt, 3, 0.8, (0, 0), 0.0, size=(380, 380))
    al = A.register(o, ai)
    assert warp_error_px(T, al.full_to_ai, o.shape, ai.shape) < 0.5

def test_unrelated_image_refused(scene):
    o, _ = scene
    other, _ = textured(600, 900, seed=7)
    with pytest.raises(A.AlignmentError):
        A.register(o, other)

def test_flow_corrects_local_drift(scene):
    o, gt = scene
    ai, T = make_ai(gt, 3, 1.0, (0, 0), 0.05)
    h, w = ai.shape[:2]; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = 1.2*np.sin(xx/90) ; dy = 1.0*np.cos(yy/70)        # AI px drift (~3 full-res px)
    ai2 = cv2.remap(ai, xx+dx, yy+dy, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    def true_map(P):                                        # full -> ai2: inverse of the drift
        q = (T @ np.c_[P, np.ones(len(P))].T).T
        for _ in range(10):                                 # solve q2 + d(q2) = q
            q2 = q - np.c_[1.2*np.sin(q[:, 0]/90), 1.0*np.cos(q[:, 1]/70)] if _ == 0 else q - np.c_[1.2*np.sin(q2[:, 0]/90), 1.0*np.cos(q2[:, 1]/70)]
        return q2
    rigid = A.register(o, ai2, flow=False); local = A.register(o, ai2, flow=True)
    def err(al):
        H, W = o.shape[:2]; ys, xs = np.meshgrid(np.linspace(60, H-60, 20), np.linspace(60, W-60, 20), indexing="ij")
        P = np.c_[xs.ravel(), ys.ravel()]
        return float(np.linalg.norm(true_map(P) - al.full_to_ai(P), axis=1).mean()*3)
    assert err(local) < 0.6 and err(local) < 0.5*err(rigid), (err(local), err(rigid))
