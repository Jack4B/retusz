"""ModFlows colour transfer (Larchenko et al., AAAI 2025; code and weights MIT).
An EfficientNet encoder (ONNX) predicts the weights of a tiny neural ODE per image; colours flow
source -> latent -> reference. The mapping is pointwise, so it is evaluated on a grid and baked to a LUT."""
import threading
import numpy as np, onnxruntime as ort
from PIL import Image
from .color import apply_lut

_sess = {}; _lock = threading.Lock()

from . import device

def session(path):
    path = str(path)
    with _lock:
        if path not in _sess:
            _sess[path] = device.make_session(path, device.zeros_feed)
        return _sess[path]

def reset_sessions():
    with _lock: _sess.clear()

def device_name(path):
    p = session(path).get_providers()[0]
    return "GPU (DirectML)" if p == "DmlExecutionProvider" else "CPU"

def preprocess(rgb, n):
    a = np.asarray(Image.fromarray(rgb).resize((n, n), Image.BILINEAR), np.float32)/255
    return a.reshape(1, 3, n, n)  # raw reshape, same as the reference implementation

def embed(rgb, model):
    s = session(model); n = s.get_inputs()[0].shape[2]
    with _lock:
        return s.run(None, {s.get_inputs()[0].name: preprocess(rgb, n)})[0][0].astype(np.float32)

def _params(e):
    H = (len(e)-3)//8; i = 4*H; j = i+H
    return e[:i].reshape(H, 4), e[i:j], e[j:j+3*H].reshape(3, H), e[j+3*H:j+3*H+3]

def _vel(p, z, t):
    W1, b1, W2, b2 = p
    return np.tanh(z@W1[:, :3].T + t*W1[:, 3] + b1)@W2.T + b2

def _fwd(p, x, n):
    z = x.copy()
    for i in range(n): z = z + _vel(p, z, i/n)/n
    return z

def _inv(p, x, n):
    z = x.copy()
    for i in range(n): z = z - _vel(p, z, 1-i/n)/n
    return z

def lut_from_embeddings(e_src, e_ref, N=33, steps=8, flow_N=17):
    """RGB->RGB LUT (values 0..1)."""
    g = np.linspace(0, 1, flow_N, dtype=np.float32)
    G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    c = np.clip(_inv(_params(e_ref), _fwd(_params(e_src), G, steps), steps), 0, 1).reshape(flow_N, flow_N, flow_N, 3).astype(np.float32)
    if flow_N == N: return c
    g2 = np.linspace(0, 255, N, dtype=np.float32)
    G2 = np.stack(np.meshgrid(g2, g2, g2, indexing="ij"), -1)
    return np.clip(apply_lut(G2, c), 0, 1).astype(np.float32)
