"""Hardware acceleration manager for ONNX Runtime (DirectML / CPU) and OpenCV OpenCL (cv2.ocl / UMat):
- 'auto': benchmarks DirectML vs CPU (with DX12 shader warmup and Intel iGPU sequential/no-mem-pattern flags)
  and enables OpenCV OpenCL Transparent API when an OpenCL GPU/iGPU is available.
- 'gpu': forces DirectML (with CPU fallback) + OpenCV OpenCL.
- 'ocl': runs ONNX on CPU while accelerating large-frame OpenCV filters via OpenCL (ideal for Intel UHD/Iris iGPU).
- 'cpu': disables both DirectML and OpenCV OpenCL.

Author: Jack4B.
"""
from __future__ import annotations
import json, threading, time
from pathlib import Path
import numpy as np
import cv2
from ..paths import data_dir

MODE = "auto"          # 'auto' | 'gpu' | 'ocl' | 'cpu'
_lock = threading.Lock()
_CHOICES = data_dir() / "device.json"
ACTIVE: dict[str, str] = {}   # model file name -> "CPU" / "GPU" actually used in this run
_OCL_STATE = {"available": False, "enabled": False, "device": ""}
UMAT_MIN_PIXELS = 400_000     # >= 0.4 Mpx avoids OpenCL kernel launch overhead on tiny preview arrays


def ocl_configure(mode: str | None = None) -> dict:
    """Configure OpenCV OpenCL (cv2.ocl) for integrated/discrete GPUs (Intel Iris Xe / UHD / Arc, AMD, NVIDIA)."""
    global MODE
    if mode is not None:
        MODE = mode
    want = MODE in ("auto", "gpu", "ocl")
    avail = False
    dev_name = ""
    enabled = False
    try:
        avail = bool(cv2.ocl.haveOpenCL())
        if avail:
            cv2.ocl.setUseOpenCL(bool(want))
            enabled = bool(want and cv2.ocl.useOpenCL())
            dev = cv2.ocl.Device.getDefault()
            if dev is not None and dev.available():
                dev_name = (dev.name() or "").strip()
        else:
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        enabled = False
    _OCL_STATE.update({"available": avail, "enabled": enabled, "device": dev_name})
    return dict(_OCL_STATE)


def ocl_info() -> dict:
    if not _OCL_STATE["device"] and not _OCL_STATE["available"]:
        ocl_configure(MODE)
    return dict(_OCL_STATE)


def ocl_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur accelerated via OpenCV UMat (OpenCL) on large frames (>= 0.4 Mpx), with CPU fallback."""
    if _OCL_STATE.get("enabled") and arr.shape[0] * arr.shape[1] >= UMAT_MIN_PIXELS:
        try:
            res = cv2.GaussianBlur(cv2.UMat(arr), (0, 0), float(sigma)).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.GaussianBlur(arr, (0, 0), float(sigma))


def ocl_resize(arr: np.ndarray, wh: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize accelerated via OpenCV UMat (OpenCL) on large frames (>= 0.4 Mpx), with CPU fallback."""
    if _OCL_STATE.get("enabled") and arr.shape[0] * arr.shape[1] >= UMAT_MIN_PIXELS:
        try:
            res = cv2.resize(cv2.UMat(arr), wh, interpolation=interpolation).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.resize(arr, wh, interpolation=interpolation)


def _load():
    try:
        return json.loads(_CHOICES.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(d):
    try:
        _CHOICES.parent.mkdir(parents=True, exist_ok=True)
        _CHOICES.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError:
        pass


def _new(path, providers):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in providers:
        # Required by ONNX Runtime DirectML EP (especially on Intel Iris Xe / UHD / Arc shared-memory iGPUs)
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), so, providers=providers)


def _bench(sess, feed, warmup: int = 2, n: int = 2) -> float:
    for _ in range(warmup):
        sess.run(None, feed)
    t = time.perf_counter()
    for _ in range(n):
        sess.run(None, feed)
    return (time.perf_counter() - t) / n


def make_session(path, sample_feed):
    """sample_feed(sess) -> input dict used for the benchmark."""
    ocl_configure(MODE)
    sess = _make(path, sample_feed)
    ACTIVE[Path(path).name] = "GPU" if "DmlExecutionProvider" in sess.get_providers() else "CPU"
    return sess


def summary() -> str:
    """Short Polish description of where AI models and OpenCV run (status bar / settings)."""
    ocl = ocl_info()
    ocl_tag = ""
    if ocl.get("enabled"):
        dname = ocl.get("device") or "iGPU"
        short_d = dname.replace("(R)", "").replace("(TM)", "").replace("Graphics", "").strip()
        if len(short_d) > 22:
            short_d = short_d[:20].rstrip() + "…"
        ocl_tag = f" + OpenCL ({short_d})" if short_d else " + OpenCL"
    if not ACTIVE:
        return f"modele gotowe{ocl_tag}" if ocl_tag else "modele jeszcze nie wczytane"
    vals = set(ACTIVE.values())
    base = "procesor (CPU)" if vals == {"CPU"} else "GPU (DirectML)" if vals == {"GPU"} else "CPU + GPU (DirectML)"
    return base + ocl_tag


def _make(path, sample_feed):
    import onnxruntime as ort
    has_dml = "DmlExecutionProvider" in ort.get_available_providers()
    cpu = ["CPUExecutionProvider"]
    dml = ["DmlExecutionProvider", "CPUExecutionProvider"]
    if MODE in ("cpu", "ocl") or not has_dml:
        return _new(path, cpu)
    if MODE == "gpu":
        try:
            return _new(path, dml)
        except Exception:
            return _new(path, cpu)
    key = f"v2:{Path(path).name}:{Path(path).stat().st_size}"
    with _lock:
        choices = _load()
        pick = choices.get(key)
        if pick in ("cpu", "gpu"):
            try:
                return _new(path, dml if pick == "gpu" else cpu)
            except Exception:
                return _new(path, cpu)
        s_cpu = _new(path, cpu)
        t_cpu = _bench(s_cpu, sample_feed(s_cpu))
        try:
            s_gpu = _new(path, dml)
            t_gpu = _bench(s_gpu, sample_feed(s_gpu))
        except Exception:
            s_gpu, t_gpu = None, float("inf")
        # Offloading ONNX to Intel/AMD/NVIDIA GPU frees CPU cores for Numba & UI even at near-parity latency
        pick = "gpu" if (s_gpu is not None and t_gpu <= 1.05 * t_cpu) else "cpu"
        choices[key] = pick
        choices[key + ":ms"] = [round(t_cpu * 1000), round(t_gpu * 1000) if t_gpu < 1e9 else None]
        _save(choices)
        return s_gpu if pick == "gpu" else s_cpu


def reset():
    ACTIVE.clear()
    ocl_configure(MODE)
    try:
        _CHOICES.unlink()
    except FileNotFoundError:
        pass


def zeros_feed(sess, shape_default=(1, 3, 518, 518)):
    i = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else shape_default[k] for k, d in enumerate(i.shape)]
    return {i.name: np.zeros(shape, np.float32)}
