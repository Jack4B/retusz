"""Hardware acceleration manager for OpenCV OpenCL (cv2.ocl / UMat on Intel iGPU, AMD, NVIDIA)
and ONNX Runtime (DirectML / CPU):
- 'auto': enables OpenCV OpenCL UMat acceleration whenever an OpenCL GPU/iGPU is available,
  and benchmarks DirectML vs CPU for ONNX neural models (with DX12 shader warmup and shared-RAM flags).
- 'ocl': runs ONNX neural models on CPU while offloading OpenCV image-processing filters
  (blur, boxFilter, guided filter, bilateral denoise, RL deconvolution, resize, bicubic rotation)
  onto the graphics card via OpenCL UMat — ideal for integrated Intel HD / UHD / Iris Xe / Arc GPUs.
- 'gpu': forces DirectML (with CPU fallback) + OpenCV OpenCL UMat.
- 'cpu': disables both DirectML and OpenCV OpenCL.

Author: Jack4B.
"""
from __future__ import annotations
import json, threading, time
from pathlib import Path
import numpy as np
import cv2
from ..paths import data_dir

MODE = "auto"          # 'auto' | 'ocl' | 'gpu' | 'cpu'
_lock = threading.Lock()
_CHOICES = data_dir() / "device.json"
ACTIVE: dict[str, str] = {}   # model file name -> "CPU" / "GPU" actually used in this run
_OCL_STATE = {"available": False, "enabled": False, "device": "", "version": "", "compute_units": 0}
UMAT_MIN_PIXELS = 150_000     # >= 0.15 Mpx (~450x335): accelerates all 1200/1600/2400 px previews & 24 Mpx exports


def ocl_configure(mode: str | None = None) -> dict:
    """Configure OpenCV OpenCL (cv2.ocl / UMat) for integrated & discrete GPUs."""
    global MODE
    if mode is not None:
        MODE = mode
    want = MODE in ("auto", "gpu", "ocl")
    avail = False
    dev_name = ""
    ocl_ver = ""
    cu = 0
    enabled = False
    try:
        avail = bool(cv2.ocl.haveOpenCL())
        if avail:
            cv2.ocl.setUseOpenCL(bool(want))
            enabled = bool(want and cv2.ocl.useOpenCL())
            dev = cv2.ocl.Device.getDefault()
            if dev is not None and dev.available():
                dev_name = (dev.name() or "").strip()
                ocl_ver = (dev.OpenCLVersion() or "").strip()
                cu = int(dev.maxComputeUnits() or 0)
        else:
            cv2.ocl.setUseOpenCL(False)
    except Exception:
        enabled = False
    _OCL_STATE.update({
        "available": avail,
        "enabled": enabled,
        "device": dev_name,
        "version": ocl_ver,
        "compute_units": cu,
    })
    return dict(_OCL_STATE)


def ocl_info() -> dict:
    if not _OCL_STATE["device"] and not _OCL_STATE["available"]:
        ocl_configure(MODE)
    return dict(_OCL_STATE)


def ocl_status_line() -> str:
    """Human-readable Polish status of the OpenCL graphics card for Settings & diagnostics."""
    info = ocl_info()
    if info.get("enabled"):
        dn = info.get("device") or "Karta graficzna OpenCL"
        cu = info.get("compute_units") or 0
        cu_txt = f" · {cu} bloków EU/CU" if cu > 0 else ""
        ver = f" · {info['version']}" if info.get("version") else ""
        return f"OpenCL GPU aktywne: {dn}{cu_txt}{ver}"
    if info.get("available"):
        return f"OpenCL wykryte ({info.get('device') or 'GPU'}), ale wyłączone (tryb Tylko CPU)"
    return "OpenCL niedostępne w sterowniku systemu — filtry działają na procesorze (CPU)"


def _use_umat(arr: np.ndarray) -> bool:
    return bool(_OCL_STATE.get("enabled")) and (arr.shape[0] * arr.shape[1] >= UMAT_MIN_PIXELS)


def ocl_gaussian_blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur accelerated via OpenCV UMat (OpenCL GPU) with automatic CPU fallback."""
    if _use_umat(arr):
        try:
            res = cv2.GaussianBlur(cv2.UMat(arr), (0, 0), float(sigma)).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.GaussianBlur(arr, (0, 0), float(sigma))


def ocl_box_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    """Normalized box filter (core building block of Guided Filter & D&B) via OpenCL UMat."""
    k = (2 * int(radius) + 1, 2 * int(radius) + 1)
    if _use_umat(arr):
        try:
            res = cv2.boxFilter(cv2.UMat(arr), -1, k).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.boxFilter(arr, -1, k)


def ocl_bilateral_filter(arr: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Edge-preserving bilateral filter accelerated on GPU via OpenCL UMat (Denoise & Soft Focus)."""
    if _use_umat(arr):
        try:
            res = cv2.bilateralFilter(cv2.UMat(arr), int(d), float(sigma_color), float(sigma_space)).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.bilateralFilter(arr, int(d), float(sigma_color), float(sigma_space))


def ocl_resize(arr: np.ndarray, wh: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Image resize accelerated via OpenCV UMat (OpenCL GPU) on frames >= UMAT_MIN_PIXELS."""
    if bool(_OCL_STATE.get("enabled")) and max(arr.shape[0] * arr.shape[1], wh[0] * wh[1]) >= UMAT_MIN_PIXELS:
        try:
            res = cv2.resize(cv2.UMat(arr), wh, interpolation=interpolation).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.resize(arr, wh, interpolation=interpolation)


def ocl_warp_affine(arr: np.ndarray, M: np.ndarray, wh: tuple[int, int],
                    flags: int = cv2.INTER_CUBIC, borderMode: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Bicubic/linear image rotation & straightening accelerated via OpenCL UMat."""
    if bool(_OCL_STATE.get("enabled")) and max(arr.shape[0] * arr.shape[1], wh[0] * wh[1]) >= UMAT_MIN_PIXELS:
        try:
            res = cv2.warpAffine(cv2.UMat(arr), M, wh, flags=flags, borderMode=borderMode).get()
            if res is not None:
                return res
        except Exception:
            pass
    return cv2.warpAffine(arr, M, wh, flags=flags, borderMode=borderMode)


def ocl_rl_deconv(y_norm: np.ndarray, amount: float, sigma: float = 1.05, iters: int = 3) -> np.ndarray:
    """Chained Richardson-Lucy deconvolution in OpenCL UMat memory (uploads once, runs all blurs on GPU, downloads once)."""
    amt = float(np.clip(amount, 0.0, 1.0))
    if _use_umat(y_norm):
        try:
            y_u = cv2.UMat(np.ascontiguousarray(y_norm, np.float32))
            u_u = cv2.UMat(np.ascontiguousarray(y_norm, np.float32))
            for _ in range(iters):
                conv_u = cv2.GaussianBlur(u_u, (0, 0), float(sigma))
                conv_safe = cv2.max(conv_u, 1e-3)
                ratio_u = cv2.divide(y_u, conv_safe)
                corr_u = cv2.GaussianBlur(ratio_u, (0, 0), float(sigma))
                # step = 1.0 + 1.15 * amt * (corr - 1.0) = (1 - 1.15*amt) + 1.15*amt * corr
                step_u = cv2.addWeighted(corr_u, 1.15 * amt, corr_u, 0.0, 1.0 - 1.15 * amt)
                u_u = cv2.min(cv2.max(cv2.multiply(u_u, step_u), 1e-4), 1.35)
            out = u_u.get()
            if out is not None:
                return out
        except Exception:
            pass
    u = y_norm.copy()
    for _ in range(iters):
        conv = cv2.GaussianBlur(u, (0, 0), float(sigma))
        ratio = y_norm / np.maximum(conv, 1e-3)
        corr = cv2.GaussianBlur(ratio, (0, 0), float(sigma))
        u = np.clip(u * (1.0 + 1.15 * amt * (corr - 1.0)), 1e-4, 1.35)
    return u


def ocl_benchmark(h: int = 1080, w: int = 1440) -> dict:
    """Run a live image-processing benchmark (chained Gaussian + Bilateral + WarpAffine) comparing CPU vs OpenCL UMat."""
    info = ocl_configure(MODE if MODE != "cpu" else "ocl")
    rng = np.random.default_rng(7)
    arr = rng.uniform(0, 255, size=(h, w)).astype(np.float32)

    # CPU timing
    cv2.ocl.setUseOpenCL(False)
    _ = cv2.GaussianBlur(arr, (0, 0), 2.0)
    t0 = time.perf_counter()
    c1 = cv2.GaussianBlur(arr, (0, 0), 2.2)
    c2 = cv2.bilateralFilter(c1, 7, 20.0, 5.0)
    _ = cv2.GaussianBlur(c2, (0, 0), 1.4)
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    ocl_ms = None
    speedup = None
    if info.get("available"):
        try:
            cv2.ocl.setUseOpenCL(True)
            # Warmup OpenCL kernel compilation on Intel/AMD/NVIDIA driver
            u0 = cv2.UMat(arr)
            _ = cv2.bilateralFilter(cv2.GaussianBlur(u0, (0, 0), 2.2), 7, 20.0, 5.0).get()
            t1 = time.perf_counter()
            ug = cv2.UMat(arr)
            g1 = cv2.GaussianBlur(ug, (0, 0), 2.2)
            g2 = cv2.bilateralFilter(g1, 7, 20.0, 5.0)
            _ = cv2.GaussianBlur(g2, (0, 0), 1.4).get()
            ocl_ms = (time.perf_counter() - t1) * 1000.0
            speedup = round(cpu_ms / max(ocl_ms, 0.05), 2)
        except Exception:
            ocl_ms = None

    ocl_configure(MODE)
    if ocl_ms is not None:
        dn = info.get("device") or "GPU"
        status_pl = f"OpenCL GPU ({dn}): {ocl_ms:.1f} ms vs CPU: {cpu_ms:.1f} ms (przyspieszenie {speedup:.2f}×)".replace(".", ",")
    else:
        status_pl = f"Brak sterownika OpenCL GPU w tym środowisku — czas CPU ({w}×{h}): {cpu_ms:.1f} ms".replace(".", ",")
    return {
        "available": bool(info.get("available")),
        "device": info.get("device", ""),
        "cpu_ms": round(cpu_ms, 2),
        "ocl_ms": round(ocl_ms, 2) if ocl_ms is not None else None,
        "speedup": speedup,
        "status_pl": status_pl,
    }


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
    ocl_configure(MODE)
    sess = _make(path, sample_feed)
    ACTIVE[Path(path).name] = "GPU" if "DmlExecutionProvider" in sess.get_providers() else "CPU"
    return sess


def summary() -> str:
    """Short Polish description of where AI models and OpenCV filters run (status bar / settings)."""
    ocl = ocl_info()
    ocl_tag = ""
    if ocl.get("enabled"):
        dname = ocl.get("device") or "iGPU"
        short_d = dname.replace("(R)", "").replace("(TM)", "").replace("Graphics", "").strip()
        if len(short_d) > 22:
            short_d = short_d[:20].rstrip() + "…"
        ocl_tag = f" + OpenCL ({short_d})" if short_d else " + OpenCL GPU"
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
