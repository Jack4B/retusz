"""Where ONNX models run: 'auto' (benchmark GPU vs CPU once per model, remember the faster),
'cpu' or 'gpu' (DirectML on Windows). Integrated GPUs are often slower than the CPU for these models."""
from __future__ import annotations
import json, threading, time
from pathlib import Path
import numpy as np
from ..paths import data_dir

MODE = "auto"          # set from Settings
_lock = threading.Lock()
_CHOICES = data_dir() / "device.json"
ACTIVE = {}            # model file name -> "CPU" / "GPU" actually used in this run

def _load():
    try: return json.loads(_CHOICES.read_text(encoding="utf-8"))
    except (OSError, ValueError): return {}

def _save(d):
    try: _CHOICES.write_text(json.dumps(d, indent=1), encoding="utf-8")
    except OSError: pass

def _new(path, providers):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=providers)

def _bench(sess, feed, n=2):
    sess.run(None, feed); t = time.perf_counter()
    for _ in range(n): sess.run(None, feed)
    return (time.perf_counter()-t)/n

def make_session(path, sample_feed):
    """sample_feed(sess) -> input dict used for the benchmark."""
    sess = _make(path, sample_feed)
    ACTIVE[Path(path).name] = "GPU" if "DmlExecutionProvider" in sess.get_providers() else "CPU"
    return sess

def summary():
    """Short Polish description of where models run (status bar / settings)."""
    if not ACTIVE: return "modele jeszcze nie wczytane"
    vals = set(ACTIVE.values())
    return "procesor (CPU)" if vals == {"CPU"} else "karta graficzna (DirectML)" if vals == {"GPU"} else "CPU + GPU"

def _make(path, sample_feed):
    import onnxruntime as ort
    has_dml = "DmlExecutionProvider" in ort.get_available_providers()
    cpu = ["CPUExecutionProvider"]; dml = ["DmlExecutionProvider", "CPUExecutionProvider"]
    if MODE == "cpu" or not has_dml: return _new(path, cpu)
    if MODE == "gpu":
        try: return _new(path, dml)
        except Exception: return _new(path, cpu)
    key = f"{Path(path).name}:{Path(path).stat().st_size}"
    with _lock:
        choices = _load(); pick = choices.get(key)
        if pick in ("cpu", "gpu"):
            try: return _new(path, dml if pick == "gpu" else cpu)
            except Exception: return _new(path, cpu)
        s_cpu = _new(path, cpu); t_cpu = _bench(s_cpu, sample_feed(s_cpu))
        try:
            s_gpu = _new(path, dml); t_gpu = _bench(s_gpu, sample_feed(s_gpu))
        except Exception:
            s_gpu, t_gpu = None, float("inf")
        pick = "gpu" if t_gpu < 0.8*t_cpu else "cpu"
        choices[key] = pick; choices[key + ":ms"] = [round(t_cpu*1000), round(t_gpu*1000) if t_gpu < 1e9 else None]
        _save(choices)
        return s_gpu if pick == "gpu" else s_cpu

def reset():
    ACTIVE.clear()
    try: _CHOICES.unlink()
    except FileNotFoundError: pass

def zeros_feed(sess, shape_default=(1, 3, 518, 518)):
    i = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else shape_default[k] for k, d in enumerate(i.shape)]
    return {i.name: np.zeros(shape, np.float32)}
