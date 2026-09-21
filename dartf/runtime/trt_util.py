"""Small TensorRT 10 Python helpers shared by the W8A8 tooling on the Orin."""
import numpy as np, tensorrt as trt, ctypes, os, json, time
from cuda_alloc import DevBuf, memcpy_d2d  # thin cudart wrapper (see cuda_alloc.py)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
NP_DTYPE = {trt.float32: np.float32, trt.float16: np.float16, trt.int8: np.int8, trt.int32: np.int32, trt.bool: np.bool_}

# --- LQ plugin library (fc1 GELU / RoPE GEMM plugins): register before any engine is deserialized ---------------------
import os as _os, ctypes as _ctypes
for _lib in [p for p in (_os.environ.get("LQ_PLUGINS") or _os.path.expanduser("~/w8a8/plugins/lq_plugins2.so")).split(",") if p and _os.path.exists(p)]:
    try:
        _reg = trt.get_plugin_registry()
        if _reg.get_creator("LQFc1Gelu", "1", "") is None or _reg.get_creator("LQGemmRope", "1", "") is None:
            _h = _ctypes.CDLL(_lib, mode=_ctypes.RTLD_GLOBAL); _h.lq_register()
    except Exception as _e: print("LQ plugin load failed:", _e)

def load_engine(path, plugin=None):
    if plugin: ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    rt = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f: eng = rt.deserialize_cuda_engine(f.read())
    assert eng is not None, path
    return eng

class Runner:
    """Synchronous single-stream executor. Feeds numpy inputs, returns numpy outputs."""
    def __init__(self, engine, outputs=None):
        self.engine = engine; self.ctx = engine.create_execution_context()
        self.names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if outputs is not None: self.outputs = [n for n in self.outputs if n in outputs]
        # aliases: 'images' -> first input; fpn_0/1/2 -> the three largest 4-D outputs by width (desc)
        self.alias = {}
        if self.inputs and "images" not in self.inputs: self.alias["images"] = self.inputs[0]
        outs4 = [n for n in self.outputs if len(engine.get_tensor_shape(n)) == 4]
        outs4.sort(key=lambda n: -engine.get_tensor_shape(n)[-1])
        for i, n in enumerate(outs4[:3]):
            if f"fpn_{i}" not in self.outputs: self.alias[f"fpn_{i}"] = n
        self.bufs = {}
        for n in self.names:
            shape = tuple(engine.get_tensor_shape(n)); dt = NP_DTYPE[engine.get_tensor_dtype(n)]
            nbytes = int(np.prod(shape)) * np.dtype(dt).itemsize
            self.bufs[n] = (DevBuf(nbytes), shape, dt)
            self.ctx.set_tensor_address(n, self.bufs[n][0].ptr)
    def _res(self, n): return self.alias.get(n, n)
    def __call__(self, feeds, want=None):
        feeds = {self._res(k): v for k, v in feeds.items()}
        for n in self.inputs:
            buf, shape, dt = self.bufs[n]
            if hasattr(feeds[n], "data_ptr"):      # torch CUDA tensor already in the engine dtype/shape: device-to-device, no host round trip
                x = feeds[n]; assert x.is_cuda and x.is_contiguous() and tuple(x.shape) == shape and x.element_size() == np.dtype(dt).itemsize, (n, tuple(x.shape), shape)
                memcpy_d2d(buf.ptr, x.data_ptr(), x.numel() * x.element_size()); continue
            x = np.ascontiguousarray(feeds[n], dtype=dt); assert x.shape == shape, (n, x.shape, shape)
            buf.upload(x)
        ok = self.ctx.execute_v2([self.bufs[n][0].ptr for n in self.names]); assert ok
        out = {}
        for n in (self.outputs if want is None else want):
            rn = self._res(n); buf, shape, dt = self.bufs[rn]; out[n] = buf.download(shape, dt)
            if rn != n: out[rn] = out[n]
        return out
    def time(self, feeds, warmup=10, iters=50):
        feeds = {self._res(k): v for k, v in feeds.items()}
        for n in self.inputs:
            buf, shape, dt = self.bufs[n]; buf.upload(np.ascontiguousarray(feeds[n], dtype=dt))
        ptrs=[self.bufs[n][0].ptr for n in self.names]
        for _ in range(warmup): self.ctx.execute_v2(ptrs)
        DevBuf.sync(); t=time.perf_counter()
        for _ in range(iters): self.ctx.execute_v2(ptrs)
        DevBuf.sync(); return (time.perf_counter()-t)*1000/iters
