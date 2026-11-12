"""Raw CUDA kernels compiled at run time, without a CUDA toolkit or compiler on the machine: the
NVRTC library that ships with torch compiles CUDA C to a cubin for this GPU, the NVIDIA driver
(nvcuda) loads and launches it on torch's current stream with torch tensors' memory. Cubins are
cached on disk (keyed by source, GPU architecture and NVRTC version), so a worker loads them in
milliseconds; only the first run after a change compiles (~1 s per module)."""

import ctypes
import hashlib
import os
from pathlib import Path

import torch

CACHE = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "GPUCoadd" / "cubin_cache"


def _lib(names):
    torch_lib = Path(torch.__file__).parent / "lib"
    for name in names:
        for candidate in (torch_lib / name, Path(name)):
            try:
                return ctypes.CDLL(str(candidate))
            except OSError:
                continue
    raise OSError(f"none of {names} could be loaded")


_nvrtc = None
_cuda = None


def _init():
    global _nvrtc, _cuda
    if _cuda is None:
        _nvrtc = _lib(
            [
                p.name
                for p in sorted((Path(torch.__file__).parent / "lib").glob("nvrtc64_*.dll"))
                if ".alt" not in p.name
            ]
            + ["libnvrtc.so"]
        )
        _cuda = _lib(["nvcuda.dll", "libcuda.so.1"])
        torch.zeros(1, device="cuda")  # the runtime makes the primary context current on this thread


def _check_cu(code, what):
    if code:
        name = ctypes.c_char_p()
        _cuda.cuGetErrorName(code, ctypes.byref(name))
        raise RuntimeError(f"{what} failed: {name.value.decode() if name.value else code}")


def _compile(source, arch, fmad=False):
    prog = ctypes.c_void_p()
    if _nvrtc.nvrtcCreateProgram(ctypes.byref(prog), source.encode(), b"kernel.cu", 0, None, None):
        raise RuntimeError("nvrtcCreateProgram failed")
    # fmad=False: every operation rounds as torch's one-operation-per-step kernels do; True as
    # nvcc builds torch's own multi-operation kernels (e.g. grid_sample)
    opts = [f"--gpu-architecture=sm_{arch}".encode(), b"--fmad=true" if fmad else b"--fmad=false", b"-default-device"]
    res = _nvrtc.nvrtcCompileProgram(prog, len(opts), (ctypes.c_char_p * len(opts))(*opts))
    size = ctypes.c_size_t()
    _nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(size))
    log = ctypes.create_string_buffer(size.value)
    _nvrtc.nvrtcGetProgramLog(prog, log)
    if res:
        raise RuntimeError("CUDA kernel compilation failed:\n" + log.value.decode(errors="replace"))
    _nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(size))
    cubin = ctypes.create_string_buffer(size.value)
    _nvrtc.nvrtcGetCUBIN(prog, cubin)
    _nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
    return cubin.raw


class Double:
    """A kernel argument passed as double (plain Python floats go as float32)."""

    def __init__(self, value):
        self.value = float(value)


# A cache file: this header, the cubin's SHA-256, the cubin. The driver reads a cubin without its
# length, so a damaged file is caught here.
MAGIC = b"GPUCoadd cubin 1\n"


def _read_cache(path):
    """The cubin in a cache file, or None when the file is missing, empty, of the old headerless
    format, or damaged (its bytes no longer match their SHA-256)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    head = len(MAGIC) + 32
    if len(raw) <= head or not raw.startswith(MAGIC):
        return None
    data = raw[head:]
    return data if hashlib.sha256(data).digest() == raw[len(MAGIC) : head] else None


def _write_cache(path, data):
    """Store a cubin; a cache that can't be written only means compiling again next time."""
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(MAGIC + hashlib.sha256(data).digest() + data)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class Module:
    """Compile (or load from the cache) CUDA C `source` and launch its extern "C" kernels.
    A cached cubin that is damaged or refused by the driver is compiled again and loaded from
    memory; the cache file is replaced when possible."""

    def __init__(self, source, fmad=False):
        _init()
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        ver = (ctypes.c_int(), ctypes.c_int())
        _nvrtc.nvrtcVersion(ctypes.byref(ver[0]), ctypes.byref(ver[1]))
        key = hashlib.sha256(f"{source}|{arch}|{ver[0].value}.{ver[1].value}|fmad={fmad}".encode()).hexdigest()[:32]
        path = CACHE / f"{key}.cubin"
        data = _read_cache(path)
        cached = data is not None
        if not cached:
            data = _compile(source, arch, fmad)
            _write_cache(path, data)
        self._fns = {}
        try:
            self._load(data)
        except RuntimeError:
            if not cached:
                raise
            data = _compile(source, arch, fmad)  # intact on disk, refused by this driver
            self._load(data)
            _write_cache(path, data)

    def _load(self, data):
        self._data = ctypes.create_string_buffer(data, len(data))
        self._mod = ctypes.c_void_p()
        _check_cu(_cuda.cuModuleLoadData(ctypes.byref(self._mod), self._data), "cuModuleLoadData")

    def launch(self, name, grid, block, args):
        """args: torch tensors (passed as device pointers; None = null pointer), ints (int32) or
        floats (float32). grid: a block count, or (x, y) block counts for a two-dimensional
        grid (e.g. one row of blocks per light). Runs on torch's current CUDA stream."""
        gx, gy = (grid, 1) if isinstance(grid, int) else (int(grid[0]), int(grid[1]))
        fn = self._fns.get(name)
        if fn is None:
            fn = ctypes.c_void_p()
            _check_cu(_cuda.cuModuleGetFunction(ctypes.byref(fn), self._mod, name.encode()), f"kernel {name}")
            self._fns[name] = fn
        holders = []
        for a in args:
            if a is None:
                holders.append(ctypes.c_void_p(0))
            elif isinstance(a, torch.Tensor):
                holders.append(ctypes.c_void_p(a.data_ptr()))
            elif isinstance(a, float):
                holders.append(ctypes.c_float(a))
            elif isinstance(a, Double):
                holders.append(ctypes.c_double(a.value))
            else:
                holders.append(ctypes.c_int(int(a)))
        params = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(h) for h in holders])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        _check_cu(_cuda.cuLaunchKernel(fn, gx, gy, 1, block, 1, 1, 0, stream, params, None), f"launch {name}")
