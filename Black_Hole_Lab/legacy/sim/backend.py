import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class Backend:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np

    def array(self, data):
        return self.xp.array(data)

    def zeros(self, shape, dtype=float):
        return self.xp.zeros(shape, dtype=dtype)

    def exp(self, x):
        return self.xp.exp(x)

    def meshgrid(self, x, y):
        return self.xp.meshgrid(x, y)

    def sqrt(self, x):
        return self.xp.sqrt(x)

    def sum(self, x, axis=None):
        return self.xp.sum(x, axis=axis)

    def to_numpy(self, x):
        if self.use_gpu:
            return cp.asnumpy(x)
        return x

    def norm(self, x, axis=None):
        return self.xp.linalg.norm(x, axis=axis)
