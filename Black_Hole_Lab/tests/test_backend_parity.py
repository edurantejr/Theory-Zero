import numpy as np
import pytest

from theory_zero.backend import get_backend


def test_backend_parity_cpu_gpu():
    cpu = get_backend("numpy")
    pytest.importorskip("cupy")
    gpu = get_backend("cupy")

    data = [1.0, 2.0, 3.0]
    cpu_arr = cpu.asarray(data, dtype=np.float64)
    gpu_arr = gpu.asarray(data, dtype=np.float64)
    assert np.allclose(np.asarray(cpu_arr), np.asarray(gpu_arr))
