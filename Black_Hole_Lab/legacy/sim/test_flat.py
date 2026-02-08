import numpy as np
from sim.physics import ricci_tensor

def test_flat_space():
    S  = np.ones((6,6,6), np.float32)
    R00, Rii = ricci_tensor(S, h=1.0)
    assert np.allclose(R00, 0, atol=1e-6)
