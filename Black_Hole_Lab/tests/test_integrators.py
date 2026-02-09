import numpy as np

from theory_zero.integrators import build_integrator
from theory_zero.models.base import HarmonicOscillator


def test_rk4_harmonic_small_error():
    model = HarmonicOscillator(omega=1.0, x0=1.0, v0=0.0)
    integrator = build_integrator({"name": "rk4"})
    dt = 0.01
    steps = 100
    state = model.initial_state()
    time = 0.0
    for _ in range(steps):
        state = integrator.step(state, time, dt, model.derivative)
        time += dt

    expected_x = np.cos(time)
    expected_v = -np.sin(time)
    assert np.allclose(state[0], expected_x, atol=1e-3)
    assert np.allclose(state[1], expected_v, atol=1e-3)
