import numpy as np

from theory_zero.metrics import energy_harmonic


def test_energy_harmonic():
    state = np.array([2.0, 3.0])
    energy = energy_harmonic(state, omega=2.0)
    assert np.isclose(energy, 0.5 * (3.0**2 + (2.0 * 2.0) ** 2))
