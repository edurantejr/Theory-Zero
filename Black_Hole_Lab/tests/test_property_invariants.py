import numpy as np
from hypothesis import given, strategies as st

from theory_zero.core.checks import ensure_finite
from theory_zero.metrics import energy_harmonic


@given(
    x=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    v=st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
    omega=st.floats(min_value=0.1, max_value=5, allow_nan=False, allow_infinity=False),
)
def test_energy_finite(x, v, omega):
    state = np.array([x, v], dtype=float)
    energy = energy_harmonic(state, omega=omega)
    ensure_finite(np.array([energy]), name="energy")
