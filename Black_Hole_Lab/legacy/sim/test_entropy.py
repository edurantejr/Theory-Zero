def test_entropy_normalisation():
    import numpy as np
    from sim.entropy_field import node_entropy
    wij = np.eye(4) * 0.25 + 0.25             # fake uniform weights
    S    = node_entropy(wij)
    assert np.allclose(S, 0.0, atol=1e-6)     # uniform → zero gradient
