import numpy as np
from backend import Backend

class Physics:
    def __init__(self, backend: Backend, sigma=0.1, kappa=-0.01):
        self.backend = backend
        self.xp = backend.xp
        self.sigma = sigma
        self.kappa = kappa
        self.center = self.xp.array([0.5, 0.5])

    def entropy_field(self, positions):
        dx = positions[..., 0] - self.center[0]
        dy = positions[..., 1] - self.center[1]
        r2 = dx**2 + dy**2
        S = self.xp.exp(-r2 / (2 * self.sigma**2))
        return S

    def entropy_gradient(self, positions):
        S = self.entropy_field(positions)
        dx = positions[..., 0] - self.center[0]
        dy = positions[..., 1] - self.center[1]
        factor = -S / (self.sigma**2)
        grad = self.xp.stack([factor * dx, factor * dy], axis=-1)
        return grad

    def curvature_scalar(self, positions):
        h = 0.01
        pos_shape = positions.shape[:-1]
        pos = positions.reshape(-1, 2)
        S = self.entropy_field(pos).reshape(pos_shape)

        # Finite differences
        pos_dx = pos + self.xp.array([h, 0])
        pos_dx_neg = pos - self.xp.array([h, 0])
        pos_dy = pos + self.xp.array([0, h])
        pos_dy_neg = pos - self.xp.array([0, h])

        S_dx = self.entropy_field(pos_dx).reshape(pos_shape)
        S_dx_neg = self.entropy_field(pos_dx_neg).reshape(pos_shape)
        S_dy = self.entropy_field(pos_dy).reshape(pos_shape)
        S_dy_neg = self.entropy_field(pos_dy_neg).reshape(pos_shape)

        S_dx2 = (S_dx - 2 * S + S_dx_neg) / h**2
        S_dy2 = (S_dy - 2 * S + S_dy_neg) / h**2

        laplacian = S_dx2 + S_dy2
        return self.kappa * laplacian
