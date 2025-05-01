from backend import Backend
from physics import Physics

class Integrator:
    def __init__(self, backend: Backend, physics: Physics, dt=0.01):
        self.backend = backend
        self.xp = backend.xp
        self.physics = physics
        self.dt = dt

    def step(self, positions, velocities):
        grad = self.physics.entropy_gradient(positions)
        force = grad * 10.0  # Force scaling
        velocities += force * self.dt
        positions += velocities * self.dt

        # Boundary conditions
        mask_x_low = positions[..., 0] < 0
        mask_x_high = positions[..., 0] > 1
        mask_y_low = positions[..., 1] < 0
        mask_y_high = positions[..., 1] > 1

        velocities[mask_x_low | mask_x_high, 0] *= -1
        velocities[mask_y_low | mask_y_high, 1] *= -1

        positions[..., 0] = self.xp.clip(positions[..., 0], 0, 1)
        positions[..., 1] = self.xp.clip(positions[..., 1], 0, 1)

        return positions, velocities
