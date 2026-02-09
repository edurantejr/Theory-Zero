# --- Phase 5.0: Cosmic Curvature & Multi-Singularity Warping (Theory Zero) ---

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import numpy as np

# -----------------------------
# MULTI-SINGULARITY SETUP
# -----------------------------
SOURCES = [
    {'center': np.array([0.3, 0.5]), 'sigma': 0.05, 'strength': 1.0},
    {'center': np.array([0.7, 0.5]), 'sigma': 0.08, 'strength': 0.8},
]

# Entropy Field & Gradient (Multi-Well)
def entropy_field_multi(x, y):
    return sum(s['strength'] * np.exp(-((x - s['center'][0])**2 + (y - s['center'][1])**2) / (2 * s['sigma']**2)) for s in SOURCES)

def entropy_gradient_multi(x, y):
    grad = np.zeros(2)
    for s in SOURCES:
        dx = x - s['center'][0]
        dy = y - s['center'][1]
        r2 = dx**2 + dy**2
        factor = -s['strength'] * np.exp(-r2 / (2 * s['sigma']**2)) / s['sigma']**2
        grad += factor * np.array([dx, dy])
    return grad

# Entropy Field (Single-Singularity from Phase 4.5)
SINGLE_CENTER = np.array([0.5, 0.5])
SIGMA_SINGLE = 0.06

def entropy_field_single(x, y):
    dx, dy = x - SINGLE_CENTER[0], y - SINGLE_CENTER[1]
    r2 = dx**2 + dy**2
    return np.exp(-r2 / (2 * SIGMA_SINGLE**2))

def entropy_gradient_single(x, y):
    dx, dy = x - SINGLE_CENTER[0], y - SINGLE_CENTER[1]
    r2 = dx**2 + dy**2
    factor = -np.exp(-r2 / (2 * SIGMA_SINGLE**2)) / SIGMA_SINGLE**2
    return np.array([factor * dx, factor * dy])

# -----------------------------
# PHOTON CLASS
# -----------------------------
class Photon:
    def __init__(self, pos, vel):
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.path = [self.pos.copy()]

    def step(self, field_grad_func, dt=0.01):
        grad = field_grad_func(*self.pos)
        force = grad * 50.0
        self.vel += force * dt
        self.pos += self.vel * (dt * 2.0)  # Helps show faster progression
        self.path.append(self.pos.copy())

# Create photons for each simulation
NUM_PHOTONS = 32
RADIUS_RING = 0.6
photons_single = []
photons_multi = []

for i in range(NUM_PHOTONS):
    theta = 2 * np.pi * i / NUM_PHOTONS
    x = SINGLE_CENTER[0] + RADIUS_RING * np.cos(theta)
    y = SINGLE_CENTER[1] + RADIUS_RING * np.sin(theta)
    dir_vec = np.array([-np.sin(theta), np.cos(theta)]) * 0.3
    photons_single.append(Photon([x, y], dir_vec))
    photons_multi.append(Photon([x, y], dir_vec))

# -----------------------------
# ANIMATION SETUP
# -----------------------------
EXPORT = True
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
for ax in (ax1, ax2):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

# Plot heatmaps
res = 300
X, Y = np.meshgrid(np.linspace(0, 1, res), np.linspace(0, 1, res))
Z1 = entropy_field_single(X, Y)
Z2 = entropy_field_multi(X, Y)
ax1.imshow(Z1, extent=[0,1,0,1], origin='lower', cmap='plasma', alpha=0.35)
ax2.imshow(Z2, extent=[0,1,0,1], origin='lower', cmap='plasma', alpha=0.35)
ax1.set_title("Phase 4.5: Single Singularity")
ax2.set_title("Phase 5.0: Multi-Singularity Warp")

# Draw grid overlays for spatial distortion visualization
grid_spacing = 0.1
grid_lines = []
for gx in np.arange(0, 1.01, grid_spacing):
    gy = np.linspace(0, 1, 100)
    gx_distorted = gx + 0.015 * np.sin(8 * (gy - 0.5)) * entropy_field_multi(gx, gy)
    grid_lines.append(ax2.plot(gx_distorted, gy, color='white', lw=0.3, alpha=0.5)[0])
for gy in np.arange(0, 1.01, grid_spacing):
    gx = np.linspace(0, 1, 100)
    gy_distorted = gy + 0.015 * np.sin(8 * (gx - 0.5)) * entropy_field_multi(gx, gy)
    grid_lines.append(ax2.plot(gx, gy_distorted, color='white', lw=0.3, alpha=0.5)[0])

lines1 = [ax1.plot([], [], lw=1, color='yellow')[0] for _ in photons_single]
lines2 = [ax2.plot([], [], lw=1, color='lime')[0] for _ in photons_multi]

# New: Add optional vector field arrows (for clarity, especially in Phase 1 validation)
def plot_vector_field(ax, grad_func, density=10):
    x = np.linspace(0, 1, density)
    y = np.linspace(0, 1, density)
    X, Y = np.meshgrid(x, y)
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    for i in range(density):
        for j in range(density):
            grad = grad_func(X[i, j], Y[i, j])
            U[i, j], V[i, j] = grad[0], grad[1]
    ax.quiver(X, Y, U, V, color='white', alpha=0.25, scale=15)

plot_vector_field(ax1, entropy_gradient_single)
plot_vector_field(ax2, entropy_gradient_multi)

def update(frame):
    for p, line in zip(photons_single, lines1):
        p.step(entropy_gradient_single)
        path = np.array(p.path)
        line.set_data(path[:, 0], path[:, 1])

    for p, line in zip(photons_multi, lines2):
        p.step(entropy_gradient_multi)
        path = np.array(p.path)
        line.set_data(path[:, 0], path[:, 1])

    return lines1 + lines2 + grid_lines

ani = animation.FuncAnimation(fig, update, frames=250, interval=50, blit=True)

if EXPORT:
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Theory Zero'), bitrate=2000)
    ani.save("phase5_cosmic_curvature.mp4", writer=writer, dpi=200)
    print("✅ Exported: phase5_cosmic_curvature.mp4")
else:
    plt.tight_layout()
    plt.show()
