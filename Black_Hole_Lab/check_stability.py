import numpy as np
from sim.physics import evolve_metric

dx = 1.0
dt = 0.1
S  = np.random.rand(10,10,10).astype(np.float32)
g  = np.zeros((8,8,8), np.float32)

g1 = evolve_metric(g, S, dt, dx, kappa=-100.0)

print("max |g00| after update:", np.abs(g1).max())
