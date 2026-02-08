import argparse
import time
import numpy as np
from backend import Backend
from physics import Physics
from integrators import Integrator

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Simulation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--nodes", type=int, default=1000, help="Number of particles")
    parser.add_argument("--frames", type=int, default=50, help="Number of frames")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--dx", type=float, default=0.01, help="Spatial resolution")
    args = parser.parse_args()

    backend = Backend(use_gpu=args.gpu)
    xp = backend.xp

    # Initialize particles
    positions = xp.random.uniform(0, 1, (args.nodes, 2))
    velocities = xp.random.uniform(-0.1, 0.1, (args.nodes, 2))

    physics = Physics(backend, sigma=0.1, kappa=-0.01)
    integrator = Integrator(backend, physics, dt=args.dt)

    # Run simulation
    trajectories = [backend.to_numpy(positions)]
    start_time = time.time()

    for frame in range(args.frames):
        positions, velocities = integrator.step(positions, velocities)
        trajectories.append(backend.to_numpy(positions))

    end_time = time.time()
    duration = end_time - start_time
    nodes_per_sec = (args.nodes * args.frames) / duration

    # Save trajectories
    trajectories = np.stack(trajectories, axis=0)
    np.savez("phase3_traj.npz", trajectories=trajectories)

    # Performance report
    print(f"Simulation completed in {duration:.2f} seconds")
    print(f"Nodes per second: {nodes_per_sec:.2f}")
    print(f"Memory usage: {trajectories.nbytes / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
