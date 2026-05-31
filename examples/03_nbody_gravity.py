"""
Example 03: N-body gravitational simulation.

Multiple bodies orbit a central mass under mutual gravitational attraction.
Uses the Leapfrog integrator for best long-term energy conservation.
After closing the window, trajectory data is exported to .npz.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    NBodyGravity, get_integrator,
    RealtimeAnimator, NumpyExporter,
)

rng = np.random.default_rng(42)
N   = 8

ps = ParticleSystem(dim=2)

# Orbiting bodies
for i in range(N):
    angle = 2 * np.pi * i / N
    r     = rng.uniform(3.0, 7.0)
    pos   = [r * np.cos(angle), r * np.sin(angle)]
    # Approximate circular orbit speed for G=1, central mass = N*10
    v_mag = np.sqrt(N * 10 / r) * 0.85
    vel   = [-v_mag * np.sin(angle), v_mag * np.cos(angle)]
    ps.add(mass=1.0, position=pos, velocity=vel, tag=f"body_{i}")

# Heavy central body (fixed approximately by large mass)
ps.add(mass=float(N) * 10, position=[0.0, 0.0], velocity=[0.0, 0.0], tag="center")

state      = ps.build()
recorder   = DataRecorder(record_every=10, record_energy=True)
integrator = get_integrator("leapfrog")

sim = (
    Simulation(state=state, integrator=integrator, dt=0.005, recorder=recorder)
    .add_force(NBodyGravity(G=1.0, epsilon=0.1))
)

anim = RealtimeAnimator(
    simulation=sim,
    steps_per_frame=8,
    interval=16,
    xlim=(-12.0, 12.0),
    ylim=(-12.0, 12.0),
    trail_length=400,
    title=f"N-Body Gravity (N={N+1}, Leapfrog)",
)
anim.run()

# Export data after window closes
os.makedirs("output", exist_ok=True)
NumpyExporter("output/03_nbody.npz").export(recorder)

e = recorder.total_energy
print(f"Steps recorded : {len(recorder)}")
print(f"Energy at t=0  : {e[0]:.6f}")
print(f"Energy at end  : {e[-1]:.6f}")
print(f"Max drift      : {abs(e - e[0]).max():.2e}")
