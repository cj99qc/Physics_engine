"""
Example 05: Custom force fields — vortex + pairwise repulsion.

Demonstrates plugging arbitrary callables into the force system.
Particles are pulled into a vortex while soft repulsion prevents
them from clumping.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from physengine import (
    ParticleSystem, Simulation,
    LinearDrag, CustomForceField,
    get_integrator, RealtimeAnimator,
)

N = 60
rng = np.random.default_rng(7)

ps = ParticleSystem(dim=2)
for _ in range(N):
    ps.add(
        mass=1.0,
        position=rng.uniform(-4.0, 4.0, 2),
        velocity=rng.uniform(-0.3, 0.3, 2),
    )
state = ps.build()


def vortex(pos, vel, masses, t):
    """Tangential force proportional to distance from origin."""
    # Rotate position vector 90 degrees to get tangential direction
    perp = np.stack([-pos[:, 1], pos[:, 0]], axis=1)  # (N, 2)
    r    = np.linalg.norm(pos, axis=1, keepdims=True) + 1e-6
    return 1.5 * perp / r


def soft_repulsion(pos, vel, masses, t):
    """Vectorized pairwise soft repulsion (inverse-square)."""
    # dr[i,j] = pos[i] - pos[j]
    dr = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]   # (N, N, 2)
    r2 = np.einsum('ijk,ijk->ij', dr, dr) + 0.25          # (N, N), softened
    np.fill_diagonal(r2, np.inf)                           # no self-force
    # F_i = sum_j  strength * dr_ij / r_ij^2
    force_contributions = 0.08 * dr / r2[:, :, np.newaxis]
    return force_contributions.sum(axis=1)                 # (N, 2)


sim = (
    Simulation(state=state, integrator=get_integrator("rk4"), dt=0.01)
    .add_force(CustomForceField(vortex, "vortex"))
    .add_force(CustomForceField(soft_repulsion, "soft_repulsion"))
    .add_force(LinearDrag(b=0.05))
)

anim = RealtimeAnimator(
    simulation=sim,
    steps_per_frame=2,
    interval=16,
    xlim=(-7.0, 7.0),
    ylim=(-7.0, 7.0),
    trail_length=80,
    title="Vortex + Soft Repulsion (RK4)",
)
anim.run()
