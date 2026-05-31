"""
Example 04: Cloth simulation using MassSpringNetwork.

A 10x10 grid of particles connected by structural and shear springs.
The top row is pinned via a stiff custom force. Gravity + drag pulls
the cloth down into a draping shape.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, LinearDrag, CustomForceField,
    get_integrator, RealtimeAnimator,
)
from physengine.forces.spring import MassSpringNetwork

NX, NY  = 10, 10
SPACING = 0.5

# Build a grid — particle index = i * NY + j
ps = ParticleSystem(dim=2)
ps.add_grid(NX, NY, SPACING, mass=0.05, origin=(0.0, 0.0))
state = ps.build()

# Remember initial positions of the top row so we can pin them
top_row_indices = [i * NY + (NY - 1) for i in range(NX)]
initial_positions = state.positions.copy()

def pin_force(pos, vel, masses, t):
    """High-stiffness spring pulling pinned particles to their initial positions."""
    f = np.zeros_like(pos)
    k_pin = 5000.0
    for idx in top_row_indices:
        f[idx] = k_pin * (initial_positions[idx] - pos[idx])
    return f

# Spring network: structural k=300, shear k=150
net = MassSpringNetwork.from_grid(NX, NY, k_structural=300.0, k_shear=150.0, spacing=SPACING)

recorder   = DataRecorder(record_every=3)
integrator = get_integrator("verlet")

sim = (
    Simulation(state=state, integrator=integrator, dt=0.002, substeps=3, recorder=recorder)
    .add_force(UniformGravity(g=9.81))
    .add_force(LinearDrag(b=1.0))
    .add_force(net)
    .add_force(CustomForceField(pin_force, name="pin_top_row"))
)

margin = 0.5
anim = RealtimeAnimator(
    simulation=sim,
    steps_per_frame=3,
    interval=20,
    xlim=(-margin, NX * SPACING + margin),
    ylim=(-NY * SPACING - margin, SPACING + margin),
    particle_size=8.0,
    trail_length=0,
    title="Cloth Simulation (Verlet + sub-steps)",
)
anim.run()
