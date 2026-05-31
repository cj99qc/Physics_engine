"""
Example 09: SPH Dam Break — 2D fluid simulation.

A column of water is released inside a box and collapses under gravity.
This demonstrates the SPH (Smoothed Particle Hydrodynamics) module which
discretizes the Navier-Stokes equations in Lagrangian particle form:

    Dv/Dt = -(1/rho) grad(P) + nu * laplacian(v) + g

Components used:
    - SPHDensityComputer : computes particle densities via kernel summation
    - SPHPressureForce   : pressure gradient (Tait equation of state)
    - SPHViscosityForce  : viscous diffusion (Morris 1997)
    - BoxBoundaryForce   : repulsive wall boundaries
    - UniformGravity     : external body force

Usage:
    python examples/09_sph_dam_break.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, get_integrator,
    SPHDensityComputer, SPHPressureForce, SPHViscosityForce,
    BoxBoundaryForce, SPHParticleInitializer,
)

# ── Parameters ────────────────────────────────────────────────────────────
SPACING   = 0.08          # particle spacing (resolution)
H         = SPACING * 2   # smoothing length (2x spacing is standard)
RHO_0     = 1000.0        # rest density (kg/m^3)
K_PRESS   = 2000.0        # pressure stiffness
MU_VISC   = 5.0           # viscosity coefficient
GRAVITY   = 9.81
DT        = 0.001         # small timestep for stability
N_STEPS   = 500           # total simulation steps
RECORD_N  = 5             # record every N steps

# Domain
BOX       = ((0.0, 1.0), (0.0, 1.0))

# Initial fluid column: left quarter of the box, full height
FLUID_REGION = ((0.02, 0.3), (0.02, 0.6))

# ── Build particle system ─────────────────────────────────────────────────
positions, particle_mass = SPHParticleInitializer.fill_box(
    region=FLUID_REGION,
    spacing=SPACING,
    rho_0=RHO_0,
)
N = positions.shape[0]
print(f"SPH Dam Break: {N} particles, h={H}, dt={DT}")

ps = ParticleSystem(dim=2)
for i in range(N):
    ps.add(mass=particle_mass, position=positions[i].tolist(), velocity=[0.0, 0.0])
state = ps.build()

# ── SPH forces ────────────────────────────────────────────────────────────
density = SPHDensityComputer(h=H, rest_rho=RHO_0)

recorder   = DataRecorder(record_every=RECORD_N, record_energy=False)
integrator = get_integrator("verlet")

sim = (
    Simulation(state=state, integrator=integrator, dt=DT, recorder=recorder)
    .add_force(SPHPressureForce(density, k=K_PRESS))
    .add_force(SPHViscosityForce(density, mu=MU_VISC))
    .add_force(UniformGravity(g=GRAVITY, direction=[0.0, -1.0]))
    .add_force(BoxBoundaryForce(bounds=BOX, stiffness=20000.0, margin=SPACING))
)

# ── Run simulation ────────────────────────────────────────────────────────
print(f"Running {N_STEPS} steps...")
sim.run(n_steps=N_STEPS)
print(f"Done. Sim time = {sim.time:.4f} s")

# ── Visualization ─────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

pos_data = recorder.positions    # (T, N, 2)
times    = recorder.times        # (T,)
T_frames = pos_data.shape[0]

BG   = '#0d1117'
GREY = '#8b949e'
EDGE = '#30363d'

fig, ax = plt.subplots(figsize=(10, 10))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect('equal')
ax.tick_params(colors=GREY)
for sp in ax.spines.values():
    sp.set_edgecolor(EDGE)
ax.set_title('SPH Dam Break', color='white', fontsize=14)

# Draw box outline
box_x = [BOX[0][0], BOX[0][1], BOX[0][1], BOX[0][0], BOX[0][0]]
box_y = [BOX[1][0], BOX[1][0], BOX[1][1], BOX[1][1], BOX[1][0]]
ax.plot(box_x, box_y, color='#58a6ff', linewidth=2, zorder=1)

# Plot snapshots: initial, 1/4, 1/2, 3/4, final
snapshot_indices = [0, T_frames // 4, T_frames // 2, 3 * T_frames // 4, T_frames - 1]
alphas = [0.2, 0.35, 0.55, 0.75, 1.0]
colors = ['#58a6ff', '#388bfd', '#1f6feb', '#58a6ff', '#f6c90e']

for idx, alpha, color in zip(snapshot_indices, alphas, colors):
    p = pos_data[idx]
    ax.scatter(p[:, 0], p[:, 1], s=3, c=color, alpha=alpha,
              label=f't={times[idx]:.3f}s', zorder=2)

ax.legend(facecolor='#161b22', labelcolor='white', edgecolor=EDGE,
          loc='upper right', fontsize=9)

plt.tight_layout()
out_path = os.path.join("output", "09_sph_dam_break.png")
plt.savefig(out_path, dpi=150, facecolor=BG)
print(f"Plot saved -> {out_path}")
