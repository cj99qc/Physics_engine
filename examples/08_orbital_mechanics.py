"""
Example 08: Orbital Mechanics and Hohmann Transfer

This example demonstrates using the new orbital dynamics utilities.
We set up a spacecraft in a Low Earth Orbit (LEO) using Keplerian elements,
then sequence two impulsive maneuvers (burns) to transfer it to a 
Geostationary Orbit (GEO) using a Hohmann transfer.

Units are normalized so G*M_earth = 1.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    NBodyGravity, get_integrator,
    keplerian_to_cartesian, hohmann_transfer,
    orbital_period, vis_viva_velocity,
)

# ── Normalized Earth System ────────────────────────────────────────────────
G = 1.0
M_earth = 1.0
mu = G * M_earth

# Radii
r_LEO = 1.0      # Normalised LEO radius
r_GEO = 4.0      # Selected larger radius for GEO

# ── 1. Calculate the Hohmann Transfer Burns ───────────────────────────────
dv1, dv2, tof = hohmann_transfer(r_LEO, r_GEO, mu)

print("--- Hohmann Transfer Plan ------------")
print(f"Transfer from r={r_LEO} to r={r_GEO}")
print(f"Burn 1 (LEO departure): Delta v = {dv1:.4f}")
print(f"Burn 2 (GEO arrival)  : Delta v = {dv2:.4f}")
print(f"Time of Flight        : {tof:.4f}")

# ── 2. Initial State from Keplerian Elements ──────────────────────────────
# We initialize the spacecraft in LEO (circular, e=0)
pos_ijk, vel_ijk = keplerian_to_cartesian(
    a=r_LEO, 
    e=0.0, 
    i=0.0, 
    omega=0.0, 
    Omega=0.0, 
    nu=0.0, 
    mu=mu
)

ps = ParticleSystem(dim=2)
ps.add(mass=M_earth, position=[0.0, 0.0], velocity=[0.0, 0.0], tag="Earth")
# We take only x, y components as the orbit is planar (i=0)
ps.add(mass=1e-6, position=pos_ijk[:2], velocity=vel_ijk[:2], tag="Spacecraft")
state = ps.build()

# ── 3. Simulation & Maneuver Scheduling ──────────────────────────────────
# Set up recording & integrator
recorder = DataRecorder(record_every=2, record_energy=False)
integrator = get_integrator("rk4")

sim = (
    Simulation(state=state, integrator=integrator, dt=0.01, recorder=recorder)
    .add_force(NBodyGravity(G=G, epsilon=1e-8))
)

# To avoid guessing the spacecraft's heading at t=1.0, let's just do Burn 1 at t=0
# and Burn 2 at t=tof.
sim = (
    Simulation(state=state, integrator=integrator, dt=0.01, recorder=recorder)
    .add_force(NBodyGravity(G=G, epsilon=1e-8))
)

sim.schedule_event(time=0.0, callback=lambda s: s.state.velocities.__setitem__(1, s.state.velocities[1] + [0.0, dv1]))
sim.schedule_event(time=tof, callback=lambda s: s.state.velocities.__setitem__(1, s.state.velocities[1] + [0.0, -dv2])) # Arrival heading is down

# --- 4. Run Simulation ----------------------------------------------------
T_GEO = orbital_period(r_GEO, mu)
total_time = tof + T_GEO

steps = int(total_time / sim.dt)
print(f"Running simulation for {steps} steps...")
sim.run(n_steps=steps)
print("Done.")

# ── 5. Plots ─────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

positions = recorder.positions

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor('#0d1117')
fig.patch.set_facecolor('#0d1117')

# Plot Earth
ax.scatter([0], [0], color='#58a6ff', s=300, label='Earth', zorder=5)

# Plot Spacecraft trajectory
traj = positions[:, 1, :]
ax.plot(traj[:, 0], traj[:, 1], color='#f6c90e', linewidth=1.5, label='Spacecraft Trajectory')

# Mark the burns
ax.scatter(traj[0, 0], traj[0, 1], color='#ff6b6b', s=100, label='Burn 1 (LEO)', zorder=6)
# Find the position near Time Of Flight
idx_tof = int(tof / sim.dt / recorder.record_every)
if idx_tof < len(traj):
    ax.scatter(traj[idx_tof, 0], traj[idx_tof, 1], color='#7ee787', s=100, label='Burn 2 (GEO)', zorder=6)

ax.set_aspect('equal')
ax.legend(facecolor='#161b22', labelcolor='white', edgecolor='#30363d')
ax.set_title("Hohmann Transfer to GEO", color='white')

ax.tick_params(colors='#8b949e')
for sp in ax.spines.values():
    sp.set_edgecolor('#30363d')

plt.tight_layout()
out_path = os.path.join("output", "08_orbital_mechanics.png")
plt.savefig(out_path, dpi=150, facecolor='#0d1117')
print(f"Plot saved -> {out_path}")
