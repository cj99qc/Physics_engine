"""
Example 07: Lagrange points in the Sun-Jupiter system.

Uses the circular restricted three-body problem (CR3BP) in normalized units:
    G = M_total = R (Sun-Jupiter distance) = 1
    mu = M_Jupiter / M_total = 0.001  (Sun-Jupiter mass ratio ~0.000953)

The 5 Lagrange points:
    L1  Between Sun and Jupiter          — UNSTABLE (saddle point)
    L2  Beyond Jupiter, same line        — UNSTABLE (saddle point)
    L3  Beyond Sun, opposite side        — UNSTABLE (saddle point)
    L4  60 degrees ahead of Jupiter      — STABLE   (real Jupiter Trojans live here)
    L5  60 degrees behind Jupiter        — STABLE   (real Jupiter Trojans live here)

Method:
  - L4, L5 computed analytically (equilateral triangle geometry)
  - L1, L2, L3 found by root-finding the effective potential gradient on the x-axis
  - Test particles placed at each Lagrange point with a small perturbation
  - Simulation run in the inertial frame using NBodyGravity
  - Results visualised in BOTH frames: inertial + rotating

No modifications to the engine are needed. The rotating frame is recovered
by post-processing: pos_rot(t) = R(-omega*t) @ pos_inertial(t).

Usage:
    python examples/07_lagrange_points.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from physengine import ParticleSystem, Simulation, DataRecorder, NBodyGravity, get_integrator

# ── System parameters ─────────────────────────────────────────────────────
G     = 1.0
M_tot = 1.0
mu    = 0.001              # mass ratio M_Jupiter / M_total
R     = 1.0                # normalised Sun-Jupiter distance
M1    = (1.0 - mu) * M_tot   # Sun
M2    = mu * M_tot            # Jupiter
omega = np.sqrt(G * M_tot / R ** 3)   # rotating frame angular velocity
T_orb = 2.0 * np.pi / omega           # orbital period

# ── Rotating-frame positions of Sun and Jupiter ───────────────────────────
#   CoM at origin; Sun at x = -mu, Jupiter at x = 1-mu
x_sun_rot = -mu
x_jup_rot =  1.0 - mu

# ── Find L1, L2, L3 by root-finding dOmega/dx = 0 on the x-axis ──────────
# Gradient of the effective potential (Jacobi integral) on y=0:
#   dOmega/dx = x - (1-mu)*(x+mu)/|x+mu|^3 - mu*(x-(1-mu))/|x-(1-mu)|^3
def dOmega_dx(x):
    r1 = abs(x - x_sun_rot)   # distance from Sun  (Sun at x = -mu)
    r2 = abs(x - x_jup_rot)   # distance from Jupiter (Jupiter at x = 1-mu)
    sign1 = np.sign(x - x_sun_rot)
    sign2 = np.sign(x - x_jup_rot)
    return x - (1.0 - mu) * sign1 / r1 ** 2 - mu * sign2 / r2 ** 2

eps = 1e-6
x_L1 = brentq(dOmega_dx, x_sun_rot + eps, x_jup_rot - eps)
x_L2 = brentq(dOmega_dx, x_jup_rot + eps, x_jup_rot + 2.0)
x_L3 = brentq(dOmega_dx, x_sun_rot - 2.0, x_sun_rot - eps)

# ── L4 and L5 (equilateral triangle with both primaries) ─────────────────
x_L4 =  0.5 - mu
y_L4 =  np.sqrt(3.0) / 2.0
x_L5 =  0.5 - mu
y_L5 = -np.sqrt(3.0) / 2.0

lagrange_rot = {          # positions in rotating frame
    'L1': np.array([x_L1, 0.0]),
    'L2': np.array([x_L2, 0.0]),
    'L3': np.array([x_L3, 0.0]),
    'L4': np.array([x_L4, y_L4]),
    'L5': np.array([x_L5, y_L5]),
}

print("Lagrange point positions (rotating frame, normalised):")
for name, p in lagrange_rot.items():
    print(f"  {name}: x={p[0]:+.6f}  y={p[1]:+.6f}")

# ── Convert rotating-frame equilibrium to inertial initial conditions ─────
# At t=0 the frames coincide, so position is unchanged.
# Velocity: a particle at rest in the rotating frame has inertial velocity
#           v = omega x r  →  v = omega * (-y, +x)
def rot_rest_vel(pos_rot):
    x, y = pos_rot
    return np.array([-omega * y, omega * x])

# ── Build particle system ─────────────────────────────────────────────────
ps = ParticleSystem(dim=2)

# Sun: at (-mu, 0), counterclockwise orbit around CoM
ps.add(mass=M1,
       position=[x_sun_rot, 0.0],
       velocity=[0.0, -omega * mu],       # v = omega*r for the Sun's orbit radius
       tag="sun")

# Jupiter: at (1-mu, 0), counterclockwise orbit around CoM
ps.add(mass=M2,
       position=[x_jup_rot, 0.0],
       velocity=[0.0, omega * (1.0 - mu)],
       tag="jupiter")

# Test particles at Lagrange points with a small perturbation.
# Stable L4/L5: 1 % radial nudge  → librate around the point
# Unstable L1-L3: 0.1% nudge      → exponential divergence becomes visible
perturb = {'L1': 0.001, 'L2': 0.001, 'L3': 0.001, 'L4': 0.01, 'L5': 0.01}
test_mass = 1e-10

for name, pos_rot in lagrange_rot.items():
    p = pos_rot * (1.0 + perturb[name])       # shift position
    v = rot_rest_vel(p)                        # co-rotating velocity at new position
    ps.add(mass=test_mass, position=p.tolist(), velocity=v.tolist(), tag=name)

state = ps.build()

# ── Simulation ────────────────────────────────────────────────────────────
N_orbits    = 20
steps_orbit = 1000
dt          = T_orb / steps_orbit

recorder = DataRecorder(record_every=2, record_energy=False)
sim = (
    Simulation(state=state, integrator=get_integrator("rk4"),
               dt=dt, recorder=recorder)
    .add_force(NBodyGravity(G=G, epsilon=1e-9))
)

total_steps = N_orbits * steps_orbit
print(f"\nRunning {N_orbits} orbital periods ({total_steps} steps, dt={dt:.5f})...")
sim.run(n_steps=total_steps)
print("Done.")

# ── Post-process: convert to rotating frame ────────────────────────────────
times = recorder.times        # (T,)
pos   = recorder.positions    # (T, N_particles, 2)

theta  = -omega * times       # (T,) rotation angle
c, s   = np.cos(theta), np.sin(theta)

pos_rot = np.empty_like(pos)
pos_rot[:, :, 0] = c[:, None] * pos[:, :, 0] - s[:, None] * pos[:, :, 1]
pos_rot[:, :, 1] = s[:, None] * pos[:, :, 0] + c[:, None] * pos[:, :, 1]

# ── Effective potential (Jacobi integral) for contour background ──────────
def U_eff(X, Y):
    r1 = np.sqrt((X - x_sun_rot) ** 2 + Y ** 2)
    r2 = np.sqrt((X - x_jup_rot) ** 2 + Y ** 2)
    r1 = np.where(r1 < 1e-4, 1e-4, r1)
    r2 = np.where(r2 < 1e-4, 1e-4, r2)
    return -(1.0 - mu) / r1 - mu / r2 - 0.5 * (X ** 2 + Y ** 2)

grid_x = np.linspace(-1.8, 1.8, 400)
grid_y = np.linspace(-1.6, 1.6, 400)
XX, YY = np.meshgrid(grid_x, grid_y)
ZZ     = U_eff(XX, YY)

# Jacobi constant values at each Lagrange point
C_L = {name: -2.0 * U_eff(*p) for name, p in lagrange_rot.items()}

# ── Plots ─────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

BG   = '#0d1117'
GREY = '#8b949e'
EDGE = '#30363d'

particle_colors = {
    'sun':     '#f6c90e',
    'jupiter': '#c8a96e',
    'L1':      '#ff6b6b',
    'L2':      '#ff9f43',
    'L3':      '#ff6b9d',
    'L4':      '#58a6ff',
    'L5':      '#7ee787',
}
tags   = ['sun', 'jupiter', 'L1', 'L2', 'L3', 'L4', 'L5']
msizes = [220,    60,        20,   20,   20,   20,   20]

fig, axes = plt.subplots(1, 2, figsize=(18, 9))
fig.patch.set_facecolor(BG)
fig.suptitle(
    f'Lagrange Points - Sun-Jupiter (mu={mu})  .  {N_orbits} orbital periods',
    color='white', fontsize=14, y=1.01
)

for ax, frame_pos, title in [
    (axes[0], pos,     'Inertial frame'),
    (axes[1], pos_rot, 'Co-rotating frame'),
]:
    ax.set_facecolor(BG)
    ax.set_aspect('equal')
    ax.set_title(title, color='white', pad=10)
    ax.tick_params(colors=GREY)
    for sp in ax.spines.values():
        sp.set_edgecolor(EDGE)
    ax.set_xlabel('x', color=GREY)
    ax.set_ylabel('y', color=GREY)

    # Draw zero-velocity contours in rotating frame
    if title.startswith('Co-rotating'):
        levels = sorted(C_L.values())
        ax.contour(XX, YY, -2.0 * ZZ, levels=levels,
                   colors=['#ff6b6b', '#ff9f43', '#ff6b9d', '#58a6ff', '#7ee787'],
                   linewidths=0.7, alpha=0.45)

        # Mark the theoretical Lagrange point positions with crosses
        for name, lp in lagrange_rot.items():
            ax.scatter(lp[0], lp[1], color=particle_colors[name], s=120,
                       marker='x', linewidths=2.0, zorder=7)
            ax.annotate(name, lp, color=particle_colors[name],
                        fontsize=9, xytext=(4, 4), textcoords='offset points')

    # Plot trajectories
    for i, (tag, ms) in enumerate(zip(tags, msizes)):
        c = particle_colors[tag]
        traj = frame_pos[:, i, :]
        lw   = 0.9 if tag not in ('sun', 'jupiter') else 0.6
        ax.plot(traj[:, 0], traj[:, 1], color=c, linewidth=lw, alpha=0.75,
                label=tag if tag not in ('sun', 'jupiter') else None)
        # Start marker
        ax.scatter(traj[0, 0], traj[0, 1], color=c, s=ms, zorder=5)

    ax.legend(facecolor='#161b22', labelcolor='white', edgecolor=EDGE,
              loc='lower right', fontsize=9)
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.6, 1.6)

plt.tight_layout()
out_path = os.path.join("output", "07_lagrange_points.png")
plt.savefig(out_path, dpi=150, facecolor=BG)
print(f"Plot saved -> {out_path}")

# ── Print stability summary ───────────────────────────────────────────────
print("\nFinal displacement from initial position (rotating frame):")
for i, tag in enumerate(tags[2:], start=2):
    r0    = pos_rot[0,  i, :]
    r_end = pos_rot[-1, i, :]
    disp  = np.linalg.norm(r_end - r0)
    print(f"  {tag:2s}: {disp:.6f}  {'(STABLE - librating)' if disp < 0.15 else '(UNSTABLE - drifted)'}")

plt.show()
