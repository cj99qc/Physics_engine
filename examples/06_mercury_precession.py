"""
Example 06: Mercury perihelion precession — validating the 1PN force.

Uses normalized units (G = M_sun = 1, a = 1) with Mercury's real eccentricity
e = 0.2056. The speed of light c is set low enough to give a visually clear
and numerically measurable precession — this is standard practice for 1PN
validation tests.

Precession theory (Schwarzschild, test-particle limit):
    Δφ = 6π G M / (a (1−e²) c²)  per orbit

Measurement method:
    Track the Laplace-Runge-Lenz (LRL) vector A = v²r − (r·v)v − GM r̂.
    In a pure Keplerian orbit |A| is constant and its direction gives the
    perihelion angle. With 1PN corrections the direction slowly rotates —
    the rate is the precession.

Usage
-----
    python examples/06_mercury_precession.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    NBodyGravity, PostNewtonianForce,
    get_integrator,
)

# ── Normalized units: G = M_sun = a = 1 ──────────────────────────────────
G     = 1.0
M_sun = 1.0
e     = 0.2056   # Mercury eccentricity
a     = 1.0      # semi-major axis
c     = 100.0    # speed of light in normalized units.
                 # c=100 keeps the 1PN correction small (~0.11 deg/orbit) so
                 # the perturbative formula is accurate to <0.2%.
                 # Use c=50 for more visually dramatic precession (~0.45 deg/orbit)
                 # at the cost of ~0.5% extra error from higher-order terms.

# Theoretical precession per orbit
delta_phi_theory = 6.0 * np.pi * G * M_sun / (a * (1.0 - e ** 2) * c ** 2)
print(f"--- Mercury Precession Test -----------------------------")
print(f"G={G}, M={M_sun}, a={a}, e={e}, c={c}")
print(f"Theoretical Delta phi/orbit : {np.degrees(delta_phi_theory):.5f} deg  "
      f"({delta_phi_theory:.6f} rad)")

# ── Initial conditions: perihelion of a Keplerian ellipse ────────────────
r_peri = a * (1.0 - e)
v_peri = np.sqrt(G * M_sun * (1.0 + e) / (a * (1.0 - e)))  # vis-viva at perihelion

# Keplerian period (used to set dt and run duration)
T_orbit = 2.0 * np.pi * np.sqrt(a ** 3 / (G * M_sun))

# ── Particle system ───────────────────────────────────────────────────────
# Particle 0: Sun (very heavy, barely moves)
# Particle 1: Mercury (test particle, mass ≪ M_sun)
ps = ParticleSystem(dim=2)
ps.add(mass=M_sun, position=[0.0,    0.0],    velocity=[0.0,    0.0],    tag="sun")
ps.add(mass=1e-8,  position=[r_peri, 0.0],    velocity=[0.0,    v_peri], tag="mercury")
state = ps.build()

# ── Simulation ────────────────────────────────────────────────────────────
N_orbits    = 80   # more orbits to accumulate visible precession at c=100
steps_orbit = 3000
dt          = T_orbit / steps_orbit

recorder = DataRecorder(record_every=1, record_energy=False)

sim = (
    Simulation(state=state, integrator=get_integrator("rk4"),
               dt=dt, recorder=recorder)
    .add_force(NBodyGravity(G=G, epsilon=1e-8))
    .add_force(PostNewtonianForce(G=G, c=c, epsilon=1e-8, include_indirect=True))
)

total_steps = int(N_orbits * steps_orbit)
print(f"Running {N_orbits} orbits ({total_steps} steps, dt={dt:.5f}) …")
sim.run(n_steps=total_steps)
print(f"Done. Sim time = {sim.time:.3f}")

# ── Measure precession via Laplace-Runge-Lenz vector ─────────────────────
positions  = recorder.positions    # (T, 2, 2)
velocities = recorder.velocities   # (T, 2, 2)
times      = recorder.times        # (T,)

# Sun-centred frame
r_vec = positions[:, 1, :] - positions[:, 0, :]    # (T, 2)
v_vec = velocities[:, 1, :] - velocities[:, 0, :]  # (T, 2)
r_mag = np.linalg.norm(r_vec, axis=1)              # (T,)
r_hat = r_vec / r_mag[:, np.newaxis]               # (T, 2)

v2      = np.einsum('ti,ti->t', v_vec, v_vec)      # (T,)
r_dot_v = np.einsum('ti,ti->t', r_vec, v_vec)      # (T,)

# LRL vector: A = v²·r − (r·v)·v − GM·r̂
A = (v2[:, np.newaxis] * r_vec
     - r_dot_v[:, np.newaxis] * v_vec
     - G * M_sun * r_hat)                          # (T, 2)

peri_angle = np.unwrap(np.arctan2(A[:, 1], A[:, 0]))   # (T,) in radians

# Linear fit → precession rate [rad / time]
coeffs           = np.polyfit(times, peri_angle, 1)
precession_rate  = coeffs[0]
precession_orbit = precession_rate * T_orbit

rel_error = abs(precession_orbit - delta_phi_theory) / delta_phi_theory * 100.0

print(f"Measured  Delta phi/orbit   : {np.degrees(precession_orbit):.5f} deg  "
      f"({precession_orbit:.6f} rad)")
print(f"Relative error       : {rel_error:.3f}%")
print(f"---------------------------------------------------------")

# ── Plots ─────────────────────────────────────────────────────────────────
os.makedirs("output", exist_ok=True)

BG    = '#0d1117'
BLUE  = '#58a6ff'
GOLD  = '#f6c90e'
RED   = '#ff6b6b'
GREY  = '#8b949e'
EDGE  = '#30363d'

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(BG)

# — Left: orbit trajectory ——
ax = axes[0]
ax.set_facecolor(BG)
ax.plot(r_vec[:, 0], r_vec[:, 1], color=BLUE, linewidth=0.5, alpha=0.7)
ax.scatter([0.0], [0.0], color=GOLD, s=250, zorder=5, label='Sun')
ax.scatter(r_vec[0, 0], r_vec[0, 1], color=RED, s=60, zorder=5, label='Start (perihelion)')

# Draw the precessing perihelion direction
n_arrows = 6
arrow_times = np.linspace(0, len(times) - 1, n_arrows, dtype=int)
for idx in arrow_times:
    A_norm = A[idx] / (np.linalg.norm(A[idx]) + 1e-30)
    ax.annotate("", xy=A_norm * r_peri * 0.9, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.0, alpha=0.5))

ax.set_aspect('equal')
ax.set_title(f'Orbit — {N_orbits} revolutions', color='white', pad=10)
ax.legend(facecolor='#161b22', labelcolor='white', edgecolor=EDGE)
ax.tick_params(colors=GREY)
for sp in ax.spines.values():
    sp.set_edgecolor(EDGE)
ax.set_xlabel('x', color=GREY)
ax.set_ylabel('y', color=GREY)

# — Right: perihelion angle vs time ——
ax2 = axes[1]
ax2.set_facecolor(BG)

orbit_num = times / T_orbit
ax2.plot(orbit_num, np.degrees(peri_angle), color=BLUE, linewidth=1.0, label='LRL angle')

t_fit = np.array([times[0], times[-1]])
ax2.plot(t_fit / T_orbit,
         np.degrees(np.polyval(coeffs, t_fit)),
         '--', color=GOLD, linewidth=1.5,
         label=f'Fit: {np.degrees(precession_orbit):.4f} deg/orbit')
ax2.axhline(0, color=EDGE, linewidth=0.5)

# Annotate theory line
theory_line = np.degrees(delta_phi_theory) * (t_fit / T_orbit)
ax2.plot(t_fit / T_orbit, theory_line, ':', color=RED, linewidth=1.2,
         label=f'Theory: {np.degrees(delta_phi_theory):.4f} deg/orbit')

ax2.set_xlabel('Orbit number', color=GREY)
ax2.set_ylabel('Perihelion angle (deg)', color=GREY)
ax2.set_title('Perihelion precession (LRL vector angle)', color='white', pad=10)
ax2.legend(facecolor='#161b22', labelcolor='white', edgecolor=EDGE)
ax2.tick_params(colors=GREY)
for sp in ax2.spines.values():
    sp.set_edgecolor(EDGE)

plt.tight_layout()
out_path = os.path.join("output", "06_mercury_precession.png")
plt.savefig(out_path, dpi=150, facecolor=BG)
print(f"Plot saved -> {out_path}")
plt.show()
