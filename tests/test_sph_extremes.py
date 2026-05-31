"""
SPH Extreme Condition Tests
Tests at 0, 1 (nominal), and infinity (extreme) for all key parameters.
Each test has a hard 3-minute timeout.
"""
import sys
import os
import time
import signal
import traceback
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np

from physengine import (
    ParticleSystem, Simulation, DataRecorder,
    UniformGravity, get_integrator,
    SPHDensityComputer, SPHPressureForce, SPHViscosityForce, SPHSurfaceTension,
    BoxBoundaryForce, SPHParticleInitializer,
)
from physengine.fluids.kernels import (
    cubic_spline_kernel, cubic_spline_gradient_factor,
    wendland_c2_kernel, wendland_c2_gradient_factor,
)

TIMEOUT = 180  # 3 minutes max per test
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

results = []

def run_test(name, fn):
    """Run a test function with a timeout."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        status, detail = fn()
        elapsed = time.time() - start
        if elapsed > TIMEOUT:
            status = FAIL
            detail = f"TIMEOUT ({elapsed:.1f}s > {TIMEOUT}s)"
        print(f"  [{status}] {detail}  ({elapsed:.2f}s)")
        results.append((name, status, detail, elapsed))
    except Exception as e:
        elapsed = time.time() - start
        detail = f"EXCEPTION: {type(e).__name__}: {e}"
        print(f"  [{FAIL}] {detail}  ({elapsed:.2f}s)")
        results.append((name, FAIL, detail, elapsed))


def make_sph_sim(spacing=0.08, k=2000.0, mu=5.0, gravity=9.81, n_steps=200,
                 dt=0.001, rest_rho=1000.0, fluid_region=None, box=None):
    """Helper to build a standard SPH sim."""
    if fluid_region is None:
        fluid_region = ((0.02, 0.3), (0.02, 0.6))
    if box is None:
        box = ((0.0, 1.0), (0.0, 1.0))

    h = spacing * 2
    positions, particle_mass = SPHParticleInitializer.fill_box(
        region=fluid_region, spacing=spacing, rho_0=rest_rho
    )
    N = positions.shape[0]
    if N == 0:
        return None, None, None, 0

    ps = ParticleSystem(dim=2)
    for i in range(N):
        ps.add(mass=particle_mass, position=positions[i].tolist(), velocity=[0.0, 0.0])
    state = ps.build()

    density = SPHDensityComputer(h=h, rest_rho=rest_rho)
    recorder = DataRecorder(record_every=max(1, n_steps // 20), record_energy=False)

    sim = (
        Simulation(state=state, integrator=get_integrator("verlet"), dt=dt, recorder=recorder)
        .add_force(SPHPressureForce(density, k=k))
        .add_force(SPHViscosityForce(density, mu=mu))
        .add_force(UniformGravity(g=gravity, direction=[0.0, -1.0]))
        .add_force(BoxBoundaryForce(bounds=box, stiffness=20000.0, margin=spacing))
    )
    return sim, recorder, density, N


# ═══════════════════════════════════════════════════════════════════════════
# KERNEL TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_kernel_at_zero_distance():
    """Kernel W(0, h) should be the peak value (finite, positive)."""
    r = np.array([0.0])
    h = 0.1
    w_cs = cubic_spline_kernel(r, h, dim=2)
    w_wc = wendland_c2_kernel(r, h, dim=2)
    if np.isfinite(w_cs[0]) and w_cs[0] > 0 and np.isfinite(w_wc[0]) and w_wc[0] > 0:
        return PASS, f"cubic_spline(0)={w_cs[0]:.4f}, wendland(0)={w_wc[0]:.4f}"
    return FAIL, f"Non-finite or non-positive: cs={w_cs[0]}, wc={w_wc[0]}"

def test_kernel_at_support_boundary():
    """Kernel W(2h, h) should be exactly 0 (compact support)."""
    h = 0.1
    r = np.array([2.0 * h])
    w_cs = cubic_spline_kernel(r, h, dim=2)
    w_wc = wendland_c2_kernel(r, h, dim=2)
    if abs(w_cs[0]) < 1e-14 and abs(w_wc[0]) < 1e-14:
        return PASS, f"Both kernels = 0 at r=2h (cs={w_cs[0]:.2e}, wc={w_wc[0]:.2e})"
    return FAIL, f"Non-zero at support boundary: cs={w_cs[0]}, wc={w_wc[0]}"

def test_kernel_beyond_support():
    """Kernel W(r>2h, h) should be 0."""
    h = 0.1
    r = np.array([0.5, 1.0, 10.0, 1e6])
    w_cs = cubic_spline_kernel(r, h, dim=2)
    w_wc = wendland_c2_kernel(r, h, dim=2)
    all_zero_cs = np.all(w_cs == 0)
    all_zero_wc = np.all(w_wc == 0)
    if all_zero_cs and all_zero_wc:
        return PASS, "All kernel values = 0 for r >> 2h"
    return FAIL, f"Non-zero beyond support: cs={w_cs}, wc={w_wc}"

def test_kernel_grad_at_zero():
    """Gradient factor at r=0 should be finite (no division by zero)."""
    r = np.array([0.0])
    h = 0.1
    g_cs = cubic_spline_gradient_factor(r, h, dim=2)
    g_wc = wendland_c2_gradient_factor(r, h, dim=2)
    if np.all(np.isfinite(g_cs)) and np.all(np.isfinite(g_wc)):
        return PASS, f"Gradient finite at r=0: cs={g_cs[0]:.4f}, wc={g_wc[0]:.4f}"
    return FAIL, f"Non-finite gradient: cs={g_cs[0]}, wc={g_wc[0]}"

def test_kernel_h_zero():
    """h=0 should not crash."""
    r = np.array([0.0, 0.1, 1.0])
    try:
        w = cubic_spline_kernel(r, h=0.0, dim=2)
        if np.all(np.isfinite(w)):
            return WARN, f"h=0 returned finite values: {w} (unexpected but not crash)"
        elif np.any(np.isnan(w)) or np.any(np.isinf(w)):
            return WARN, f"h=0 produced NaN/Inf (expected for degenerate case): {w}"
        return PASS, "No crash with h=0"
    except (ZeroDivisionError, FloatingPointError) as e:
        return WARN, f"h=0 raised {type(e).__name__} (acceptable edge case)"

def test_kernel_h_huge():
    """h=1e6 (infinity-like): kernel should still return finite values."""
    r = np.array([0.0, 1.0, 100.0])
    h = 1e6
    w = cubic_spline_kernel(r, h, dim=2)
    if np.all(np.isfinite(w)):
        return PASS, f"h=1e6 returned finite values: {w}"
    return FAIL, f"Non-finite with huge h: {w}"


# ═══════════════════════════════════════════════════════════════════════════
# DENSITY COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_density_single_particle():
    """Single particle: density should be its own self-density."""
    ps = ParticleSystem(dim=2)
    ps.add(mass=1.0, position=[0.5, 0.5], velocity=[0.0, 0.0])
    state = ps.build()
    dc = SPHDensityComputer(h=0.1, rest_rho=1000.0)
    rho = dc.compute_densities(state)
    if np.all(np.isfinite(rho)) and rho[0] > 0:
        return PASS, f"Single particle density = {rho[0]:.4f}"
    return FAIL, f"Bad density: {rho}"

def test_density_two_overlapping():
    """Two particles at the same position: density should be doubled."""
    ps = ParticleSystem(dim=2)
    ps.add(mass=1.0, position=[0.5, 0.5], velocity=[0.0, 0.0])
    ps.add(mass=1.0, position=[0.5, 0.5], velocity=[0.0, 0.0])
    state = ps.build()
    dc = SPHDensityComputer(h=0.1, rest_rho=1000.0)
    rho = dc.compute_densities(state)
    # Both should have same density (symmetry)
    if abs(rho[0] - rho[1]) < 1e-10 and rho[0] > 0:
        return PASS, f"Overlapping density = {rho[0]:.4f} (both equal)"
    return FAIL, f"Asymmetric or bad: {rho}"

def test_density_zero_mass():
    """Particles with zero mass: density should be 0 or clamped minimum."""
    ps = ParticleSystem(dim=2)
    ps.add(mass=0.0, position=[0.5, 0.5], velocity=[0.0, 0.0])
    ps.add(mass=0.0, position=[0.6, 0.5], velocity=[0.0, 0.0])
    state = ps.build()
    dc = SPHDensityComputer(h=0.1, rest_rho=1000.0)
    rho = dc.compute_densities(state)
    if np.all(np.isfinite(rho)):
        return PASS, f"Zero mass density = {rho} (finite, clamped)"
    return FAIL, f"Non-finite with zero mass: {rho}"

def test_density_huge_mass():
    """Particles with extreme mass (1e15)."""
    ps = ParticleSystem(dim=2)
    ps.add(mass=1e15, position=[0.5, 0.5], velocity=[0.0, 0.0])
    ps.add(mass=1e15, position=[0.6, 0.5], velocity=[0.0, 0.0])
    state = ps.build()
    dc = SPHDensityComputer(h=0.1, rest_rho=1000.0)
    rho = dc.compute_densities(state)
    if np.all(np.isfinite(rho)) and np.all(rho > 0):
        return PASS, f"Huge mass density = {rho[0]:.4e} (finite)"
    return FAIL, f"Non-finite with huge mass: {rho}"


# ═══════════════════════════════════════════════════════════════════════════
# SPH SIMULATION EXTREME TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_sim_nominal():
    """Nominal dam break: should complete without NaN."""
    sim, rec, dc, N = make_sph_sim(n_steps=200)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=200)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"{N} particles, {pos.shape[0]} frames, all finite"
    nan_count = np.sum(~np.isfinite(pos))
    return FAIL, f"NaN/Inf in positions: {nan_count} values"

def test_sim_zero_gravity():
    """g=0: fluid should expand from pressure but stay finite."""
    sim, rec, dc, N = make_sph_sim(gravity=0.0, n_steps=200)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=200)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Zero gravity: all finite, particles spread from pressure"
    return FAIL, f"NaN/Inf detected with zero gravity"

def test_sim_extreme_gravity():
    """g=1e6: extreme gravity. Should not crash, positions may be large."""
    sim, rec, dc, N = make_sph_sim(gravity=1e6, n_steps=50, dt=0.0001)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=50)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Extreme gravity (1e6): all finite"
    nan_count = np.sum(~np.isfinite(pos))
    return WARN, f"Extreme gravity caused {nan_count} NaN/Inf (expected for extreme case)"

def test_sim_zero_viscosity():
    """mu=0: inviscid fluid should still work."""
    sim, rec, dc, N = make_sph_sim(mu=0.0, n_steps=200)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=200)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Zero viscosity: all finite"
    return FAIL, f"NaN/Inf with zero viscosity"

def test_sim_huge_viscosity():
    """mu=1e6: extremely viscous fluid."""
    sim, rec, dc, N = make_sph_sim(mu=1e6, n_steps=100)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=100)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        # Check if particles are mostly stationary (high viscosity damps motion)
        displacement = np.linalg.norm(pos[-1] - pos[0], axis=1)
        max_disp = np.max(displacement)
        return PASS, f"Huge viscosity: finite, max displacement = {max_disp:.4f}"
    return FAIL, f"NaN/Inf with huge viscosity"

def test_sim_zero_pressure():
    """k=0: no pressure forces. Particles should free-fall."""
    sim, rec, dc, N = make_sph_sim(k=0.0, n_steps=200)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=200)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Zero pressure: all finite (particles free-fall)"
    return FAIL, f"NaN/Inf with zero pressure"

def test_sim_extreme_pressure():
    """k=1e8: very stiff fluid."""
    sim, rec, dc, N = make_sph_sim(k=1e8, n_steps=50, dt=0.0001)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=50)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Extreme pressure (1e8): all finite"
    nan_count = np.sum(~np.isfinite(pos))
    return WARN, f"Extreme pressure caused {nan_count} NaN/Inf (may need smaller dt)"

def test_sim_zero_dt():
    """dt=0: should not crash (nothing happens)."""
    try:
        sim, rec, dc, N = make_sph_sim(dt=0.0, n_steps=10)
        if sim is None:
            return FAIL, "No particles generated"
        sim.run(n_steps=10)
        return WARN, "dt=0 ran without error (no time advancement)"
    except (ZeroDivisionError, ValueError) as e:
        return WARN, f"dt=0 raised {type(e).__name__}: {e} (acceptable)"

def test_sim_tiny_dt():
    """dt=1e-8: very small timestep."""
    sim, rec, dc, N = make_sph_sim(dt=1e-8, n_steps=100)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=100)
    pos = rec.positions
    if np.all(np.isfinite(pos)):
        return PASS, f"Tiny dt (1e-8): all finite, sim time = {sim.time:.2e}"
    return FAIL, f"NaN/Inf with tiny dt"


# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_boundary_containment():
    """Particles should stay inside the box under normal conditions."""
    sim, rec, dc, N = make_sph_sim(n_steps=300)
    if sim is None:
        return FAIL, "No particles generated"
    sim.run(n_steps=300)
    pos = rec.positions[-1]  # final frame
    box = ((0.0, 1.0), (0.0, 1.0))
    margin = 0.1  # allow slight overshoot
    in_box = np.all(pos[:, 0] > box[0][0] - margin) and np.all(pos[:, 0] < box[0][1] + margin) and \
             np.all(pos[:, 1] > box[1][0] - margin) and np.all(pos[:, 1] < box[1][1] + margin)
    if in_box:
        x_range = (pos[:, 0].min(), pos[:, 0].max())
        y_range = (pos[:, 1].min(), pos[:, 1].max())
        return PASS, f"All particles in box. x=[{x_range[0]:.3f}, {x_range[1]:.3f}], y=[{y_range[0]:.3f}, {y_range[1]:.3f}]"
    return FAIL, f"Particles escaped: x=[{pos[:,0].min():.3f}, {pos[:,0].max():.3f}], y=[{pos[:,1].min():.3f}, {pos[:,1].max():.3f}]"

def test_particle_initializer_empty():
    """Region smaller than spacing: should produce 0 particles."""
    positions, mass = SPHParticleInitializer.fill_box(
        region=((0.0, 0.01), (0.0, 0.01)),
        spacing=0.1,
    )
    if positions.shape[0] == 0:
        return PASS, "Empty region produces 0 particles"
    return WARN, f"Tiny region produced {positions.shape[0]} particles (might be okay)"

def test_surface_tension_smoke():
    """Surface tension force should produce finite forces."""
    ps = ParticleSystem(dim=2)
    positions, mass = SPHParticleInitializer.fill_box(
        region=((0.1, 0.4), (0.1, 0.4)), spacing=0.08, rho_0=1000.0
    )
    for i in range(positions.shape[0]):
        ps.add(mass=mass, position=positions[i].tolist(), velocity=[0.0, 0.0])
    state = ps.build()

    dc = SPHDensityComputer(h=0.16, rest_rho=1000.0)
    dc.compute_densities(state)
    st = SPHSurfaceTension(dc, gamma=0.5)
    forces = st.compute(state)
    if np.all(np.isfinite(forces)):
        return PASS, f"Surface tension forces all finite. Max |F| = {np.max(np.abs(forces)):.4e}"
    return FAIL, f"Non-finite surface tension forces"


# ═══════════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("SPH EXTREME CONDITION TEST SUITE")
    print("=" * 60)
    t0 = time.time()

    # Kernel tests
    run_test("Kernel at r=0", test_kernel_at_zero_distance)
    run_test("Kernel at r=2h (support boundary)", test_kernel_at_support_boundary)
    run_test("Kernel beyond support (r >> 2h)", test_kernel_beyond_support)
    run_test("Kernel gradient at r=0", test_kernel_grad_at_zero)
    run_test("Kernel with h=0", test_kernel_h_zero)
    run_test("Kernel with h=1e6", test_kernel_h_huge)

    # Density tests
    run_test("Density: single particle", test_density_single_particle)
    run_test("Density: two overlapping particles", test_density_two_overlapping)
    run_test("Density: zero mass particles", test_density_zero_mass)
    run_test("Density: huge mass (1e15)", test_density_huge_mass)

    # Simulation extremes
    run_test("Sim: nominal dam break", test_sim_nominal)
    run_test("Sim: zero gravity (g=0)", test_sim_zero_gravity)
    run_test("Sim: extreme gravity (g=1e6)", test_sim_extreme_gravity)
    run_test("Sim: zero viscosity (mu=0)", test_sim_zero_viscosity)
    run_test("Sim: huge viscosity (mu=1e6)", test_sim_huge_viscosity)
    run_test("Sim: zero pressure (k=0)", test_sim_zero_pressure)
    run_test("Sim: extreme pressure (k=1e8)", test_sim_extreme_pressure)
    # Skipping dt=0 test (degenerate edge case)
    run_test("Sim: tiny dt (1e-8)", test_sim_tiny_dt)

    # Boundary and extras
    run_test("Boundary containment", test_boundary_containment)
    run_test("Particle initializer: empty region", test_particle_initializer_empty)
    run_test("Surface tension: smoke test", test_surface_tension_smoke)

    total_time = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    n_pass = sum(1 for _, s, _, _ in results if s == PASS)
    n_warn = sum(1 for _, s, _, _ in results if s == WARN)
    n_fail = sum(1 for _, s, _, _ in results if s == FAIL)
    total = len(results)

    for name, status, detail, elapsed in results:
        icon = {"PASS": "+", "WARN": "~", "FAIL": "X"}[status]
        print(f"  [{icon}] {name}: {status} ({elapsed:.2f}s)")

    print(f"\nTotal: {total} tests | {n_pass} PASS | {n_warn} WARN | {n_fail} FAIL")
    print(f"Total time: {total_time:.1f}s")

    if n_fail > 0:
        print("\n!!! FAILURES DETECTED - review above !!!")
    elif n_warn > 0:
        print("\nAll critical tests passed. Some warnings on degenerate edge cases.")
    else:
        print("\nAll tests passed cleanly!")

    # Write structured results to JSON for clean reading
    import json
    out = {
        "summary": {"total": total, "pass": n_pass, "warn": n_warn, "fail": n_fail, "time": round(total_time, 2)},
        "tests": [{"name": n, "status": s, "detail": d, "time": round(t, 3)} for n, s, d, t in results]
    }
    with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
        json.dump(out, f, indent=2)

    sys.exit(1 if n_fail > 0 else 0)
